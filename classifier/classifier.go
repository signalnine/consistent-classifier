package classifier

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/FrenchMajesty/consistent-classifier/utils/disjoint_set"
	"github.com/google/uuid"
)

// Classifier performs text classification with vector caching and label clustering
type Classifier struct {
	embedding     EmbeddingClient
	vectorContent VectorClient
	vectorLabel   VectorClient
	llm           LLMClient
	dsu           *disjoint_set.DSU
	dsuPersist    DisjointSetPersistence
	minSimilarity float32

	// Metrics tracking
	totalClassifications int
	cacheHits            int
	metricsLock          sync.RWMutex
}

// NewClassifier creates a new Classifier with the given configuration
func NewClassifier(cfg Config) (*Classifier, error) {
	cfg.applyDefaults()

	// Validate required clients are provided
	if cfg.EmbeddingClient == nil {
		return nil, fmt.Errorf("EmbeddingClient is required")
	}
	if cfg.VectorClient == nil {
		return nil, fmt.Errorf("VectorClient is required")
	}
	if cfg.LLMClient == nil {
		return nil, fmt.Errorf("LLMClient is required")
	}

	// Initialize DSU persistence (only field with a default)
	var dsuPersist DisjointSetPersistence
	if cfg.DSUPersistence != nil {
		dsuPersist = cfg.DSUPersistence
	} else {
		dsuPersist = NewFileDSUPersistence(DefaultDSUFilePath)
	}

	// Load DSU from persistence
	dsu, err := dsuPersist.Load()
	if err != nil {
		return nil, fmt.Errorf("failed to load DSU: %w", err)
	}

	return &Classifier{
		embedding:     cfg.EmbeddingClient,
		vectorContent: cfg.VectorClient,
		vectorLabel:   cfg.VectorClient, // Same client used for both content and label vectors
		llm:           cfg.LLMClient,
		dsu:           dsu,
		dsuPersist:    dsuPersist,
		minSimilarity: cfg.MinSimilarity,
	}, nil
}

// Classify classifies the given text and returns the classification result
func (c *Classifier) Classify(ctx context.Context, text string) (*Result, error) {
	userFacingStart := time.Now()

	// Step 1: Generate embedding for this text
	embedding, err := c.embedding.GenerateEmbedding(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Step 2: Search vector cache for similar text
	matches, err := c.vectorContent.Search(ctx, embedding, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to search vector cache: %w", err)
	}

	// Check if we have a cache hit
	if len(matches) > 0 && matches[0].Score >= c.minSimilarity {
		// Cache HIT - return cached label
		userFacingLatency := time.Since(userFacingStart)

		label, ok := matches[0].Metadata["label"].(string)
		if !ok {
			return nil, fmt.Errorf("cached vector missing label metadata")
		}

		c.recordCacheHit()

		return &Result{
			Label:             label,
			CacheHit:          true,
			Confidence:        matches[0].Score,
			UserFacingLatency: userFacingLatency,
			BackgroundLatency: 0,
		}, nil
	}

	// Cache MISS - call LLM for classification
	label, err := c.llm.Classify(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("failed to classify with LLM: %w", err)
	}

	userFacingLatency := time.Since(userFacingStart)
	c.recordClassification()

	// Background processing - run asynchronously but wait for completion
	backgroundStart := time.Now()
	err = c.processBackgroundTasks(ctx, text, embedding, label)
	if err != nil {
		// Don't fail the classification, just log the error
		// In production you might want to handle this differently
		fmt.Printf("Warning: background processing failed: %v\n", err)
	}
	backgroundLatency := time.Since(backgroundStart)

	return &Result{
		Label:             label,
		CacheHit:          false,
		Confidence:        0,
		UserFacingLatency: userFacingLatency,
		BackgroundLatency: backgroundLatency,
	}, nil
}

// processBackgroundTasks handles label clustering and vector caching
func (c *Classifier) processBackgroundTasks(ctx context.Context, text string, embedding []float32, label string) error {
	var wg sync.WaitGroup
	errChan := make(chan error, 3)

	// Task 1: Find similar labels and update DSU
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := c.updateLabelClustering(ctx, label); err != nil {
			errChan <- fmt.Errorf("label clustering failed: %w", err)
		}
	}()

	// Task 2: Cache the text embedding for future lookups
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := c.cacheTextEmbedding(ctx, text, embedding, label); err != nil {
			errChan <- fmt.Errorf("text caching failed: %w", err)
		}
	}()

	// Task 3: Cache the label embedding
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := c.cacheLabelEmbedding(ctx, label); err != nil {
			errChan <- fmt.Errorf("label caching failed: %w", err)
		}
	}()

	wg.Wait()
	close(errChan)

	// Return first error if any
	for err := range errChan {
		if err != nil {
			return err
		}
	}

	return nil
}

// updateLabelClustering finds similar labels and merges them in the DSU
func (c *Classifier) updateLabelClustering(ctx context.Context, label string) error {
	// Generate embedding for the label
	labelEmbedding, err := c.embedding.GenerateEmbedding(ctx, label)
	if err != nil {
		return err
	}

	// Search for similar labels
	matches, err := c.vectorLabel.Search(ctx, labelEmbedding, 1)
	if err != nil {
		return err
	}

	// Find the root label
	rootLabel := label
	if len(matches) > 0 && matches[0].Score >= c.minSimilarity {
		root, ok := matches[0].Metadata["root"].(string)
		if ok {
			rootLabel = root
		}
	}

	// Union the label with the root label in DSU
	c.dsu.Union(c.dsu.FindOrCreate(rootLabel), c.dsu.FindOrCreate(label))

	return nil
}

// cacheTextEmbedding stores the text embedding in the vector database
func (c *Classifier) cacheTextEmbedding(ctx context.Context, text string, embedding []float32, label string) error {
	id := uuid.New().String()
	metadata := map[string]any{
		"vector_text": text,
		"label":       label,
	}
	return c.vectorContent.Upsert(ctx, id, embedding, metadata)
}

// cacheLabelEmbedding stores the label embedding in the vector database
func (c *Classifier) cacheLabelEmbedding(ctx context.Context, label string) error {
	// Generate embedding for the label
	labelEmbedding, err := c.embedding.GenerateEmbedding(ctx, label)
	if err != nil {
		return err
	}

	// Find root label from DSU
	rootIdx := c.dsu.FindOrCreate(label)
	rootLabel := c.dsu.FindLabel(rootIdx)
	if rootLabel == "" {
		rootLabel = label
	}

	metadata := map[string]any{
		"vector_text": label,
		"label":       label,
		"root":        rootLabel,
	}
	return c.vectorLabel.Upsert(ctx, label, labelEmbedding, metadata)
}

// SaveDSU saves the current DSU state to persistent storage
func (c *Classifier) SaveDSU() error {
	return c.dsuPersist.Save(c.dsu)
}

// GetMetrics returns current classification metrics
func (c *Classifier) GetMetrics() Metrics {
	c.metricsLock.RLock()
	defer c.metricsLock.RUnlock()

	var cacheHitRate float32
	if c.totalClassifications > 0 {
		cacheHitRate = float32(c.cacheHits) / float32(c.totalClassifications) * 100
	}

	return Metrics{
		UniqueLabels:    c.dsu.Size(),
		ConvergedLabels: c.dsu.CountSets(),
		CacheHitRate:    cacheHitRate,
	}
}

// recordCacheHit records a cache hit for metrics
func (c *Classifier) recordCacheHit() {
	c.metricsLock.Lock()
	defer c.metricsLock.Unlock()
	c.totalClassifications++
	c.cacheHits++
}

// recordClassification records a classification (cache miss) for metrics
func (c *Classifier) recordClassification() {
	c.metricsLock.Lock()
	defer c.metricsLock.Unlock()
	c.totalClassifications++
}
