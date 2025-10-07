package benchmark

import (
	"context"
	"time"

	"github.com/FrenchMajesty/consistent-classifier/clients/pinecone"
	"github.com/FrenchMajesty/consistent-classifier/clients/voyage"
	"github.com/austinfhunter/voyageai"
)

// IndexOperationsInterface defines the interface for Pinecone index operations
type IndexOperationsInterface interface {
	Search(ctx context.Context, queryVector []float32, topK int, filter map[string]any, includeMetadata bool) ([]pinecone.QueryMatch, error)
	Upsert(ctx context.Context, vectors []pinecone.Vector) error
	Delete(ctx context.Context, ids []string) error
	FindById(ctx context.Context, id string) (*pinecone.Vector, error)
	UpdateMetadata(ctx context.Context, vectorID string, metadata *pinecone.Metadata) error
}

type EmbeddingInterface interface {
	GenerateEmbedding(ctx context.Context, text string, embeddingType voyage.VoyageEmbeddingType) ([]float32, error)
	GenerateEmbeddings(ctx context.Context, texts []string, embeddingType voyage.VoyageEmbeddingType) ([]voyageai.EmbeddingObject, error)
}

type BenchmarkMetrics struct {
	// Overall metrics
	TotalDuration   time.Duration
	TotalTweets     int
	UniqueLabels    int
	ConvergedLabels int

	// Vector operations metrics
	VectorReads     int
	VectorWrites    int
	VectorReplyHits int
	VectorLabelHits int

	// Per-tweet metrics (old - kept for backwards compatibility)
	ProcessingTime []time.Duration
	TokenUsage     []TokenUsageMetrics

	// Production-representative metrics (user-facing latency)
	UserFacingLatency []time.Duration // Time user waits (vector search + LLM if miss)
	BackgroundTime    []time.Duration // Time for async work (label clustering + DSU + upserts)
	CacheHit          []bool          // Whether each request was a cache hit
}

type TokenUsageMetrics struct {
	InputTokens       int
	CachedInputTokens int
	OutputTokens      int
}

type DatasetItem struct {
	Content      string
	UserResponse string
	UserCategory string
}

type Result struct {
	Post       string
	Reply      string
	ReplyLabel string
}

type VectorHit struct {
	Score      float32
	VectorText string
}

type ContentVectorHit struct {
	*VectorHit
	Label string
}

type LabelVectorHit struct {
	*VectorHit
	Root string
}
