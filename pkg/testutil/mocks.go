package testutil

import (
	"context"
	"sync"

	"github.com/FrenchMajesty/consistent-classifier/pkg/types"
	"github.com/FrenchMajesty/consistent-classifier/utils/disjoint_set"
)

// MockEmbeddingClient is a mock implementation of EmbeddingClient for testing
type MockEmbeddingClient struct {
	GenerateEmbeddingFunc func(ctx context.Context, text string) ([]float32, error)
	mu                    sync.Mutex
	CallCount             int
	LastText              string
}

func (m *MockEmbeddingClient) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	m.mu.Lock()
	m.CallCount++
	m.LastText = text
	m.mu.Unlock()

	if m.GenerateEmbeddingFunc != nil {
		return m.GenerateEmbeddingFunc(ctx, text)
	}
	// Default: return a simple embedding based on text length
	embedding := make([]float32, 10)
	for i := range embedding {
		embedding[i] = float32(len(text)) / 100.0
	}
	return embedding, nil
}

// MockVectorClient is a mock implementation of VectorClient for testing
type MockVectorClient struct {
	SearchFunc func(ctx context.Context, vector []float32, topK int) ([]types.VectorMatch, error)
	UpsertFunc func(ctx context.Context, id string, vector []float32, metadata map[string]any) error

	mu          sync.Mutex
	CallCount   int
	UpsertCount int
	Storage     map[string]struct {
		Vector   []float32
		Metadata map[string]any
	}
}

func NewMockVectorClient() *MockVectorClient {
	return &MockVectorClient{
		Storage: make(map[string]struct {
			Vector   []float32
			Metadata map[string]any
		}),
	}
}

func (m *MockVectorClient) Search(ctx context.Context, vector []float32, topK int) ([]types.VectorMatch, error) {
	m.mu.Lock()
	m.CallCount++
	m.mu.Unlock()

	if m.SearchFunc != nil {
		return m.SearchFunc(ctx, vector, topK)
	}

	// Default: return empty results (cache miss)
	return []types.VectorMatch{}, nil
}

func (m *MockVectorClient) Upsert(ctx context.Context, id string, vector []float32, metadata map[string]any) error {
	m.mu.Lock()
	m.UpsertCount++
	m.Storage[id] = struct {
		Vector   []float32
		Metadata map[string]any
	}{Vector: vector, Metadata: metadata}
	m.mu.Unlock()

	if m.UpsertFunc != nil {
		return m.UpsertFunc(ctx, id, vector, metadata)
	}

	return nil
}

// MockLLMClient is a mock implementation of LLMClient for testing
type MockLLMClient struct {
	ClassifyFunc func(ctx context.Context, text string) (string, error)

	mu        sync.Mutex
	CallCount int
	LastText  string
}

func (m *MockLLMClient) Classify(ctx context.Context, text string) (string, error) {
	m.mu.Lock()
	m.CallCount++
	m.LastText = text
	m.mu.Unlock()

	if m.ClassifyFunc != nil {
		return m.ClassifyFunc(ctx, text)
	}

	// Default: return a simple label based on text
	if len(text) > 50 {
		return "long_text", nil
	}
	return "short_text", nil
}

// MockDSUPersistence is a mock implementation of DisjointSetPersistence for testing
type MockDSUPersistence struct {
	LoadFunc func() (*disjoint_set.DSU, error)
	SaveFunc func(dsu *disjoint_set.DSU) error

	mu        sync.Mutex
	SaveCount int
	LastDSU   *disjoint_set.DSU
}

func (m *MockDSUPersistence) Load() (*disjoint_set.DSU, error) {
	if m.LoadFunc != nil {
		return m.LoadFunc()
	}

	// Default: return empty DSU
	return disjoint_set.NewDSU(), nil
}

func (m *MockDSUPersistence) Save(dsu *disjoint_set.DSU) error {
	m.mu.Lock()
	m.SaveCount++
	m.LastDSU = dsu
	m.mu.Unlock()

	if m.SaveFunc != nil {
		return m.SaveFunc(dsu)
	}

	return nil
}
