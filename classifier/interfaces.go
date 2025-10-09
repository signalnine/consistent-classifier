package classifier

import (
	"context"

	"github.com/FrenchMajesty/consistent-classifier/utils/disjoint_set"
)

// EmbeddingClient generates vector embeddings for text
type EmbeddingClient interface {
	GenerateEmbedding(ctx context.Context, text string) ([]float32, error)
}

// VectorMatch represents a single match from a vector search
type VectorMatch struct {
	ID       string
	Score    float32
	Metadata map[string]any
}

// VectorClient performs vector similarity search and storage operations
type VectorClient interface {
	Search(ctx context.Context, vector []float32, topK int) ([]VectorMatch, error)
	Upsert(ctx context.Context, id string, vector []float32, metadata map[string]any) error
}

// LLMClient classifies text into category labels
type LLMClient interface {
	Classify(ctx context.Context, text string) (string, error)
}

// DisjointSetPersistence handles loading and saving the Disjoint Set Union structure
type DisjointSetPersistence interface {
	Load() (*disjoint_set.DSU, error)
	Save(dsu *disjoint_set.DSU) error
}
