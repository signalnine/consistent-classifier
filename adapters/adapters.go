package adapters

import (
	"context"
	"fmt"
	"os"

	"github.com/FrenchMajesty/consistent-classifier/adapters/pinecone"
	"github.com/FrenchMajesty/consistent-classifier/adapters/voyage"
	"github.com/FrenchMajesty/consistent-classifier/types"
	"google.golang.org/protobuf/types/known/structpb"
)

// VoyageEmbeddingAdapter adapts the Voyage client to the EmbeddingClient interface
type VoyageEmbeddingAdapter struct {
	client interface {
		GenerateEmbedding(ctx context.Context, text string, embeddingType voyage.VoyageEmbeddingType) ([]float32, error)
	}
}

// NewVoyageEmbeddingAdapter creates a new adapter for Voyage AI
func NewVoyageEmbeddingAdapter(apiKey *string) (*VoyageEmbeddingAdapter, error) {
	key, err := loadEnvVar(apiKey, "VOYAGEAI_API_KEY")
	if err != nil {
		return nil, err
	}

	return &VoyageEmbeddingAdapter{
		client: voyage.NewEmbeddingService(*key),
	}, nil
}

// GenerateEmbedding implements EmbeddingClient interface
func (a *VoyageEmbeddingAdapter) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	return a.client.GenerateEmbedding(ctx, text, voyage.VoyageEmbeddingTypeDefault)
}

// PineconeVectorAdapter adapts the Pinecone client to the VectorClient interface
type PineconeVectorAdapter struct {
	index interface {
		Search(ctx context.Context, queryVector []float32, topK int, filter map[string]any, includeMetadata bool) ([]pinecone.QueryMatch, error)
		Upsert(ctx context.Context, vectors []pinecone.Vector) error
	}
}

// NewPineconeVectorAdapter creates a new adapter for Pinecone
func NewPineconeVectorAdapter(apiKey *string, host *string, namespace string) (*PineconeVectorAdapter, error) {
	key, err := loadEnvVar(apiKey, "PINECONE_API_KEY")
	if err != nil {
		return nil, err
	}

	h, err := loadEnvVar(host, "PINECONE_HOST")
	if err != nil {
		return nil, err
	}

	client, err := pinecone.NewPineconeService(*key)
	if err != nil {
		return nil, fmt.Errorf("failed to create pinecone service: %w", err)
	}

	index, err := client.ForBaseIndex(*h, namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to pinecone index: %w", err)
	}

	return &PineconeVectorAdapter{
		index: index,
	}, nil
}

// Search implements VectorClient interface
func (a *PineconeVectorAdapter) Search(ctx context.Context, vector []float32, topK int) ([]types.VectorMatch, error) {
	matches, err := a.index.Search(ctx, vector, topK, nil, true)
	if err != nil {
		return nil, err
	}

	// Convert Pinecone matches to our VectorMatch type
	results := make([]types.VectorMatch, len(matches))
	for i, match := range matches {
		metadata := make(map[string]any)
		if match.Vector != nil && match.Vector.Metadata != nil {
			metadata = match.Vector.Metadata.AsMap()
		}

		results[i] = types.VectorMatch{
			ID:       match.Vector.Id,
			Score:    match.Score,
			Metadata: metadata,
		}
	}

	return results, nil
}

// Upsert implements VectorClient interface
func (a *PineconeVectorAdapter) Upsert(ctx context.Context, id string, vector []float32, metadata map[string]any) error {
	// Convert metadata to structpb format
	metadataStruct, err := structpb.NewStruct(metadata)
	if err != nil {
		return err
	}

	vectors := []pinecone.Vector{
		{
			Id:     id,
			Values: vector,
			Metadata: &pinecone.Metadata{
				Fields: metadataStruct.Fields,
			},
		},
	}

	return a.index.Upsert(ctx, vectors)
}

// loadEnvVar loads an environment variable into a pointer if no value is provided
func loadEnvVar(target *string, envKey string) (*string, error) {
	if target == nil {
		envVar := os.Getenv(envKey)
		if envVar == "" {
			return nil, fmt.Errorf("%s environment variable not set and no value provided", envKey)
		}
		return &envVar, nil
	}
	return target, nil
}
