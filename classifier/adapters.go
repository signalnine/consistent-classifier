package classifier

import (
	"context"
	"os"

	"github.com/FrenchMajesty/consistent-classifier/clients/pinecone"
	"github.com/FrenchMajesty/consistent-classifier/clients/voyage"
	"google.golang.org/protobuf/types/known/structpb"
)

// VoyageEmbeddingAdapter adapts the Voyage client to the EmbeddingClient interface
type VoyageEmbeddingAdapter struct {
	client interface {
		GenerateEmbedding(ctx context.Context, text string, embeddingType voyage.VoyageEmbeddingType) ([]float32, error)
	}
}

// NewVoyageEmbeddingAdapter creates a new adapter for Voyage AI
func NewVoyageEmbeddingAdapter(apiKey *string) *VoyageEmbeddingAdapter {
	loadEnvVar(apiKey, "VOYAGEAI_API_KEY")

	return &VoyageEmbeddingAdapter{
		client: voyage.NewEmbeddingService(*apiKey),
	}
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
func NewPineconeVectorAdapter(apiKey *string, host *string, namespace string) *PineconeVectorAdapter {
	loadEnvVar(apiKey, "PINECONE_API_KEY")
	loadEnvVar(host, "PINECONE_HOST")

	client := pinecone.NewPineconeService(*apiKey)
	index := client.ForBaseIndex(*host, namespace)
	return &PineconeVectorAdapter{
		index: index,
	}
}

// Search implements VectorClient interface
func (a *PineconeVectorAdapter) Search(ctx context.Context, vector []float32, topK int) ([]VectorMatch, error) {
	matches, err := a.index.Search(ctx, vector, topK, nil, true)
	if err != nil {
		return nil, err
	}

	// Convert Pinecone matches to our VectorMatch type
	results := make([]VectorMatch, len(matches))
	for i, match := range matches {
		metadata := make(map[string]any)
		if match.Vector != nil && match.Vector.Metadata != nil {
			metadata = match.Vector.Metadata.AsMap()
		}

		results[i] = VectorMatch{
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
func loadEnvVar(target *string, envKey string) {
	if target == nil {
		envVar := os.Getenv(envKey)
		if envVar == "" {
			panic(envKey + " environment variable not set and no " + envKey + " provided")
		}
		target = &envVar
	}
}
