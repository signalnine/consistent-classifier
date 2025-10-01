package voyage

import (
	"context"
	"fmt"
	"os"
	"sync"

	"github.com/austinfhunter/voyageai"
)

var client *voyageai.VoyageClient
var once sync.Once

const EMBEDDING_DIMENSIONS = 1024

const VOYAGEAI_EMBEDDING_MODEL = "voyage-3.5-lite"

type VoyageEmbeddingType string

const (
	VoyageEmbeddingTypeDocument VoyageEmbeddingType = "document"
	VoyageEmbeddingTypeQuery    VoyageEmbeddingType = "query"
)

// embeddingService handles generating embeddings for text
type voyageService struct {
}

// NewEmbeddingService creates a new embedding service
func NewEmbeddingService() *voyageService {
	once.Do(func() {
		apiKey := os.Getenv("VOYAGEAI_API_KEY")

		client = voyageai.NewClient(&voyageai.VoyageClientOpts{
			Key: apiKey,
		})
	})

	return &voyageService{}
}

// GenerateEmbedding generates an embedding for a single text using VoyageAI
func (es *voyageService) GenerateEmbedding(ctx context.Context, text string, embeddingType VoyageEmbeddingType) ([]float32, error) {
	dimensions := es.GetEmbeddingDimensions()
	inputType := string(embeddingType)
	embeddings, err := client.Embed(
		[]string{text},
		VOYAGEAI_EMBEDDING_MODEL,
		&voyageai.EmbeddingRequestOpts{
			InputType:       &inputType,
			OutputDimension: &dimensions,
		},
	)

	if err != nil {
		return nil, fmt.Errorf("could not get embedding: %w", err)
	}

	return embeddings.Data[0].Embedding, nil
}

// GetEmbeddingDimensions returns the dimension count for the embedding model
func (es *voyageService) GetEmbeddingDimensions() int {
	return EMBEDDING_DIMENSIONS
}
