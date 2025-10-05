package pinecone

import (
	"context"
	"fmt"
	"os"
	"sync"

	"github.com/pinecone-io/go-pinecone/pinecone"

	"google.golang.org/protobuf/types/known/structpb"
)

var (
	// client is the singleton Pinecone client
	client *pinecone.Client
	once   sync.Once

	// baseIndexInstance is the singleton index instance for the base index
	baseIndexInstance *indexOperations
	baseSetup         sync.Once
)

// NewPineconeService creates a new Pinecone service instance using the official SDK
func NewPineconeService() *pineconeService {
	once.Do(func() {
		apiKey := os.Getenv("PINECONE_API_KEY")

		pcClient, err := pinecone.NewClient(pinecone.NewClientParams{
			ApiKey: apiKey,
		})
		if err != nil {
			panic("Failed to initialize Pinecone client: " + err.Error())
		}
		client = pcClient
	})

	return &pineconeService{
		client: client,
	}
}

// ForBaseIndex returns an index gateway for the base index
func (ps *pineconeService) ForBaseIndex(namespace string) *indexOperations {
	baseSetup.Do(func() {
		host := os.Getenv("PINECONE_BASE_HOST")
		indexConnection, err := ps.client.Index(pinecone.NewIndexConnParams{
			Host:      host,
			Namespace: namespace,
		})
		if err != nil {
			panic("Failed to initialize Pinecone client: " + err.Error())
		}

		baseIndexInstance = &indexOperations{
			index: indexConnection,
		}
	})

	return baseIndexInstance
}

// FindById finds a vector by its ID
func (idx *indexOperations) FindById(ctx context.Context, id string) (*pinecone.Vector, error) {
	vector, err := idx.index.QueryByVectorId(ctx, &pinecone.QueryByVectorIdRequest{
		VectorId:        id,
		TopK:            1,
		IncludeMetadata: true,
	})
	if err != nil {
		return nil, err
	}

	if len(vector.Matches) == 0 {
		return nil, fmt.Errorf("vector not found")
	}

	return vector.Matches[0].Vector, nil
}

// Search performs a vector similarity search in the index
func (idx *indexOperations) Search(ctx context.Context, queryVector []float32, topK int, filter map[string]any, includeMetadata bool) (_ []QueryMatch, finalErr error) {
	// Convert filter to Pinecone's MetadataFilter format
	metadataFilter, err := structpb.NewStruct(filter)
	if err != nil {
		return nil, fmt.Errorf("failed to create metadata map: %v", err)
	}

	queryRequest := &pinecone.QueryByVectorValuesRequest{
		Vector:          queryVector,
		TopK:            uint32(topK),
		IncludeValues:   false,
		IncludeMetadata: includeMetadata,
		MetadataFilter:  metadataFilter,
	}

	queryResponse, err := idx.index.QueryByVectorValues(ctx, queryRequest)
	if err != nil {
		return nil, err
	}

	// Convert matches to our expected format
	matches := make([]QueryMatch, len(queryResponse.Matches))
	for i, match := range queryResponse.Matches {
		matches[i] = *match
	}

	return matches, nil
}

// Upsert stores vectors in the index
func (idx *indexOperations) Upsert(ctx context.Context, vectors []Vector) error {
	// Convert our Vector type to Pinecone's *Vector type
	pineconeVectors := make([]*pinecone.Vector, len(vectors))
	for i, v := range vectors {
		pineconeVectors[i] = &v
	}

	_, err := idx.index.UpsertVectors(ctx, pineconeVectors)
	return err
}

// UpdateMetadata updates the metadata for a vector
func (idx *indexOperations) UpdateMetadata(ctx context.Context, vectorID string, metadata *pinecone.Metadata) error {
	return idx.index.UpdateVector(ctx, &pinecone.UpdateVectorRequest{
		Id:       vectorID,
		Metadata: metadata,
	})
}

// Delete removes vectors from the index
func (idx *indexOperations) Delete(ctx context.Context, ids []string) error {
	return idx.index.DeleteVectorsById(ctx, ids)
}
