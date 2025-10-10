package pinecone

import (
	"github.com/pinecone-io/go-pinecone/pinecone"
)

// pineconeService provides interface to Pinecone vector database with ForIndex API
type pineconeService struct {
	client *pinecone.Client
}

// IndexOperations provides operations for a specific Pinecone index
type indexOperations struct {
	index *pinecone.IndexConnection
}

// Vector represents a vector with metadata (re-exported from SDK for convenience)
type Vector = pinecone.Vector

// QueryMatch represents a match from query results (re-exported from SDK for convenience)
type QueryMatch = pinecone.ScoredVector

// Metadata represents the metadata for a vector (re-exported from SDK for convenience)
type Metadata = pinecone.Metadata
