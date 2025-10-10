package types

// VectorMatch represents a single match from a vector search
type VectorMatch struct {
	ID       string
	Score    float32
	Metadata map[string]any
}
