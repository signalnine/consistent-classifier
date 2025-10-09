package classifier

import "time"

// Result represents the classification result
type Result struct {
	// Label is the classification category assigned to the text
	Label string

	// CacheHit indicates whether the classification was retrieved from the vector cache
	CacheHit bool

	// Confidence is the similarity score if cache hit, 0 otherwise
	Confidence float32

	// UserFacingLatency is the time the user waited for the classification
	UserFacingLatency time.Duration

	// BackgroundLatency is the time spent on background tasks (clustering, vector upserts)
	// This is 0 if cache hit, since no background work is needed
	BackgroundLatency time.Duration
}

// Metrics provides statistics about the classifier's state
type Metrics struct {
	// UniqueLabels is the total number of unique labels seen
	UniqueLabels int

	// ConvergedLabels is the number of distinct label clusters after DSU merging
	ConvergedLabels int

	// CacheHitRate is the percentage of classifications served from cache
	CacheHitRate float32
}
