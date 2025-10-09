package classifier_test

import (
	"context"
	"fmt"
	"log"

	"github.com/FrenchMajesty/consistent-classifier/classifier"
)

// Example shows basic usage of the classifier
func Example_basic() {
	// Create classifier - no clients provided, rely on defaults with environment variables
	clf, err := classifier.NewClassifier(classifier.Config{})
	if err != nil {
		log.Fatal(err)
	}

	// Classify some text
	result, err := clf.Classify(context.Background(), "Thanks for the help!")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Label: %s\n", result.Label)
	fmt.Printf("Cache Hit: %v\n", result.CacheHit)
	fmt.Printf("Latency: %v\n", result.UserFacingLatency)

	// Save DSU state
	if err := clf.SaveDSU(); err != nil {
		log.Fatal(err)
	}
}

// Example shows customizing the configuration
func Example_customConfig() {
	// Customize configuration with higher similarity threshold
	clf, err := classifier.NewClassifier(classifier.Config{
		EmbeddingClient:     classifier.NewVoyageEmbeddingAdapter(nil),
		VectorClientLabel:   classifier.NewPineconeVectorAdapter(nil, nil, "my_namespace_label"),
		VectorClientContent: classifier.NewPineconeVectorAdapter(nil, nil, "my_namespace_content"),
		LLMClient:           classifier.NewDefaultLLMClient(nil, "production"),
		MinSimilarity:       0.85, // Higher threshold for cache hits
		DSUPersistence:      classifier.NewFileDSUPersistence("./my_labels.bin"),
	})
	if err != nil {
		log.Fatal(err)
	}

	// Classify text
	result, err := clf.Classify(context.Background(), "How do I install this package?")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Label: %s\n", result.Label)

	// Get metrics
	metrics := clf.GetMetrics()
	fmt.Printf("Unique Labels: %d\n", metrics.UniqueLabels)
	fmt.Printf("Converged Labels: %d\n", metrics.ConvergedLabels)
	fmt.Printf("Cache Hit Rate: %.2f%%\n", metrics.CacheHitRate)

	// Save state
	if err := clf.SaveDSU(); err != nil {
		log.Fatal(err)
	}
}
