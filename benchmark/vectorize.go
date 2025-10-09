package benchmark

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/FrenchMajesty/consistent-classifier/classifier"
)

// Vectorize will classify texts using Bag of Words (BoW) vector clustering.
func Vectorize(limit int) {
	// Initialize classifier - no clients provided, rely on defaults with environment variables
	clf, err := classifier.NewClassifier(classifier.Config{})
	if err != nil {
		log.Fatal(err)
	}

	dataset, err := loadDataset(limit)
	if err != nil {
		log.Fatal(err)
	}

	results := make([]Result, 0)
	startTime := time.Now()
	benchmarkMetrics := BenchmarkMetrics{
		TotalTweets: len(dataset),
	}

	// Classify tweets
	progressInterval := 5
	if limit > 100 {
		progressInterval = 10
	}

	for i, tweet := range dataset {
		if i%progressInterval == 0 {
			fmt.Printf("Classifying tweet %d/%d\n", i, limit)
		}

		// Use the classifier to classify the tweet reply
		result, err := clf.Classify(context.Background(), tweet.UserResponse)
		if err != nil {
			log.Fatal(err)
		}

		// Track vector operations for benchmarking
		benchmarkMetrics.VectorReads++ // At minimum one read for content search
		if result.CacheHit {
			benchmarkMetrics.VectorReplyHits++
		} else {
			// On cache miss, we do additional vector operations in background
			benchmarkMetrics.VectorReads++  // Label similarity search
			benchmarkMetrics.VectorWrites++ // Content vector upsert
			benchmarkMetrics.VectorWrites++ // Label vector upsert
		}

		// Record metrics
		benchmarkMetrics.UserFacingLatency = append(benchmarkMetrics.UserFacingLatency, result.UserFacingLatency)
		benchmarkMetrics.BackgroundTime = append(benchmarkMetrics.BackgroundTime, result.BackgroundLatency)
		benchmarkMetrics.CacheHit = append(benchmarkMetrics.CacheHit, result.CacheHit)

		// Store result
		results = append(results, Result{
			Post:       tweet.Content,
			Reply:      tweet.UserResponse,
			ReplyLabel: result.Label,
		})

		// Backwards compatibility - approximate token usage
		benchmarkMetrics.ProcessingTime = append(benchmarkMetrics.ProcessingTime, result.UserFacingLatency)
		benchmarkMetrics.TokenUsage = append(benchmarkMetrics.TokenUsage, TokenUsageMetrics{
			InputTokens:       0, // Not tracked in new classifier
			CachedInputTokens: 0,
			OutputTokens:      0,
		})
	}

	// Save DSU state at the end
	err = clf.SaveDSU()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Computing metrics...")

	// Get metrics from classifier
	clfMetrics := clf.GetMetrics()

	benchmarkMetrics.TotalDuration = time.Since(startTime)
	benchmarkMetrics.UniqueLabels = clfMetrics.UniqueLabels
	benchmarkMetrics.ConvergedLabels = clfMetrics.ConvergedLabels

	err = saveMetricsToFile(benchmarkMetrics)
	if err != nil {
		log.Fatal(err)
	}

	err = saveResultsToFile(results)
	if err != nil {
		log.Fatal(err)
	}
}
