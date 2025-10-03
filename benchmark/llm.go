package benchmark

import (
	"log"
	"time"
)

func LLM() {
	dataset, err := loadDataset()
	if err != nil {
		log.Fatal(err)
	}

	results := make([]Result, 0)
	labels := make(map[string]bool)
	startTime := time.Now()
	benchmarkMetrics := BenchmarkMetrics{
		TotalTweets: len(dataset),
	}

	for _, tweet := range dataset {
		tweetStartTime := time.Now()
		label, tokenUsage, err := classifyTextWithLLM(tweet.UserResponse)
		if err != nil {
			log.Fatal(err)
		}

		benchmarkMetrics.ProcessingTime = append(benchmarkMetrics.ProcessingTime, time.Since(tweetStartTime))
		benchmarkMetrics.TokenUsage = append(benchmarkMetrics.TokenUsage, tokenUsage)
		results = append(results, Result{
			Tweet: tweet.UserResponse,
			Label: label,
		})
		labels[label] = true
	}

	benchmarkMetrics.TotalDuration = time.Since(startTime)
	benchmarkMetrics.UniqueLabels = len(labels)
	benchmarkMetrics.ConvergedLabels = len(labels)

	err = saveMetricsToFile(benchmarkMetrics)
	if err != nil {
		log.Fatal(err)
	}
}

func classifyTextWithLLM(text string) (string, TokenUsageMetrics, error) {
	return "", TokenUsageMetrics{}, nil
}
