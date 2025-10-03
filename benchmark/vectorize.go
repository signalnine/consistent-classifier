package benchmark

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"github.com/FrenchMajesty/consistent-classifier/clients/pinecone"
	"github.com/FrenchMajesty/consistent-classifier/clients/voyage"
	"github.com/FrenchMajesty/consistent-classifier/utils/disjoint_set"
	"github.com/austinfhunter/voyageai"
	"google.golang.org/protobuf/types/known/structpb"
)

// Vectorize will classify texts using Bag of Words (BoW) vector clustering.
func Vectorize() {
	// Prepare
	voyageClient := voyage.NewEmbeddingService()
	pineconeClient := pinecone.NewPineconeService()
	vectorIndex := pineconeClient.ForBaseIndex()
	DSU := disjoint_set.NewDSU()

	dataset, err := loadDataset()
	if err != nil {
		log.Fatal(err)
	}

	results := make([]Result, 0)
	startTime := time.Now()
	benchmarkMetrics := BenchmarkMetrics{
		TotalTweets: len(dataset),
	}

	// Read the DSU from file
	filepath := os.Getenv("DSU_FILEPATH")
	DSU.ReadFromFile(filepath)

	queryEmbeddings, storageEmbeddings, err := embedDataset(voyageClient, dataset)
	if err != nil {
		log.Fatal(err)
	}

	// Classify the tweet
	for i, tweet := range dataset {
		tweetStartTime := time.Now()
		hit := searchPineconeForTweet(queryEmbeddings[i].Embedding)
		benchmarkMetrics.VectorReads++
		if hit != nil {
			results = append(results, Result{
				Tweet: tweet.UserResponse,
				Label: hit.Label,
			})
			benchmarkMetrics.TokenUsage = append(benchmarkMetrics.TokenUsage, TokenUsageMetrics{
				InputTokens:       0,
				CachedInputTokens: 0,
				OutputTokens:      0,
			})
			benchmarkMetrics.ProcessingTime = append(benchmarkMetrics.ProcessingTime, time.Since(tweetStartTime))
			continue
		}

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
	}

	var wg sync.WaitGroup
	// Find the root of the label & persist vector data
	for i, result := range results {
		// Find the root of the label
		rootLabel := result.Label
		similarLabel := searchPineconeForLabel(voyageClient, result.Label)
		benchmarkMetrics.VectorReads++
		if similarLabel != nil {
			rootLabel = similarLabel.Root
		}

		// Union the label with the root label
		DSU.Union(DSU.FindOrCreate(rootLabel), DSU.FindOrCreate(result.Label))
		err = DSU.Save()
		if err != nil {
			log.Fatal(err)
		}

		wg.Add(2)
		// Create lookup vector ref by tweet to shorcut classification
		go func() {
			defer wg.Done()
			id := fmt.Sprintf("content:%d", i)
			upsertTweetToVector(vectorIndex, id, result.Tweet, storageEmbeddings[i].Embedding, result.Label)
			benchmarkMetrics.VectorWrites++
		}()

		// Create lookup vector ref by label to find root
		go func() {
			defer wg.Done()
			id := fmt.Sprintf("label:%d", i)
			upsertLabelToVector(vectorIndex, voyageClient, id, result.Label, rootLabel)
			benchmarkMetrics.VectorWrites++
		}()

		wg.Wait()
	}

	benchmarkMetrics.TotalDuration = time.Since(startTime)
	benchmarkMetrics.UniqueLabels = DSU.Size()
	benchmarkMetrics.ConvergedLabels = DSU.CountSets()

	err = saveMetricsToFile(benchmarkMetrics)
	if err != nil {
		log.Fatal(err)
	}
}

// Embed the dataset both as a query and document
func embedDataset(voyageClient EmbeddingInterface, dataset []DatasetItem) ([]voyageai.EmbeddingObject, []voyageai.EmbeddingObject, error) {
	// Generate embeddings for the entire dataset
	tweets := []string{}
	for _, tweet := range dataset {
		tweets = append(tweets, tweet.UserResponse)
	}

	queryEmbeddings := []voyageai.EmbeddingObject{}
	storageEmbeddings := []voyageai.EmbeddingObject{}
	var err error
	var finalErr error

	// Parallelize the embedding generation
	wg := sync.WaitGroup{}
	wg.Add(2)

	go func() {
		defer wg.Done()
		queryEmbeddings, err = voyageClient.GenerateEmbeddings(context.Background(), tweets, voyage.VoyageEmbeddingTypeQuery)
		if err != nil {
			finalErr = err
		}
	}()

	go func() {
		defer wg.Done()
		storageEmbeddings, err = voyageClient.GenerateEmbeddings(context.Background(), tweets, voyage.VoyageEmbeddingTypeDocument)
		if err != nil {
			finalErr = err
		}
	}()

	wg.Wait()

	if finalErr != nil {
		return nil, nil, finalErr
	}

	return queryEmbeddings, storageEmbeddings, nil
}

// Search for a tweet vector in Pinecone
func searchPineconeForTweet(vectors []float32) *ContentVectorHit {
	return &ContentVectorHit{}
}

// Search for a label vector in Pinecone
func searchPineconeForLabel(voyageClient EmbeddingInterface, label string) *LabelVectorHit {
	embedding, err := voyageClient.GenerateEmbedding(context.Background(), label, voyage.VoyageEmbeddingTypeQuery)
	if err != nil {
		log.Fatal(err)
	}
	return &LabelVectorHit{}
}

// Upsert a tweet vector to Pinecone
func upsertTweetToVector(vectorIndex IndexOperationsInterface, id string, tweet string, embedding []float32, label string) error {
	metadata, err := structpb.NewStruct(map[string]any{
		"vector_text": tweet,
		"label":       label,
	})
	if err != nil {
		log.Fatal(err)
	}

	err = vectorIndex.Upsert(context.Background(), []pinecone.Vector{
		{
			Id:     id,
			Values: embedding,
			Metadata: &pinecone.Metadata{
				Fields: metadata.Fields,
			},
		},
	})
	if err != nil {
		log.Fatal(err)
		return err
	}

	return nil
}

// Upsert a label vector to Pinecone
func upsertLabelToVector(vectorIndex IndexOperationsInterface, voyageClient EmbeddingInterface, id string, label string, root string) error {
	embedding, err := voyageClient.GenerateEmbedding(context.Background(), label, voyage.VoyageEmbeddingTypeDocument)
	if err != nil {
		log.Fatal(err)
	}
	metadata, err := structpb.NewStruct(map[string]any{
		"label": label,
		"root":  root,
	})
	if err != nil {
		log.Fatal(err)
	}

	err = vectorIndex.Upsert(context.Background(), []pinecone.Vector{
		{
			Id:     id,
			Values: embedding,
			Metadata: &pinecone.Metadata{
				Fields: metadata.Fields,
			},
		},
	})
	if err != nil {
		log.Fatal(err)
		return err
	}

	return nil
}
