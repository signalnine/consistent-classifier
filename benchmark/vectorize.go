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
	"github.com/google/uuid"
	"google.golang.org/protobuf/types/known/structpb"
)

const MIN_SIMILARITY_SCORE = 0.80

// Vectorize will classify texts using Bag of Words (BoW) vector clustering.
func Vectorize(limit int) {
	// Prepare
	voyageClient := voyage.NewEmbeddingService()
	pineconeClient := pinecone.NewPineconeService()
	vectorLabelIndex := pineconeClient.ForBaseIndex("label")
	vectorContentIndex := pineconeClient.ForBaseIndex("content")
	DSU := disjoint_set.NewDSU()

	dataset, err := loadDataset(limit)
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

	// Classify the tweet
	progressInterval := 5
	if limit > 100 {
		progressInterval = 10
	}

	for i, tweet := range dataset {
		if i%progressInterval == 0 {
			fmt.Printf("Classifying tweet %d/%d\n", i, limit)
		}

		// User-facing latency starts here
		userFacingStart := time.Now()

		// Step 1: Generate embedding for this tweet
		tweetEmbedding, err := voyageClient.GenerateEmbedding(context.Background(), tweet.UserResponse, voyage.VoyageEmbeddingTypeDocument)
		if err != nil {
			log.Fatal(err)
		}

		// Step 2: Vector search (user waits for this)
		hit := searchPineconeForTweet(vectorContentIndex, tweetEmbedding)
		benchmarkMetrics.VectorReads++

		if hit != nil {
			// Cache HIT - user gets instant response
			userFacingLatency := time.Since(userFacingStart)
			benchmarkMetrics.VectorReplyHits++
			benchmarkMetrics.UserFacingLatency = append(benchmarkMetrics.UserFacingLatency, userFacingLatency)
			benchmarkMetrics.BackgroundTime = append(benchmarkMetrics.BackgroundTime, 0) // No background work on cache hit
			benchmarkMetrics.CacheHit = append(benchmarkMetrics.CacheHit, true)

			results = append(results, Result{
				Post:       tweet.Content,
				Reply:      tweet.UserResponse,
				ReplyLabel: hit.Label,
			})

			// Backwards compatibility
			benchmarkMetrics.ProcessingTime = append(benchmarkMetrics.ProcessingTime, userFacingLatency)
			benchmarkMetrics.TokenUsage = append(benchmarkMetrics.TokenUsage, TokenUsageMetrics{
				InputTokens:       0,
				CachedInputTokens: 0,
				OutputTokens:      0,
			})
			continue
		}

		// Cache MISS - call LLM (user waits for this)
		label, tokenUsage, err := classifyTextWithLLM(tweet.Content, tweet.UserResponse)
		if err != nil {
			log.Fatal(err)
		}

		// User-facing latency ends here (they got their classification)
		userFacingLatency := time.Since(userFacingStart)
		benchmarkMetrics.UserFacingLatency = append(benchmarkMetrics.UserFacingLatency, userFacingLatency)
		benchmarkMetrics.CacheHit = append(benchmarkMetrics.CacheHit, false)

		results = append(results, Result{
			Post:       tweet.Content,
			Reply:      tweet.UserResponse,
			ReplyLabel: label,
		})

		// Backwards compatibility
		benchmarkMetrics.ProcessingTime = append(benchmarkMetrics.ProcessingTime, userFacingLatency)
		benchmarkMetrics.TokenUsage = append(benchmarkMetrics.TokenUsage, *tokenUsage)

		// Background processing starts here (happens async in production)
		backgroundStart := time.Now()

		// Find the root of the label
		rootLabel := label
		similarLabel := searchPineconeForLabel(voyageClient, vectorLabelIndex, label)
		benchmarkMetrics.VectorReads++
		if similarLabel != nil {
			fmt.Printf("Found root for %s: %s\n", label, similarLabel.Root)
			rootLabel = similarLabel.Root
			benchmarkMetrics.VectorLabelHits++
		}

		// Union the label with the root label
		DSU.Union(DSU.FindOrCreate(rootLabel), DSU.FindOrCreate(label))
		err = DSU.Save()
		if err != nil {
			log.Fatal(err)
		}

		var wg sync.WaitGroup
		wg.Add(2)

		// Create lookup vector ref by tweet to shorcut classification
		go func() {
			defer wg.Done()
			uuid := uuid.New().String()
			err = upsertTweetToVector(vectorContentIndex, uuid, tweet.UserResponse, tweetEmbedding, label)
			if err != nil {
				log.Fatal(err)
			}
			benchmarkMetrics.VectorWrites++
		}()

		// Create lookup vector ref by label to find root
		go func() {
			defer wg.Done()
			id := label
			err = upsertLabelToVector(vectorLabelIndex, voyageClient, id, label, rootLabel)
			if err != nil {
				log.Fatal(err)
			}
			benchmarkMetrics.VectorWrites++
		}()

		wg.Wait()

		// Background processing ends here
		backgroundTime := time.Since(backgroundStart)
		benchmarkMetrics.BackgroundTime = append(benchmarkMetrics.BackgroundTime, backgroundTime)
	}

	fmt.Println("Computing metrics...")

	benchmarkMetrics.TotalDuration = time.Since(startTime)
	benchmarkMetrics.UniqueLabels = DSU.Size()
	benchmarkMetrics.ConvergedLabels = DSU.CountSets()

	err = saveMetricsToFile(benchmarkMetrics)
	if err != nil {
		log.Fatal(err)
	}

	err = saveResultsToFile(results)
	if err != nil {
		log.Fatal(err)
	}
}


// Search for a tweet vector in Pinecone
func searchPineconeForTweet(vectorIndex IndexOperationsInterface, vectors []float32) *ContentVectorHit {
	matches, err := vectorIndex.Search(context.Background(), vectors, 1, nil, true)
	if err != nil {
		log.Fatal(err)
	}

	if len(matches) == 0 || matches[0].Score <= MIN_SIMILARITY_SCORE {
		return nil
	}

	metadata := matches[0].Vector.Metadata.AsMap()
	return &ContentVectorHit{
		VectorHit: &VectorHit{
			Score:      matches[0].Score,
			VectorText: metadata["vector_text"].(string),
		},
		Label: metadata["label"].(string),
	}
}

// Search for a label vector in Pinecone
func searchPineconeForLabel(voyageClient EmbeddingInterface, vectorIndex IndexOperationsInterface, label string) *LabelVectorHit {
	embedding, err := voyageClient.GenerateEmbedding(context.Background(), label, voyage.VoyageEmbeddingTypeDocument)
	if err != nil {
		log.Fatal(err)
	}

	matches, err := vectorIndex.Search(context.Background(), embedding, 1, nil, true)
	if err != nil {
		log.Fatal(err)
	}

	if len(matches) == 0 || matches[0].Score <= MIN_SIMILARITY_SCORE {
		return nil
	}

	metadata := matches[0].Vector.Metadata.AsMap()
	return &LabelVectorHit{
		VectorHit: &VectorHit{
			Score:      matches[0].Score,
			VectorText: metadata["vector_text"].(string),
		},
		Root: metadata["root"].(string),
	}
}

// Upsert a tweet vector to Pinecone
func upsertTweetToVector(vectorIndex IndexOperationsInterface, id string, tweet string, embedding []float32, label string) error {
	metadata, err := structpb.NewStruct(map[string]any{
		"vector_text": tweet,
		"label":       label,
	})
	if err != nil {
		log.Fatal(err)
		return err
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
		"vector_text": label,
		"label":       label,
		"root":        root,
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
