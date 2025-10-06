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

	queryEmbeddings, storageEmbeddings, err := embedDataset(voyageClient, dataset)
	if err != nil {
		log.Fatal(err)
	}

	// Classify the tweet
	progressInterval := limit / 20
	if progressInterval == 0 {
		progressInterval = 5
	}

	for i, tweet := range dataset {
		if i%progressInterval == 0 {
			fmt.Printf("Classifying tweet %d/%d\n", i, limit)
		}

		tweetStartTime := time.Now()
		hit := searchPineconeForTweet(vectorContentIndex, queryEmbeddings[i].Embedding)
		benchmarkMetrics.VectorReads++
		if hit != nil {
			benchmarkMetrics.VectorReplyHits++
			results = append(results, Result{
				Post:       tweet.Content,
				Reply:      tweet.UserResponse,
				ReplyLabel: hit.Label,
			})
			benchmarkMetrics.TokenUsage = append(benchmarkMetrics.TokenUsage, TokenUsageMetrics{
				InputTokens:       0,
				CachedInputTokens: 0,
				OutputTokens:      0,
			})
			benchmarkMetrics.ProcessingTime = append(benchmarkMetrics.ProcessingTime, time.Since(tweetStartTime))
			continue
		}

		label, tokenUsage, err := classifyTextWithLLM(tweet.Content, tweet.UserResponse)
		if err != nil {
			log.Fatal(err)
		}

		benchmarkMetrics.ProcessingTime = append(benchmarkMetrics.ProcessingTime, time.Since(tweetStartTime))
		benchmarkMetrics.TokenUsage = append(benchmarkMetrics.TokenUsage, *tokenUsage)
		results = append(results, Result{
			Post:       tweet.Content,
			Reply:      tweet.UserResponse,
			ReplyLabel: label,
		})
	}

	for i, result := range results {
		if i%progressInterval == 0 {
			fmt.Printf("Finding root of label %d/%d\n", i, limit)
		}

		// Find the root of the label
		rootLabel := result.ReplyLabel
		similarLabel := searchPineconeForLabel(voyageClient, vectorLabelIndex, result.ReplyLabel)
		benchmarkMetrics.VectorReads++
		if similarLabel != nil {
			rootLabel = similarLabel.Root
			benchmarkMetrics.VectorLabelHits++
		}

		// Union the label with the root label
		DSU.Union(DSU.FindOrCreate(rootLabel), DSU.FindOrCreate(result.ReplyLabel))
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
			err = upsertTweetToVector(vectorContentIndex, uuid, result.Reply, storageEmbeddings[i].Embedding, result.ReplyLabel)
			if err != nil {
				fmt.Println(err)
				log.Fatal(err)
			}
			benchmarkMetrics.VectorWrites++
		}()

		// Create lookup vector ref by label to find root
		go func() {
			defer wg.Done()
			id := fmt.Sprintf("label:%s", result.ReplyLabel)
			err = upsertLabelToVector(vectorLabelIndex, voyageClient, id, result.ReplyLabel, rootLabel)
			if err != nil {
				log.Fatal(err)
			}
			benchmarkMetrics.VectorWrites++
		}()

		wg.Wait()
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
	embedding, err := voyageClient.GenerateEmbedding(context.Background(), label, voyage.VoyageEmbeddingTypeQuery)
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
