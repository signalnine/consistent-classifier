package benchmark

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"

	"github.com/FrenchMajesty/consistent-classifier/clients/pinecone"
	"github.com/FrenchMajesty/consistent-classifier/clients/voyage"
	"github.com/FrenchMajesty/consistent-classifier/utils/disjoint_set"
	"google.golang.org/protobuf/types/known/structpb"
)

type DatasetItem struct {
	UserResponse string
	UserCategory string
}

type Result struct {
	Tweet string
	Label string
}

type VectorHit struct {
	Score      float32
	VectorText string
}

type ContentVectorHit struct {
	*VectorHit
	Label string
}

type LabelVectorHit struct {
	*VectorHit
	Root string
}

// Vectorize will classify texts using Bag of Words (BoW) vector clustering.
func Vectorize() {
	// Prepare
	voyageClient := voyage.NewEmbeddingService()
	pineconeClient := pinecone.NewPineconeService()
	vectorIndex := pineconeClient.ForBaseIndex()
	DSU := disjoint_set.NewDSU(0)
	dataset := []DatasetItem{}
	results := make([]Result, 0)

	// Read the DSU from file
	filepath := os.Getenv("DSU_FILEPATH")
	DSU.ReadFromFile(filepath)

	// Generate embeddings for the entire dataset
	tweets := []string{}
	for _, tweet := range dataset {
		tweets = append(tweets, tweet.UserResponse)
	}

	queryEmbeddings, err := voyageClient.GenerateEmbeddings(context.Background(), tweets, voyage.VoyageEmbeddingTypeQuery)
	if err != nil {
		log.Fatal(err)
	}

	storageEmbeddings, err := voyageClient.GenerateEmbeddings(context.Background(), tweets, voyage.VoyageEmbeddingTypeDocument)
	if err != nil {
		log.Fatal(err)
	}

	// Classify the tweet
	for i, tweet := range dataset {
		hit := searchPineconeForTweet(queryEmbeddings[i].Embedding)
		if hit != nil {
			results = append(results, Result{
				Tweet: tweet.UserResponse,
				Label: hit.Label,
			})
			continue
		}

		class := classifyTextWithLLM(tweet.UserResponse)
		results = append(results, Result{
			Tweet: tweet.UserResponse,
			Label: class,
		})
	}

	// Find the root of the label & persist vector data
	for i, result := range results {
		// Find the root of the label
		similarLabel := searchPineconeForLabel(voyageClient, result.Label)
		rootLabel := result.Label
		if similarLabel != nil {
			rootLabel = similarLabel.Root
		}

		// Union the label with the root label
		DSU.Union(DSU.FindOrCreate(rootLabel), DSU.FindOrCreate(result.Label))
		err = DSU.Save()
		if err != nil {
			log.Fatal(err)
		}

		var wg sync.WaitGroup
		wg.Add(2)
		// Create lookup vector ref by tweet to shorcut classification
		go func() {
			defer wg.Done()
			id := fmt.Sprintf("content:%d", i)
			upsertTweetToVector(vectorIndex, id, result.Tweet, storageEmbeddings[i].Embedding, result.Label)
		}()

		// Create lookup vector ref by label to find root
		go func() {
			defer wg.Done()
			id := fmt.Sprintf("label:%d", i)
			upsertLabelToVector(vectorIndex, voyageClient, id, result.Label, rootLabel)
		}()

		wg.Wait()
	}
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
