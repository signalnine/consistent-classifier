package benchmark

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/FrenchMajesty/consistent-classifier/clients/voyage"
	"github.com/FrenchMajesty/consistent-classifier/utils/disjoint_set"
)

type DatasetItem struct {
	UserResponse string
	UserCategory string
}

type Result struct {
	Content string
	Label   string
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

// Vectorize will classify texts using BoW vector clustering.
func Vectorize() {
	// Prepare
	voyageClient := voyage.NewEmbeddingService()
	DSU := disjoint_set.NewDSU(0)
	dataset := []DatasetItem{}
	results := make([]Result, 0)

	// Read the DSU from file
	filepath := os.Getenv("DSU_FILEPATH")
	DSU.ReadFromFile(filepath)

	// Generate embeddings for the entire dataset
	texts := []string{}
	for _, text := range dataset {
		texts = append(texts, text.UserResponse)
	}

	embeddings, err := voyageClient.GenerateEmbeddings(context.Background(), texts, voyage.VoyageEmbeddingTypeDocument)
	if err != nil {
		log.Fatal(err)
	}

	// Classify the tweet
	for i, text := range dataset {
		hit := searchPineconeForContent(embeddings[i].Embedding)
		if hit != nil {
			results = append(results, Result{
				Content: text.UserResponse,
				Label:   hit.Label,
			})
			continue
		}

		class := classifyTextWithLLM(text.UserResponse)
		results = append(results, Result{
			Content: text.UserResponse,
			Label:   class,
		})
	}

	// Find the root of the label & persist vector data
	for _, result := range results {
		fmt.Println(result)
		similarLabel := searchPineconeForLabel(result.Label)
		rootLabel := result.Label
		if similarLabel != nil {
			rootLabel = similarLabel.Root
		}

		DSU.Union(DSU.FindOrCreate(rootLabel), DSU.FindOrCreate(result.Label))
		err := DSU.Save()
		if err != nil {
			log.Fatal(err)
		}

		// 3- Upsert the [content]:[label] into Pinecone
		// 4- Upsert the [label]:[root] into Pinecone
	}

	/*
		// Another is --classify=vectorize


		// First search for the tweet in Pinecone to find a class. If none, use LLM to classify it.
		// Then use the class label to find a similar label, to retrace to its root label (via metadata).
		// Use DSU serialized in a file as a JSON file.

		// DSU contains all unique labels. Extract them all as an array
		// Then do k-means clustering on the array.
		// Then do path-compression on the clusters.
		// Update the labels metadata in Pinecone.
		// Save the updated DSU in file.
	*/
}

func searchPineconeForContent(vectors []float32) *ContentVectorHit {
	return &ContentVectorHit{}
}

func searchPineconeForLabel(label string) *LabelVectorHit {
	return &LabelVectorHit{}
}
