package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/FrenchMajesty/consistent-classifier/benchmark"
	"github.com/joho/godotenv"
)

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	// Define CLI flags
	classifyMode := flag.String("classify", "", "Classification mode: 'llm' or 'vectorize'")
	smokeTest := flag.Bool("smoke-test", false, "Run a quick smoke test with a small dataset")
	limit := flag.Int("limit", 0, "Number of rows to classify (0 = use default max)")

	flag.Parse()

	// Validate required environment variables
	if os.Getenv("DATASET_FILEPATH") == "" {
		log.Fatal("DATASET_FILEPATH environment variable not set")
	}

	// Determine the limit to use
	datasetLimit := *limit
	if *smokeTest {
		datasetLimit = 10 // Default smoke test size
		fmt.Println("Running smoke test with 10 items...")
	}

	// Route to the appropriate benchmark function
	switch *classifyMode {
	case "llm":
		fmt.Println("Running LLM classification...")
		benchmark.LLM(datasetLimit)
		fmt.Println("LLM classification complete!")
	case "vectorize":
		fmt.Println("Running vector clustering classification...")
		benchmark.Vectorize(datasetLimit)
		fmt.Println("Vector clustering classification complete!")
	case "":
		log.Fatal("Please specify a classification mode with --classify=llm or --classify=vectorize")
	default:
		log.Fatalf("Unknown classification mode: %s. Use 'llm' or 'vectorize'", *classifyMode)
	}
}
