package main

import (
	"fmt"
	"log"

	"github.com/joho/godotenv"
)

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	fmt.Println("Hello, World!")

	// This is a CLI app where you can give various flags to do various things.
	// One such thing is --smoke-test.
	// it's purpose is to try to use the LLM and Pinecone to classify a handful of texts.
	// Verify the source .csv file is there, etc.. just to make sure it's all good.

	// One is --classify=llm

	// Another is --classify=vectorize

	/**
	We start off with 20 initial labels that will be few-shot examples + roots in DSU.

	*/
}
