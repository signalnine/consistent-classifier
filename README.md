# Consistent Classifier

A high-performance Go package for classifying large volumes of unlabeled text data using LLM-powered classification with intelligent caching and label clustering.

## Features

- **Smart Caching**: Vector-based similarity search reduces redundant LLM calls by ~80-95% on similar text
- **Label Clustering**: Automatically merges semantically similar labels (e.g., "technical_question" and "tech_support") using Disjoint Set Union (DSU)
- **Production Ready**: Thread-safe, context-aware, with graceful shutdown and persistent state
- **Pluggable Adapters**: Easily swap embedding providers (Voyage AI), vector stores (Pinecone), or LLMs (OpenAI-compatible)
- **Zero Config**: Works out-of-the-box with environment variables, or fully customize every component

## Installation

```bash
go get github.com/FrenchMajesty/consistent-classifier
```

## Quick Start

### Basic Usage

```go
package main

import (
    "context"
    "log"

    classifier "github.com/FrenchMajesty/consistent-classifier"
)

func main() {
    // Create classifier with defaults (reads from environment variables)
    clf, err := classifier.NewClassifier(classifier.Config{})
    if err != nil {
        log.Fatal(err)
    }
    defer clf.Close() // Saves state and waits for background tasks

    // Classify text
    result, err := clf.Classify(context.Background(), "Thanks for the help!")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Label: %s (cache hit: %v, latency: %v)",
        result.Label, result.CacheHit, result.UserFacingLatency)
}
```

### Environment Variables

Set these to use the default adapters:

```bash
export VOYAGEAI_API_KEY="your-voyage-key"
export PINECONE_API_KEY="your-pinecone-key"
export PINECONE_HOST="your-index-host.pinecone.io"
export OPENAI_API_KEY="your-openai-key"
```

## Advanced Configuration

### Custom Clients

```go
import (
    classifier "github.com/FrenchMajesty/consistent-classifier"
    "github.com/FrenchMajesty/consistent-classifier/adapters"
)

// Create custom clients
embeddingClient, _ := adapters.NewVoyageEmbeddingAdapter(nil)
vectorClientLabel, _ := adapters.NewPineconeVectorAdapter(nil, nil, "prod_labels")
vectorClientContent, _ := adapters.NewPineconeVectorAdapter(nil, nil, "prod_content")
llmClient, _ := adapters.NewDefaultLLMClient(nil, "", "gpt-4o-mini", "")

clf, _ := classifier.NewClassifier(classifier.Config{
    EmbeddingClient:      embeddingClient,
    VectorClientLabel:    vectorClientLabel,
    VectorClientContent:  vectorClientContent,
    LLMClient:            llmClient,
    MinSimilarityContent: 0.90, // Higher threshold = fewer cache hits, more precision
    MinSimilarityLabel:   0.75, // Threshold for merging similar labels
    DSUPersistence:       classifier.NewFileDSUPersistence("./labels.bin"),
})
defer clf.Close()
```

### Custom LLM System Prompt

```go
customPrompt := `Classify the following customer support ticket into one of:
- bug_report
- feature_request
- billing_question
- other

Return only the label.`

llmClient, _ := adapters.NewDefaultLLMClient(nil, customPrompt, "gpt-4o", "")
```

### OpenAI-Compatible Providers

Works with any OpenAI-compatible API (e.g., Azure, local models):

```go
groqApiKey := os.Getenv('GROQ_API_KEY')
llmClient, _ := adapters.NewDefaultLLMClient(
    groqApiKey, // api key, fallbacks to envVar
    "",  // system prompt, fallbacks to default
    "llama-3.370b-versatile",
    "https://api.groq.com/openai/v1", // base URL
)
```

## How It Works

1. **Embedding Generation**: Text is converted to a vector using Voyage AI (or custom provider)
2. **Cache Check**: Searches vector store for similar previously-classified text
3. **On Cache Hit**: Returns cached label instantly (typically <100ms)
4. **On Cache Miss**: Calls LLM for classification, then:
   - Stores text embedding for future lookups
   - Searches for similar labels and clusters them using DSU
   - Stores label embedding for clustering

### Label Clustering Example

If the LLM generates these labels across classifications:
```
"technical_question" → "tech_question" → "technical_support"
```

The DSU automatically groups them, so future queries return the **root label** of the cluster, ensuring consistency.

## API Reference

### Core Methods

```go
// Create a new classifier
func NewClassifier(cfg Config) (*Classifier, error)

// Classify text and return result
func (c *Classifier) Classify(ctx context.Context, text string) (*Result, error)

// Get current metrics
func (c *Classifier) GetMetrics() Metrics

// Graceful shutdown (waits for background tasks and saves state)
func (c *Classifier) Close() error
```

### Result Structure

```go
type Result struct {
    Label             string        // Classified label
    CacheHit          bool          // Whether result came from cache
    Confidence        float32       // Similarity score (if cache hit)
    UserFacingLatency time.Duration // Time user waited
    BackgroundLatency time.Duration // Time spent on clustering/caching
}
```

### Metrics

```go
type Metrics struct {
    UniqueLabels    int     // Total unique labels seen
    ConvergedLabels int     // Number of label clusters after merging
    CacheHitRate    float32 // Percentage of cache hits
}
```

## Production Considerations

### Rate Limiting

⚠️ The package doesn't enforce rate limits. For production use with high volume, we recommend providing your own implementation of the `LanguageModelClient` interface:

```go
// Wrap with your own rate limiter
type RateLimitedLLM struct {
    limiter *rate.Limiter
    client  classifier.LLMClient
}

func (r *RateLimitedLLM) Classify(ctx context.Context, text string) (string, error) {
    if err := r.limiter.Wait(ctx); err != nil {
        return "", err
    }
    return r.client.Classify(ctx, text)
}
```

### Monitoring

```go
// Poll metrics periodically
ticker := time.NewTicker(30 * time.Second)
go func() {
    for range ticker.C {
        m := clf.GetMetrics()
        // Send to your metrics system (Prometheus, Datadog, etc.)
        log.Printf("Labels: %d/%d, Cache: %.1f%%",
            m.ConvergedLabels, m.UniqueLabels, m.CacheHitRate)
    }
}()
```

### Namespace Isolation

For multiple instances or environments, use unique Pinecone namespaces:

```go
apiKey := os.Getenv('PINECONE_API_KEY')
labelHost := os.Getenv('PINECONE_LABEL_HOST')
contentHost := os.Getenv('PINECONE_CONTENT_HOST')
vectorLabel, _ := adapters.NewPineconeVectorAdapter(apiKey, labelHost, "prod_labels_v2")
vectorContent, _ := adapters.NewPineconeVectorAdapter(apiKey, contentHost, "prod_content_v2")
```

## Testing

```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run benchmarks
go test -bench=. ./...
```

## Example: Classify Replies to Tweets

See [cmd/benchmark/vectorize.go](cmd/benchmark/vectorize.go) for a full example of classifying thousands of tweet replies.

## Performance

On a dataset of 10,000 tweet replies in a specific niche (using `cmd/benchmark`):
- **Cache hit rate**: 25% hit rate after 500. Reaches +50% by 2,000 and over 95% by the 10,000th
- **Avg latency (cache hit)**: <200ms
- **Avg latency (cache miss)**: ~1-2s (LLM dependent)
- **Cost reduction**: 90%+ fewer LLM calls vs naive classification

Read the [full length essay](https://verdik.substack.com/p/how-to-get-consistent-classification) where I go more in depth on the performance.

## Architecture

```
consistent-classifier/
├── *.go                # Core classification logic (classifier, config, types, etc.)
├── adapters/           # External service adapters
│   ├── adapters.go     # Voyage and Pinecone adapters
│   ├── llm_client.go   # OpenAI adapter
│   ├── openai/         # OpenAI client implementation
│   ├── pinecone/       # Pinecone client implementation
│   └── voyage/         # Voyage AI client implementation
├── types/              # Shared types
├── internal/
│   └── disjoint_set/   # DSU implementation for label clustering
└── cmd/
    └── benchmark/      # Benchmarking utilities
```

## Contributing

Contributions welcome! Please open an issue or PR.

## License

MIT License - see [LICENSE](LICENSE) for details.

