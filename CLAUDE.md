# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`consistent-classifier` is a Go package for LLM-powered text classification with vector caching and label clustering. It uses:
- **Voyage AI** for text embeddings
- **Pinecone** for vector similarity search
- **OpenAI-compatible LLMs** for classification
- **Disjoint Set Union (DSU)** for clustering semantically similar labels

## Common Commands

### Testing
```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run specific package tests
go test ./adapters/...
go test ./internal/disjoint_set/...

# Generate coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### Building
```bash
# Build the benchmark tool
go build -o consistent-classifier ./cmd/benchmark

# Run benchmark with LLM-only classification
./consistent-classifier --classify=llm --limit=100

# Run benchmark with vector caching + clustering
./consistent-classifier --classify=vectorize --limit=100

# Quick smoke test
./consistent-classifier --classify=vectorize --smoke-test
```

### Development
```bash
# Install dependencies
go mod download

# Tidy dependencies
go mod tidy

# Run benchmarks
go test -bench=. ./...
```

## Architecture

### Core Classification Flow

The classifier operates in two modes depending on cache hits:

**Cache Hit Path** (fast, <200ms):
1. Generate embedding for input text ([classifier.go:125](classifier.go#L125))
2. Search vector store for similar text ([classifier.go:131](classifier.go#L131))
3. Return cached label if similarity >= threshold ([classifier.go:137](classifier.go#L137))
4. Resolve to root label via DSU ([classifier.go:147](classifier.go#L147))

**Cache Miss Path** (slow, 1-2s):
1. Call LLM for classification ([classifier.go:159](classifier.go#L159))
2. Spawn background tasks asynchronously ([classifier.go:191](classifier.go#L191)):
   - Cache text embedding for future lookups
   - Find similar labels and cluster via DSU
   - Cache label embedding for clustering

### Key Components

**Classifier** ([classifier.go](classifier.go))
- Main orchestration logic
- Thread-safe with graceful shutdown
- Tracks metrics (cache hit rate, unique labels, converged labels)

**Interfaces** ([interfaces.go](interfaces.go))
- `EmbeddingClient`: Text → vector embeddings
- `VectorClient`: Vector search and storage
- `LLMClient`: Text → classification labels
- `DisjointSetPersistence`: DSU state save/load

**Adapters** ([adapters/](adapters/))
- Concrete implementations for external services
- `VoyageEmbeddingAdapter`: Voyage AI integration
- `PineconeVectorAdapter`: Pinecone vector DB
- `DefaultLLMClient`: OpenAI-compatible LLMs
- All accept custom clients or fall back to env vars

**Disjoint Set Union** ([internal/disjoint_set/](internal/disjoint_set/))
- Thread-safe label clustering
- Path compression for O(α(n)) operations
- Binary serialization for persistence ([internal/disjoint_set/serializer.go](internal/disjoint_set/serializer.go))

### Configuration

The `Config` struct ([config.go](config.go)) supports:
- Custom clients for all external services
- Two similarity thresholds:
  - `MinSimilarityContent`: For text cache hits (default 0.80)
  - `MinSimilarityLabel`: For label clustering (default 0.80)
- Pluggable DSU persistence (defaults to `./dsu_state.bin`)

### Environment Variables

Required for default adapters:
```bash
VOYAGEAI_API_KEY      # Voyage AI embeddings
PINECONE_API_KEY      # Pinecone vector DB
PINECONE_HOST         # Pinecone index host
OPENAI_API_KEY        # OpenAI or compatible API
DATASET_FILEPATH      # For benchmark tool only
```

### Background Task Handling

The classifier uses `sync.WaitGroup` to track background tasks ([classifier.go:32](classifier.go#L32)) and ensures:
- No classifications accepted during shutdown ([classifier.go:115](classifier.go#L115))
- All background tasks complete before `Close()` returns ([classifier.go:342](classifier.go#L342))
- DSU state persisted on graceful shutdown ([classifier.go:345](classifier.go#L345))

### Label Clustering (DSU)

Labels are clustered when new classifications occur:
1. Generate embedding for new label ([classifier.go:263](classifier.go#L263))
2. Search for similar existing labels ([classifier.go:269](classifier.go#L269))
3. If similarity >= threshold, union with root label ([classifier.go:284](classifier.go#L284))
4. Store label with root reference in metadata ([classifier.go:314](classifier.go#L314))

This ensures consistency: `"tech_question"`, `"technical_question"`, and `"tech_support"` converge to a single root label.

### Vector Namespaces

Separate Pinecone namespaces are used for:
- **Content vectors**: Store classified text embeddings for cache lookups
- **Label vectors**: Store label embeddings for clustering similar labels

Each namespace is passed when creating `PineconeVectorAdapter` ([adapters/adapters.go:47](adapters/adapters.go#L47)).

### Testing Strategy

Tests use mock implementations:
- `testutil/mocks.go`: Mock clients for unit tests
- `_test.go` files: Table-driven tests with context awareness
- Benchmark tool: Real-world integration testing with tweet dataset

### Custom LLM Providers

The `DefaultLLMClient` supports any OpenAI-compatible API via `baseUrl` parameter:
- Azure OpenAI
- Groq
- Local models (e.g., Ollama)
- See [adapters/llm_client.go:29](adapters/llm_client.go#L29)

### Production Considerations

- Rate limiting: Not enforced; wrap clients with custom rate limiters
- Metrics: Call `GetMetrics()` periodically and export to monitoring systems
- Namespace isolation: Use unique Pinecone namespaces per environment
- DSU persistence: Call `SaveDSU()` explicitly if needed before shutdown
