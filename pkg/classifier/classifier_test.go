package classifier_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/FrenchMajesty/consistent-classifier/pkg/classifier"
	"github.com/FrenchMajesty/consistent-classifier/pkg/testutil"
	"github.com/FrenchMajesty/consistent-classifier/pkg/types"
	"github.com/FrenchMajesty/consistent-classifier/utils/disjoint_set"
)

// TestClassifier_CacheHit tests that the classifier returns cached results when similarity exceeds threshold
func TestClassifier_CacheHit(t *testing.T) {
	mockEmbedding := &testutil.MockEmbeddingClient{
		GenerateEmbeddingFunc: func(ctx context.Context, text string) ([]float32, error) {
			return []float32{0.1, 0.2, 0.3}, nil
		},
	}

	mockVectorContent := testutil.NewMockVectorClient()
	mockVectorContent.SearchFunc = func(ctx context.Context, vector []float32, topK int) ([]types.VectorMatch, error) {
		// Return a high-similarity match (cache hit)
		return []types.VectorMatch{
			{
				ID:    "test-id",
				Score: 0.95, // Above default threshold of 0.80
				Metadata: map[string]any{
					"label": "test_label",
				},
			},
		}, nil
	}

	mockVectorLabel := testutil.NewMockVectorClient()
	mockLLM := &testutil.MockLLMClient{}
	mockDSU := &testutil.MockDSUPersistence{}

	clf, err := classifier.NewClassifier(classifier.Config{
		EmbeddingClient:     mockEmbedding,
		VectorClientContent: mockVectorContent,
		VectorClientLabel:   mockVectorLabel,
		LLMClient:           mockLLM,
		DSUPersistence:      mockDSU,
	})
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	result, err := clf.Classify(context.Background(), "test text")
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}

	// Verify cache hit
	if !result.CacheHit {
		t.Error("Expected cache hit, got cache miss")
	}

	if result.Confidence != 0.95 {
		t.Errorf("Expected confidence 0.95, got %f", result.Confidence)
	}

	// LLM should not be called on cache hit
	if mockLLM.CallCount != 0 {
		t.Errorf("Expected LLM to not be called, but it was called %d times", mockLLM.CallCount)
	}
}

// TestClassifier_CacheMiss tests that the classifier calls LLM when no similar vector is found
func TestClassifier_CacheMiss(t *testing.T) {
	mockEmbedding := &testutil.MockEmbeddingClient{
		GenerateEmbeddingFunc: func(ctx context.Context, text string) ([]float32, error) {
			return []float32{0.1, 0.2, 0.3}, nil
		},
	}

	mockVectorContent := testutil.NewMockVectorClient()
	mockVectorContent.SearchFunc = func(ctx context.Context, vector []float32, topK int) ([]types.VectorMatch, error) {
		// Return low-similarity match (cache miss)
		return []types.VectorMatch{
			{
				ID:    "test-id",
				Score: 0.50, // Below threshold
				Metadata: map[string]any{
					"label": "test_label",
				},
			},
		}, nil
	}

	mockVectorLabel := testutil.NewMockVectorClient()
	mockLLM := &testutil.MockLLMClient{
		ClassifyFunc: func(ctx context.Context, text string) (string, error) {
			return "llm_generated_label", nil
		},
	}
	mockDSU := &testutil.MockDSUPersistence{}

	clf, err := classifier.NewClassifier(classifier.Config{
		EmbeddingClient:     mockEmbedding,
		VectorClientContent: mockVectorContent,
		VectorClientLabel:   mockVectorLabel,
		LLMClient:           mockLLM,
		DSUPersistence:      mockDSU,
	})
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	result, err := clf.Classify(context.Background(), "test text")
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}

	// Verify cache miss
	if result.CacheHit {
		t.Error("Expected cache miss, got cache hit")
	}

	if result.Label != "llm_generated_label" {
		t.Errorf("Expected label 'llm_generated_label', got '%s'", result.Label)
	}

	// LLM should be called on cache miss
	if mockLLM.CallCount != 1 {
		t.Errorf("Expected LLM to be called once, but it was called %d times", mockLLM.CallCount)
	}

	// Verify background tasks ran (vector upserts)
	time.Sleep(100 * time.Millisecond) // Give background tasks time to complete
	if mockVectorContent.UpsertCount < 1 {
		t.Error("Expected content vector to be upserted")
	}
	if mockVectorLabel.UpsertCount < 1 {
		t.Error("Expected label vector to be upserted")
	}
}

// TestClassifier_ContextCancellation tests that context cancellation is respected
func TestClassifier_ContextCancellation(t *testing.T) {
	mockEmbedding := &testutil.MockEmbeddingClient{
		GenerateEmbeddingFunc: func(ctx context.Context, text string) ([]float32, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		},
	}

	mockVectorContent := testutil.NewMockVectorClient()
	mockVectorLabel := testutil.NewMockVectorClient()
	mockLLM := &testutil.MockLLMClient{}
	mockDSU := &testutil.MockDSUPersistence{}

	clf, err := classifier.NewClassifier(classifier.Config{
		EmbeddingClient:     mockEmbedding,
		VectorClientContent: mockVectorContent,
		VectorClientLabel:   mockVectorLabel,
		LLMClient:           mockLLM,
		DSUPersistence:      mockDSU,
	})
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err = clf.Classify(ctx, "test text")
	if err == nil {
		t.Error("Expected error from cancelled context, got nil")
	}
}

// TestClassifier_GracefulShutdown tests that Close() waits for background tasks and saves DSU
func TestClassifier_GracefulShutdown(t *testing.T) {
	mockEmbedding := &testutil.MockEmbeddingClient{
		GenerateEmbeddingFunc: func(ctx context.Context, text string) ([]float32, error) {
			return []float32{0.1, 0.2, 0.3}, nil
		},
	}

	mockVectorContent := testutil.NewMockVectorClient()
	mockVectorLabel := testutil.NewMockVectorClient()
	mockLLM := &testutil.MockLLMClient{
		ClassifyFunc: func(ctx context.Context, text string) (string, error) {
			return "test_label", nil
		},
	}
	mockDSU := &testutil.MockDSUPersistence{}

	clf, err := classifier.NewClassifier(classifier.Config{
		EmbeddingClient:     mockEmbedding,
		VectorClientContent: mockVectorContent,
		VectorClientLabel:   mockVectorLabel,
		LLMClient:           mockLLM,
		DSUPersistence:      mockDSU,
	})
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	// Trigger a cache miss to start background tasks
	_, err = clf.Classify(context.Background(), "test text")
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}

	// Close should wait for background tasks and save DSU
	err = clf.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Verify DSU was saved
	if mockDSU.SaveCount != 1 {
		t.Errorf("Expected DSU to be saved once, but it was saved %d times", mockDSU.SaveCount)
	}

	// After closing, new classifications should be rejected
	_, err = clf.Classify(context.Background(), "test text 2")
	if err == nil {
		t.Error("Expected error when classifying after Close(), got nil")
	}
}

// TestClassifier_ErrorHandling tests error propagation
func TestClassifier_ErrorHandling(t *testing.T) {
	t.Run("embedding generation error", func(t *testing.T) {
		mockEmbedding := &testutil.MockEmbeddingClient{
			GenerateEmbeddingFunc: func(ctx context.Context, text string) ([]float32, error) {
				return nil, errors.New("embedding error")
			},
		}

		clf, err := classifier.NewClassifier(classifier.Config{
			EmbeddingClient:     mockEmbedding,
			VectorClientContent: testutil.NewMockVectorClient(),
			VectorClientLabel:   testutil.NewMockVectorClient(),
			LLMClient:           &testutil.MockLLMClient{},
			DSUPersistence:      &testutil.MockDSUPersistence{},
		})
		if err != nil {
			t.Fatalf("Failed to create classifier: %v", err)
		}

		_, err = clf.Classify(context.Background(), "test")
		if err == nil {
			t.Error("Expected error from embedding generation, got nil")
		}
	})

	t.Run("LLM classification error", func(t *testing.T) {
		mockLLM := &testutil.MockLLMClient{
			ClassifyFunc: func(ctx context.Context, text string) (string, error) {
				return "", errors.New("LLM error")
			},
		}

		clf, err := classifier.NewClassifier(classifier.Config{
			EmbeddingClient:     &testutil.MockEmbeddingClient{},
			VectorClientContent: testutil.NewMockVectorClient(),
			VectorClientLabel:   testutil.NewMockVectorClient(),
			LLMClient:           mockLLM,
			DSUPersistence:      &testutil.MockDSUPersistence{},
		})
		if err != nil {
			t.Fatalf("Failed to create classifier: %v", err)
		}

		_, err = clf.Classify(context.Background(), "test")
		if err == nil {
			t.Error("Expected error from LLM classification, got nil")
		}
	})
}

// TestClassifier_Metrics tests that metrics are tracked correctly
func TestClassifier_Metrics(t *testing.T) {
	mockEmbedding := &testutil.MockEmbeddingClient{}
	mockVectorContent := testutil.NewMockVectorClient()
	mockVectorLabel := testutil.NewMockVectorClient()
	mockLLM := &testutil.MockLLMClient{}
	mockDSU := &testutil.MockDSUPersistence{}

	clf, err := classifier.NewClassifier(classifier.Config{
		EmbeddingClient:     mockEmbedding,
		VectorClientContent: mockVectorContent,
		VectorClientLabel:   mockVectorLabel,
		LLMClient:           mockLLM,
		DSUPersistence:      mockDSU,
	})
	if err != nil {
		t.Fatalf("Failed to create classifier: %v", err)
	}

	// Simulate cache hit
	mockVectorContent.SearchFunc = func(ctx context.Context, vector []float32, topK int) ([]types.VectorMatch, error) {
		return []types.VectorMatch{
			{
				ID:    "test-id",
				Score: 0.95,
				Metadata: map[string]any{
					"label": "test_label",
				},
			},
		}, nil
	}

	_, err = clf.Classify(context.Background(), "test1")
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}

	metrics := clf.GetMetrics()
	if metrics.CacheHitRate != 100.0 {
		t.Errorf("Expected 100%% cache hit rate, got %.2f%%", metrics.CacheHitRate)
	}

	// Simulate cache miss
	mockVectorContent.SearchFunc = func(ctx context.Context, vector []float32, topK int) ([]types.VectorMatch, error) {
		return []types.VectorMatch{}, nil
	}
	mockLLM.ClassifyFunc = func(ctx context.Context, text string) (string, error) {
		return "new_label", nil
	}

	_, err = clf.Classify(context.Background(), "test2")
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}

	metrics = clf.GetMetrics()
	if metrics.CacheHitRate != 50.0 {
		t.Errorf("Expected 50%% cache hit rate, got %.2f%%", metrics.CacheHitRate)
	}
}

// TestClassifier_DSUPersistence tests DSU loading and saving
func TestClassifier_DSUPersistence(t *testing.T) {
	t.Run("load existing DSU", func(t *testing.T) {
		existingDSU := disjoint_set.NewDSU()
		existingDSU.Add("label1")
		existingDSU.Add("label2")

		mockDSU := &testutil.MockDSUPersistence{
			LoadFunc: func() (*disjoint_set.DSU, error) {
				return existingDSU, nil
			},
		}

		_, err := classifier.NewClassifier(classifier.Config{
			EmbeddingClient:     &testutil.MockEmbeddingClient{},
			VectorClientContent: testutil.NewMockVectorClient(),
			VectorClientLabel:   testutil.NewMockVectorClient(),
			LLMClient:           &testutil.MockLLMClient{},
			DSUPersistence:      mockDSU,
		})
		if err != nil {
			t.Fatalf("Failed to create classifier: %v", err)
		}
	})

	t.Run("handle DSU load error", func(t *testing.T) {
		mockDSU := &testutil.MockDSUPersistence{
			LoadFunc: func() (*disjoint_set.DSU, error) {
				return nil, errors.New("load error")
			},
		}

		_, err := classifier.NewClassifier(classifier.Config{
			EmbeddingClient:     &testutil.MockEmbeddingClient{},
			VectorClientContent: testutil.NewMockVectorClient(),
			VectorClientLabel:   testutil.NewMockVectorClient(),
			LLMClient:           &testutil.MockLLMClient{},
			DSUPersistence:      mockDSU,
		})
		if err == nil {
			t.Error("Expected error from DSU load failure, got nil")
		}
	})
}
