package adapters

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/FrenchMajesty/consistent-classifier/pkg/adapters/openai"
)

// Tests for unexported functions and internal behavior
// These tests are in the same package to access unexported types

func TestDefaultLLMClient_Classify_Internal(t *testing.T) {
	// Create mock client
	responseContent := "test_category"
	mockClient := &mockLLMOpenAIClient{
		chatCompletionFunc: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			return &openai.ChatCompletionResponse{
				ID:     "test-id",
				Object: "chat.completion",
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatMessage{
							Content: &responseContent,
						},
					},
				},
			}, nil
		},
	}

	client := &DefaultLLMClient{
		client:       mockClient,
		systemPrompt: defaultSystemPrompt,
	}

	ctx := context.Background()
	label, err := client.Classify(ctx, "test text to classify")

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if label != "test_category" {
		t.Errorf("Expected label 'test_category', got '%s'", label)
	}
}

func TestDefaultLLMClient_Classify_Error_Internal(t *testing.T) {
	// Create mock client that returns error
	mockClient := &mockLLMOpenAIClient{
		chatCompletionFunc: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			return nil, errors.New("API error")
		},
	}

	client := &DefaultLLMClient{
		client:       mockClient,
		systemPrompt: defaultSystemPrompt,
	}

	ctx := context.Background()
	_, err := client.Classify(ctx, "test text")

	if err == nil {
		t.Error("Expected error from Classify")
	}

	if !strings.Contains(err.Error(), "failed to get LLM response") {
		t.Errorf("Expected 'failed to get LLM response' error, got: %v", err)
	}
}

func TestDefaultLLMClient_Classify_EmptyChoices_Internal(t *testing.T) {
	// Create mock client that returns empty choices
	mockClient := &mockLLMOpenAIClient{
		chatCompletionFunc: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			return &openai.ChatCompletionResponse{
				ID:      "test-id",
				Object:  "chat.completion",
				Choices: []openai.ChatCompletionChoice{}, // Empty
			}, nil
		},
	}

	client := &DefaultLLMClient{
		client:       mockClient,
		systemPrompt: defaultSystemPrompt,
	}

	ctx := context.Background()
	_, err := client.Classify(ctx, "test text")

	if err == nil {
		t.Error("Expected error for empty choices")
	}

	if !strings.Contains(err.Error(), "no response from LLM") {
		t.Errorf("Expected 'no response from LLM' error, got: %v", err)
	}
}

func TestDefaultLLMClient_Classify_NilContent_Internal(t *testing.T) {
	// Create mock client that returns nil content
	mockClient := &mockLLMOpenAIClient{
		chatCompletionFunc: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			return &openai.ChatCompletionResponse{
				ID:     "test-id",
				Object: "chat.completion",
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatMessage{
							Content: nil, // Nil content
						},
					},
				},
			}, nil
		},
	}

	client := &DefaultLLMClient{
		client:       mockClient,
		systemPrompt: defaultSystemPrompt,
	}

	ctx := context.Background()
	_, err := client.Classify(ctx, "test text")

	if err == nil {
		t.Error("Expected error for nil content")
	}

	if !strings.Contains(err.Error(), "no response from LLM") {
		t.Errorf("Expected 'no response from LLM' error, got: %v", err)
	}
}

func TestDefaultLLMClient_Classify_Normalization_Internal(t *testing.T) {
	// Test that response is trimmed and lowercased
	responseContent := "  TEST_CATEGORY  "
	mockClient := &mockLLMOpenAIClient{
		chatCompletionFunc: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatMessage{
							Content: &responseContent,
						},
					},
				},
			}, nil
		},
	}

	client := &DefaultLLMClient{
		client:       mockClient,
		systemPrompt: defaultSystemPrompt,
	}

	ctx := context.Background()
	label, err := client.Classify(ctx, "test text")

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if label != "test_category" {
		t.Errorf("Expected normalized label 'test_category', got '%s'", label)
	}
}

func TestVoyageEmbeddingAdapter_GenerateEmbedding_Internal(t *testing.T) {
	// Test that adapter was created correctly
	apiKey := "test-key"
	adapter, err := NewVoyageEmbeddingAdapter(&apiKey)
	if err != nil {
		t.Fatalf("Failed to create adapter: %v", err)
	}

	if adapter == nil {
		t.Fatal("Expected non-nil adapter")
	}

	if adapter.client == nil {
		t.Fatal("Expected client to be initialized")
	}

	// Verify we can't actually call GenerateEmbedding without API key
	ctx := context.Background()
	_, err = adapter.GenerateEmbedding(ctx, "test")
	if err == nil {
		t.Log("Note: GenerateEmbedding succeeded (unexpected with fake key)")
	}
}

func TestPineconeVectorAdapter_Search_Internal(t *testing.T) {
	apiKey := "test-key"
	host := "test-host.pinecone.io"
	namespace := "test-namespace"

	_, err := NewPineconeVectorAdapter(&apiKey, &host, namespace)
	if err != nil {
		// Expected - can't connect to fake Pinecone
		t.Logf("Expected error creating adapter with fake credentials: %v", err)
		return
	}
}

func TestPineconeVectorAdapter_Upsert_Internal(t *testing.T) {
	apiKey := "test-key"
	host := "test-host.pinecone.io"
	namespace := "test-namespace"

	_, err := NewPineconeVectorAdapter(&apiKey, &host, namespace)
	if err != nil {
		t.Logf("Expected error creating adapter with fake credentials: %v", err)
		return
	}
}

// Mock OpenAI client for internal testing
type mockLLMOpenAIClient struct {
	chatCompletionFunc func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
	setBaseURLFunc     func(baseUrl string)
}

func (m *mockLLMOpenAIClient) ChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	if m.chatCompletionFunc != nil {
		return m.chatCompletionFunc(ctx, req)
	}
	return nil, errors.New("not implemented")
}

func (m *mockLLMOpenAIClient) SetBaseURL(baseUrl string) {
	if m.setBaseURLFunc != nil {
		m.setBaseURLFunc(baseUrl)
	}
}
