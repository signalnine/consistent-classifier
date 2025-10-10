package adapters_test

import (
	"os"
	"testing"

	"github.com/FrenchMajesty/consistent-classifier/pkg/adapters"
)

// Tests

func TestNewDefaultLLMClient_WithAPIKey(t *testing.T) {
	apiKey := "test-openai-key"
	client, err := adapters.NewDefaultLLMClient(&apiKey, "", "", "")

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if client == nil {
		t.Fatal("Expected non-nil client")
	}
}

func TestNewDefaultLLMClient_FromEnv(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-openai-key")

	client, err := adapters.NewDefaultLLMClient(nil, "", "", "")

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if client == nil {
		t.Fatal("Expected non-nil client")
	}
}

func TestNewDefaultLLMClient_MissingKey(t *testing.T) {
	os.Unsetenv("OPENAI_API_KEY")

	_, err := adapters.NewDefaultLLMClient(nil, "", "", "")

	if err == nil {
		t.Error("Expected error when API key is missing, got nil")
	}
}

func TestNewDefaultLLMClient_CustomPrompt(t *testing.T) {
	apiKey := "test-key"
	customPrompt := "You are a custom classifier"

	client, err := adapters.NewDefaultLLMClient(&apiKey, customPrompt, "", "")

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if client == nil {
		t.Fatal("Expected non-nil client")
	}
}

func TestNewDefaultLLMClient_DefaultPrompt(t *testing.T) {
	apiKey := "test-key"

	client, err := adapters.NewDefaultLLMClient(&apiKey, "", "", "")

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if client == nil {
		t.Fatal("Expected non-nil client")
	}
}

// Note: The following tests demonstrate testing strategy but cannot run without mocking the internal client
// In a real scenario, we would inject the OpenAI client as a dependency

func TestDefaultLLMClient_ClassifyLogic(t *testing.T) {
	// This test demonstrates the classification logic

	// Test cases for label normalization
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"lowercase", "  UPPERCASE_LABEL  ", "uppercase_label"},
		{"trim spaces", "  spaced_label  ", "spaced_label"},
		{"already normalized", "normal_label", "normal_label"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// The actual Classify method does:
			// label := strings.TrimSpace(*resp.Choices[0].Message.Content)
			// label = strings.ToLower(label)

			// We're just verifying the normalization logic conceptually here
			// Full testing would require dependency injection of the OpenAI client
		})
	}
}

func TestDefaultLLMClient_ErrorScenarios(t *testing.T) {
	// These tests demonstrate error handling scenarios

	t.Run("API error", func(t *testing.T) {
		// Would test handling of API errors from OpenAI
	})

	t.Run("empty response", func(t *testing.T) {
		// Would test handling of empty choices array
	})

	t.Run("nil content", func(t *testing.T) {
		// Would test handling of nil message content
	})
}
