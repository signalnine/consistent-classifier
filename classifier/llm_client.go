package classifier

import (
	"context"
	"fmt"
	"strings"

	"github.com/FrenchMajesty/consistent-classifier/clients/openai"
)

// DefaultLLMClient implements LLMClient using OpenAI
type DefaultLLMClient struct {
	client       openai.LanguageModelClient
	systemPrompt string
}

const defaultSystemPrompt = `You are a text classification assistant. Given a text, classify it into a concise category label.

Rules:
- Return ONLY the category label, nothing else
- Use lowercase with underscores (e.g., "technical_question", "expressing_gratitude")
- Keep labels short and descriptive (2-5 words max)
- Be consistent: similar texts should get the same label`

// NewDefaultLLMClient creates a new LLM client using OpenAI with API key from environment
func NewDefaultLLMClient(apiKey *string, systemPrompt string) *DefaultLLMClient {
	loadEnvVar(apiKey, "OPENAI_API_KEY")

	instance := DefaultLLMClient{
		client:       openai.NewClient(*apiKey),
		systemPrompt: defaultSystemPrompt,
	}

	if systemPrompt != "" {
		instance.systemPrompt = systemPrompt
	}

	return &instance
}

// Classify classifies text into a category label using LLM
func (c *DefaultLLMClient) Classify(ctx context.Context, text string) (string, error) {
	userPrompt := fmt.Sprintf("Text to classify: \"%s\"", text)

	req := openai.ChatCompletionRequest{
		Model: "gpt-4o-mini",
		Messages: []openai.ChatMessage{
			{
				Role:    openai.MessageRoleSystem,
				Content: &c.systemPrompt,
			},
			{
				Role:    openai.MessageRoleUser,
				Content: &userPrompt,
			},
		},
		Temperature:         0.3,
		MaxCompletionTokens: 50,
	}

	resp, err := c.client.ChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("failed to get LLM response: %w", err)
	}

	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == nil {
		return "", fmt.Errorf("no response from LLM")
	}

	label := strings.TrimSpace(*resp.Choices[0].Message.Content)
	label = strings.ToLower(label)

	return label, nil
}
