package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/FrenchMajesty/consistent-classifier/internal/retry"
)

const openaiBaseURL = "https://api.openai.com/v1"

// Creates a new OpenAIClient
func NewClient(apiKey string) *OpenAIClient {
	client := &OpenAIClient{
		APIKey:      apiKey,
		HTTPClient:  http.DefaultClient,
		RetryConfig: retry.DefaultConfig(),
		BaseURL:     openaiBaseURL,
	}

	return client
}

var _ LanguageModelClient = (*OpenAIClient)(nil)

// Sends a chat completion request to OpenAI with retry logic
func (c *OpenAIClient) ChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	url := c.BaseURL + "/chat/completions"

	bodyBytes, err := c.createAndRunRetryableRequest(ctx, url, req, "chat")
	if err != nil {
		return nil, err
	}

	// Parse the successful response
	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(bodyBytes, &chatResp); err != nil {
		return nil, &ChatCompletionError{
			Message: fmt.Sprintf("failed to parse chat completion response: %v", err),
			RawBody: json.RawMessage(bodyBytes),
		}
	}

	return &chatResp, nil
}

// Sets the base URL for the OpenAI client
func (c *OpenAIClient) SetBaseURL(baseUrl string) {
	c.BaseURL = baseUrl
}
