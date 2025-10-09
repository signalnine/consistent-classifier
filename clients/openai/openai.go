package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/FrenchMajesty/consistent-classifier/clients/groq"
	"github.com/FrenchMajesty/consistent-classifier/utils/retry"
	"github.com/google/uuid"
	openai "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

const openaiBaseURL = "https://api.openai.com/v1"

const EmbeddingVectorDimensions = 1024

var (
	client openai.Client
	once   sync.Once
)

// OpenAIClient is a minimal client for the OpenAI Chat API
type OpenAIClient struct {
	APIKey      string
	Env         string
	HTTPClient  *http.Client
	RetryConfig retry.Config
}

type LanguageModelClient interface {
	ChatCompletion(ctx context.Context, req groq.ChatCompletionRequest) (*groq.ChatCompletionResponse, error)
}

type OpenAIClientInterface interface {
	ChatCompletion(ctx context.Context, req groq.ChatCompletionRequest) (*groq.ChatCompletionResponse, error)
	ChatCompletionStream(ctx context.Context, req groq.ChatCompletionRequest, callback func(token string)) (*groq.StreamingResult, error)
	GenerateEmbedding(ctx context.Context, text string) ([]float32, error)
	GenerateEmbeddings(ctx context.Context, texts []string) ([][]float32, error)
}

// Ensure OpenAIClient implements GroqClientInterface for drop-in replacement
var _ groq.GroqClientInterface = (*OpenAIClient)(nil)

// Creates a new OpenAIClient
func NewOpenAIClient(apiKey string, env string) *OpenAIClient {
	client := &OpenAIClient{
		APIKey:      apiKey,
		Env:         env,
		HTTPClient:  http.DefaultClient,
		RetryConfig: retry.DefaultConfig(),
	}

	return client
}

// InitSingletonClient initializes the singleton client
func InitSingletonClient() openai.Client {
	once.Do(func() {
		client = NewClient()
	})

	return client
}

func NewClient() openai.Client {
	return openai.NewClient(
		option.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	)
}

// isRetryableError determines if an error should trigger a retry
func (c *OpenAIClient) isRetryableError(err error, statusCode int, responseBody []byte) bool {
	// Retry on network errors
	if err != nil {
		return true
	}

	// Retry on server errors (5xx)
	if statusCode >= 500 {
		return true
	}

	// Retry on rate limiting (429)
	if statusCode == 429 {
		return true
	}

	// OpenAI sometimes returns 400 for transient issues
	if statusCode == 400 {
		return true
	}

	// Check for failed_generation in response body even with 200 OK
	if statusCode == 200 && responseBody != nil {
		var errorResp groq.ChatCompletionResponseError
		if json.Unmarshal(responseBody, &errorResp) == nil {
			if errorResp.Error.FailedGeneration != "" ||
				strings.Contains(errorResp.Error.Message, "failed_generation") {
				return true
			}
		}

		// Also check if the response body contains "failed_generation" string
		if strings.Contains(string(responseBody), "failed_generation") {
			return true
		}
	}

	return false
}

// retryableRequest executes an HTTP request with retry logic
func (c *OpenAIClient) retryableRequest(ctx context.Context, url string, requestBody any, apiName string) ([]byte, error) {
	// Setup retry options
	opts := retry.Options{
		Config:       c.RetryConfig,
		ErrorChecker: c.isRetryableError,
		Logger:       log.Printf,
		APIName:      "OpenAI " + apiName,
	}

	// Define the retryable function
	retryableFn := func(attempt int) (interface{}, int, []byte, error) {
		body, err := json.Marshal(requestBody)
		if err != nil {
			return nil, 0, nil, fmt.Errorf("failed to marshal %s request: %w", apiName, err)
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(body))
		if err != nil {
			return nil, 0, nil, fmt.Errorf("failed to create HTTP request: %w", err)
		}
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := c.HTTPClient.Do(httpReq)
		if err != nil {
			return nil, 0, nil, err
		}
		defer resp.Body.Close()

		// Read the response body once
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, resp.StatusCode, nil, fmt.Errorf("failed to read %s response body: %w", apiName, err)
		}

		// Check if we should dump the request/response (only for chat completions)
		if apiName == "chat" {
			if os.Getenv("DUMP_LLM_REQUESTS") == "true" {
				if chatReq, ok := requestBody.(groq.ChatCompletionRequest); ok {
					saveResponseToFile(chatReq.Model, chatReq, bodyBytes, resp.StatusCode)
				}
			}
		}

		// If we get here and status is not OK, it's an error
		if resp.StatusCode != http.StatusOK {
			return nil, resp.StatusCode, bodyBytes, &groq.ChatCompletionError{
				Message:    fmt.Sprintf("openai %s API error %d", apiName, resp.StatusCode),
				StatusCode: resp.StatusCode,
				RawBody:    json.RawMessage(bodyBytes),
			}
		}

		return bodyBytes, resp.StatusCode, bodyBytes, nil
	}

	// Execute with retry logic
	result, err := retry.Execute(ctx, opts, retryableFn)
	if err != nil {
		return nil, err
	}

	return result.([]byte), nil
}

// Sends a chat completion request to OpenAI with retry logic
func (c *OpenAIClient) ChatCompletion(ctx context.Context, req groq.ChatCompletionRequest) (*groq.ChatCompletionResponse, error) {
	url := openaiBaseURL + "/chat/completions"

	bodyBytes, err := c.retryableRequest(ctx, url, req, "chat")
	if err != nil {
		return nil, err
	}

	// Parse the successful response
	var chatResp groq.ChatCompletionResponse
	if err := json.Unmarshal(bodyBytes, &chatResp); err != nil {
		return nil, &groq.ChatCompletionError{
			Message: fmt.Sprintf("failed to parse chat completion response: %v", err),
			RawBody: json.RawMessage(bodyBytes),
		}
	}

	return &chatResp, nil
}

// EmbeddingRequest represents a request to OpenAI embeddings API
type EmbeddingRequest struct {
	Input          []string `json:"input"`
	Model          string   `json:"model"`
	EncodingFormat string   `json:"encoding_format,omitempty"`
	Dimensions     int      `json:"dimensions,omitempty"`
}

// EmbeddingData represents a single embedding in the response
type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// EmbeddingResponse represents the response from OpenAI embeddings API
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  EmbeddingUsage  `json:"usage"`
}

// EmbeddingUsage represents token usage for embeddings
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// GenerateEmbedding generates an embedding for a single text
func (c *OpenAIClient) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := c.GenerateEmbeddings(ctx, []string{text})
	if err != nil {
		return nil, err
	}

	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return embeddings[0], nil
}

// GenerateEmbeddings generates embeddings for multiple texts
func (c *OpenAIClient) GenerateEmbeddings(ctx context.Context, texts []string) ([][]float32, error) {
	url := openaiBaseURL + "/embeddings"

	request := EmbeddingRequest{
		Input:          texts,
		Model:          "text-embedding-3-small",
		EncodingFormat: "float",
		Dimensions:     EmbeddingVectorDimensions,
	}

	bodyBytes, err := c.retryableRequest(ctx, url, request, "embeddings")
	if err != nil {
		return nil, err
	}

	// Parse the successful response
	var embeddingResp EmbeddingResponse
	if err := json.Unmarshal(bodyBytes, &embeddingResp); err != nil {
		return nil, fmt.Errorf("failed to parse embeddings response: %w", err)
	}

	// Extract embeddings in order
	embeddings := make([][]float32, len(texts))
	for _, data := range embeddingResp.Data {
		if data.Index >= 0 && data.Index < len(embeddings) {
			embeddings[data.Index] = data.Embedding
		}
	}

	return embeddings, nil
}

func saveResponseToFile(model string, req groq.ChatCompletionRequest, bodyBytes []byte, statusCode int) {
	// Create a unique filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	random := uuid.New().String()[:8]
	filename := fmt.Sprintf("openai_req_%s_%s.json", timestamp, random)

	// Create model-specific directory
	modelDir := fmt.Sprintf("llm_requests/%s", model)
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		log.Printf("Error creating directory %s: %v", modelDir, err)
		return
	}

	// Parse response body as JSON
	var responseBody interface{}
	if err := json.Unmarshal(bodyBytes, &responseBody); err != nil {
		log.Printf("Error parsing response body as JSON: %v", err)
		return
	}

	// Create a response object to save
	responseData := map[string]interface{}{
		"request":  req,
		"response": responseBody,
		"status":   statusCode,
	}

	// Marshal to JSON
	jsonData, err := json.MarshalIndent(responseData, "", "  ")
	if err != nil {
		log.Printf("Error marshaling response data: %v", err)
		return
	}

	// Write to file in model-specific directory
	filepath := filepath.Join(modelDir, filename)
	err = os.WriteFile(filepath, jsonData, 0644)
	if err != nil {
		log.Printf("Error writing to file %s: %v", filepath, err)
		return
	}
}

// ChatCompletionStream sends a streaming chat completion request to OpenAI
func (c *OpenAIClient) ChatCompletionStream(ctx context.Context, req groq.ChatCompletionRequest, callback func(token string)) (*groq.StreamingResult, error) {
	url := openaiBaseURL + "/chat/completions"

	// Ensure stream is enabled
	req.Stream = true

	// Setup retry options
	opts := retry.Options{
		Config:       c.RetryConfig,
		ErrorChecker: c.isRetryableError,
		Logger:       log.Printf,
		APIName:      "OpenAI",
	}

	var requestStartTime time.Time
	var firstTokenTime *time.Time

	// Define the retryable function
	retryableFn := func(attempt int) (interface{}, int, []byte, error) {
		// Reset timing for each retry attempt
		requestStartTime = time.Now()
		firstTokenTime = nil

		body, err := json.Marshal(req)
		if err != nil {
			return nil, 0, nil, fmt.Errorf("failed to marshal request: %w", err)
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(body))
		if err != nil {
			return nil, 0, nil, fmt.Errorf("failed to create HTTP request: %w", err)
		}
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")

		resp, err := c.HTTPClient.Do(httpReq)
		if err != nil {
			return nil, 0, nil, err
		}
		defer resp.Body.Close()

		// If we get here and status is not OK, it's an error
		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			return nil, resp.StatusCode, bodyBytes, &groq.ChatCompletionError{
				Message:    fmt.Sprintf("openai API error %d", resp.StatusCode),
				StatusCode: resp.StatusCode,
				RawBody:    json.RawMessage(bodyBytes),
			}
		}

		// Parse the streaming response with callback that tracks first token time
		var wrappedCallback func(string)
		if callback != nil {
			wrappedCallback = func(token string) {
				if firstTokenTime == nil {
					now := time.Now()
					firstTokenTime = &now
				}
				callback(token)
			}
		}

		response, err := c.parseStreamingResponse(ctx, resp.Body, wrappedCallback)
		if err != nil {
			// If it's a context cancellation, don't retry
			if ctx.Err() != nil {
				return nil, resp.StatusCode, nil, ctx.Err()
			}
			return nil, resp.StatusCode, nil, fmt.Errorf("failed to parse streaming response: %w", err)
		}

		// Check if we should dump the request/response for streaming (chat API only)
		if os.Getenv("DUMP_LLM_REQUESTS") == "true" {
			responseJSON, _ := json.Marshal(response)
			saveResponseToFile(req.Model, req, responseJSON, 200) // Use 200 for successful streaming
		}

		// Calculate TTFT if we captured first token time
		var ttftMs *int
		if firstTokenTime != nil {
			ttft := int(firstTokenTime.Sub(requestStartTime).Milliseconds())
			ttftMs = &ttft
		}

		// Create streaming result with metadata
		result := &groq.StreamingResult{
			Response:         response,
			TimeToFirstToken: ttftMs,
		}

		return result, resp.StatusCode, nil, nil
	}

	// Execute with retry logic
	result, err := retry.Execute(ctx, opts, retryableFn)
	if err != nil {
		return nil, err
	}

	return result.(*groq.StreamingResult), nil
}

// parseStreamingResponse parses Server-Sent Events from the OpenAI API response body
func (c *OpenAIClient) parseStreamingResponse(ctx context.Context, body io.Reader, callback func(token string)) (*groq.ChatCompletionResponse, error) {
	scanner := bufio.NewScanner(body)
	var finalResponse *groq.ChatCompletionResponse
	var fullContent strings.Builder

	// Track tool calls being built from deltas
	toolCallsMap := make(map[int]*groq.ToolCallRequest) // Map by index to accumulate tool calls

	for scanner.Scan() {
		// Check for context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		line := scanner.Text()

		// Skip empty lines
		if line == "" {
			continue
		}

		// Parse SSE data line
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")

			// Check for stream end
			if data == "[DONE]" {
				break
			}

			// Parse JSON chunk
			var chunk groq.ChatCompletionStreamResponse
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				// Skip malformed chunks but continue processing
				continue
			}

			// Process choices and extract content
			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]

				// Handle content delta
				if choice.Delta.Content != nil && *choice.Delta.Content != "" {
					token := *choice.Delta.Content
					fullContent.WriteString(token)
					callback(token)
				}

				// Handle tool call deltas
				if choice.Delta.ToolCalls != nil {
					for _, toolCallDelta := range *choice.Delta.ToolCalls {
						index := toolCallDelta.Index

						// Initialize tool call if this is the first delta for this index
						if toolCallsMap[index] == nil {
							toolCallsMap[index] = &groq.ToolCallRequest{
								Type: "function",
								Function: groq.ToolCallFunction{
									Arguments: "",
								},
							}
						}

						toolCall := toolCallsMap[index]

						// Accumulate tool call ID
						if toolCallDelta.ID != nil {
							toolCall.ID = *toolCallDelta.ID
						}

						// Accumulate tool call type
						if toolCallDelta.Type != nil {
							toolCall.Type = *toolCallDelta.Type
						}

						// Accumulate function name and arguments
						if toolCallDelta.Function != nil {
							if toolCallDelta.Function.Name != nil {
								toolCall.Function.Name = *toolCallDelta.Function.Name
							}
							if toolCallDelta.Function.Arguments != nil {
								toolCall.Function.Arguments += *toolCallDelta.Function.Arguments
							}
						}
					}
				}

				// Build final response from first chunk or update existing
				if finalResponse == nil {
					finalResponse = &groq.ChatCompletionResponse{
						ID:     chunk.ID,
						Object: chunk.Object,
						Choices: []groq.ChatCompletionChoice{
							{
								Index: choice.Index,
								Message: groq.ChatMessage{
									Role:    groq.MessageRoleAssistant,
									Content: &[]string{""}[0], // Will be updated with full content
								},
								FinishReason: "",
							},
						},
					}
				}

				// Update finish reason if provided
				if choice.FinishReason != nil {
					finalResponse.Choices[0].FinishReason = *choice.FinishReason
				}

				// Update usage if provided (typically in final chunk)
				if chunk.Usage != nil {
					finalResponse.Usage = *chunk.Usage
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading streaming response: %w", err)
	}

	if finalResponse == nil {
		return nil, fmt.Errorf("no valid response received from stream")
	}

	// Set the full content
	fullContentStr := fullContent.String()
	finalResponse.Choices[0].Message.Content = &fullContentStr

	// Assemble complete tool calls from accumulated deltas
	if len(toolCallsMap) > 0 {
		toolCalls := make([]groq.ToolCallRequest, 0, len(toolCallsMap))

		// Convert map to ordered slice (by index)
		for i := 0; i < len(toolCallsMap); i++ {
			if toolCall := toolCallsMap[i]; toolCall != nil {
				toolCalls = append(toolCalls, *toolCall)
			}
		}

		if len(toolCalls) > 0 {
			finalResponse.Choices[0].Message.ToolCalls = &toolCalls
		}
	}

	return finalResponse, nil
}
