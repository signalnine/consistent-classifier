package openai

import (
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
	"time"

	"github.com/FrenchMajesty/consistent-classifier/internal/retry"
	"github.com/google/uuid"
)

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
		var errorResp ChatCompletionResponseError
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

// createAndRunRetryableRequest executes an HTTP request with retry logic
func (c *OpenAIClient) createAndRunRetryableRequest(ctx context.Context, url string, requestBody any, apiName string) ([]byte, error) {
	// Setup retry options
	opts := retry.Options{
		Config:       c.RetryConfig,
		ErrorChecker: c.isRetryableError,
		Logger:       log.Printf,
		APIName:      "OpenAI " + apiName,
	}

	// Define the retryable function
	retryableFn := c.buildRetryableFn(ctx, url, requestBody, apiName)

	// Execute with retry logic
	result, err := retry.Execute(ctx, opts, retryableFn)
	if err != nil {
		return nil, err
	}

	return result.([]byte), nil
}

// buildRetryableFn builds a retryable function for the given request body
func (c *OpenAIClient) buildRetryableFn(ctx context.Context, url string, requestBody any, apiName string) retry.RetryableFunc {
	retryableFn := func(attempt int) (any, int, []byte, error) {
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

		// Dump the request/response if enabled for debugging purposes
		chatReq, ok := requestBody.(ChatCompletionRequest)
		if c.DumpRequests && ok {
			saveResponseToFile(chatReq.Model, chatReq, bodyBytes, resp.StatusCode)
		}

		// If we get here and status is not OK, it's an error
		if resp.StatusCode != http.StatusOK {
			return nil, resp.StatusCode, bodyBytes, &ChatCompletionError{
				Message:    fmt.Sprintf("openai %s API error %d", apiName, resp.StatusCode),
				StatusCode: resp.StatusCode,
				RawBody:    json.RawMessage(bodyBytes),
			}
		}

		return bodyBytes, resp.StatusCode, bodyBytes, nil
	}

	return retryableFn
}

// saveResponseToFile saves the request/response to a file for debugging purposes
func saveResponseToFile(model string, req ChatCompletionRequest, bodyBytes []byte, statusCode int) {
	// Create a unique filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	random := uuid.New().String()[:8]
	filename := fmt.Sprintf("openai_req_%s_%s.json", timestamp, random)

	// Create model-specific directory
	modelDir := fmt.Sprintf("debug_llm_requests/%s", model)
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		log.Printf("Error creating directory %s: %v", modelDir, err)
		return
	}

	// Parse response body as JSON
	var responseBody any
	if err := json.Unmarshal(bodyBytes, &responseBody); err != nil {
		log.Printf("Error parsing response body as JSON: %v", err)
		return
	}

	// Create a response object to save
	responseData := map[string]any{
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
