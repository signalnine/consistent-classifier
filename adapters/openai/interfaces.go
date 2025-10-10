package openai

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/FrenchMajesty/consistent-classifier/internal/retry"
)

// OpenAIClient is a minimal client for the OpenAI Chat API
type OpenAIClient struct {
	APIKey       string
	Env          string
	DumpRequests bool
	BaseURL      string
	HTTPClient   *http.Client
	RetryConfig  retry.Config
}

type LanguageModelClient interface {
	ChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error)
	SetBaseURL(baseUrl string)
}

// ChatCompletionRequest is the request body for the chat completion endpoint
type ChatCompletionRequest struct {
	Model               string          `json:"model"`
	User                string          `json:"user,omitempty"`
	Messages            []ChatMessage   `json:"messages"`
	MaxCompletionTokens int             `json:"max_completion_tokens,omitempty"`
	Temperature         float32         `json:"temperature,omitempty"`
	ToolChoice          any             `json:"tool_choice,omitempty"`
	PresencePenalty     float32         `json:"presence_penalty,omitempty"`
	FrequencyPenalty    float32         `json:"frequency_penalty,omitempty"`
	ResponseFormat      *ResponseFormat `json:"response_format,omitempty"`
	ReasoningEffort     ReasoningEffort `json:"reasoning_effort,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
}

type ReasoningEffort string

const (
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
)

type ResponseFormat struct {
	Type       string         `json:"type,omitempty"`
	JsonSchema map[string]any `json:"json_schema,omitempty"`
}

type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// The response from the chat completion endpoint
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   ChatCompletionUsage    `json:"usage"`
}

type ChatCompletionUsage struct {
	PromptTokens        int                 `json:"prompt_tokens"`
	CompletionTokens    int                 `json:"completion_tokens"`
	TotalTokens         int                 `json:"total_tokens"`
	PromptTokensDetails *PromptTokenDetails `json:"prompt_tokens_details,omitempty"`
}

type PromptTokenDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type MessageRole string

const (
	MessageRoleUser      MessageRole = "user"
	MessageRoleAssistant MessageRole = "assistant"
	MessageRoleTool      MessageRole = "tool"
	MessageRoleSystem    MessageRole = "system"
)

type ChatMessage struct {
	Role       MessageRole `json:"role"`
	Content    *string     `json:"content,omitempty"`
	Reasoning  *string     `json:"reasoning,omitempty"`
	ToolCallID *string     `json:"tool_call_id,omitempty"`
}

type ChatError struct {
	Code             string `json:"code"`
	Message          string `json:"message"`
	Type             string `json:"type"`
	FailedGeneration string `json:"failed_generation,omitempty"`
}

type ChatCompletionResponseError struct {
	Error ChatError `json:"error"`
}

// ChatCompletionError wraps standard errors with raw response body for error logging
type ChatCompletionError struct {
	Message    string          `json:"message"`
	StatusCode int             `json:"status_code,omitempty"`
	RawBody    json.RawMessage `json:"raw_body,omitempty"`
}

func (e *ChatCompletionError) Error() string {
	return e.Message
}

// GetRawResponseBody returns the raw response body if available
func (e *ChatCompletionError) GetRawResponseBody() json.RawMessage {
	return e.RawBody
}
