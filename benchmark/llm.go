package benchmark

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/FrenchMajesty/consistent-classifier/clients/openai"
)

func LLM(limit int) {
	dataset, err := loadDataset(limit)
	if err != nil {
		log.Fatal(err)
	}

	results := make([]Result, 0)
	labels := make(map[string]bool)
	startTime := time.Now()
	benchmarkMetrics := BenchmarkMetrics{
		TotalTweets: len(dataset),
	}

	for _, tweet := range dataset {
		tweetStartTime := time.Now()
		label, tokenUsage, err := classifyTextWithLLM(tweet.Content, tweet.UserResponse)
		if err != nil {
			log.Fatal(err)
		}

		benchmarkMetrics.ProcessingTime = append(benchmarkMetrics.ProcessingTime, time.Since(tweetStartTime))
		benchmarkMetrics.TokenUsage = append(benchmarkMetrics.TokenUsage, *tokenUsage)
		results = append(results, Result{
			Post:       tweet.Content,
			Reply:      tweet.UserResponse,
			ReplyLabel: label,
		})
		labels[label] = true
	}

	benchmarkMetrics.TotalDuration = time.Since(startTime)
	benchmarkMetrics.UniqueLabels = len(labels)
	benchmarkMetrics.ConvergedLabels = len(labels)

	err = saveMetricsToFile(benchmarkMetrics)
	if err != nil {
		log.Fatal(err)
	}

	err = saveResultsToFile(results)
	if err != nil {
		log.Fatal(err)
	}
}

func classifyTextWithLLM(post, reply string) (string, *TokenUsageMetrics, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return "", nil, fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}

	client := openai.NewClient(apiKey)

	systemPrompt := `You are a text classification assistant. Given a user's response/reply, classify it into a concise category label.

Rules:
- Return ONLY the category label, nothing else
- Use lowercase with underscores (e.g., "technical_question", "expressing_gratitude")
- Keep labels short and descriptive (2-5 words max)
- Be consistent: similar responses should get the same label`

	userPrompt := fmt.Sprintf("Original Post: %s\n\n\nUser Response: \"%s\"", post, reply)

	req := openai.ChatCompletionRequest{
		Model: "gpt-4.1-mini",
		Messages: []openai.ChatMessage{
			{
				Role:    openai.MessageRoleSystem,
				Content: &systemPrompt,
			},
			{
				Role:    openai.MessageRoleUser,
				Content: &userPrompt,
			},
		},
		Temperature:         0.3,
		MaxCompletionTokens: 50,
	}

	ctx := context.Background()
	resp, err := client.ChatCompletion(ctx, req)
	if err != nil {
		return "", nil, fmt.Errorf("failed to get LLM response: %w", err)
	}

	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == nil {
		return "", nil, fmt.Errorf("no response from LLM")
	}

	label := strings.TrimSpace(*resp.Choices[0].Message.Content)
	label = strings.ToLower(label)

	tokenUsage := TokenUsageMetrics{
		InputTokens:       resp.Usage.PromptTokens,
		CachedInputTokens: resp.Usage.PromptTokensDetails.CachedTokens,
		OutputTokens:      resp.Usage.CompletionTokens,
	}

	return label, &tokenUsage, nil
}
