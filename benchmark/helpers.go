package benchmark

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/google/uuid"
)

const MAX_DATASET_SIZE = 500

// saveMetricsToFile saves the metrics to a file
func saveMetricsToFile(metrics BenchmarkMetrics) error {
	timestamp := time.Now().Format("20060102_150405")
	random := uuid.New().String()[:8]
	filename := fmt.Sprintf("metrics_%s_%s.json", timestamp, random)

	jsonData, err := json.Marshal(metrics)
	if err != nil {
		return err
	}

	err = os.WriteFile(filename, jsonData, 0644)
	if err != nil {
		return err
	}

	return nil
}

// loadDataset loads the dataset from the file
func loadDataset() ([]DatasetItem, error) {
	filepath := os.Getenv("DATASET_FILEPATH")
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open dataset file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to parse CSV: %w", err)
	}

	if len(records) < 2 {
		return nil, fmt.Errorf("dataset file must have at least a header and one row")
	}

	// Skip header row (index 0), parse data rows
	dataset := make([]DatasetItem, 0, len(records)-1)
	for _, record := range records[1:] {
		if len(record) < 3 {
			continue // Skip malformed rows
		}
		dataset = append(dataset, DatasetItem{
			UserResponse: record[1], // user_response column
			UserCategory: record[2], // user_category column
		})
	}

	return trimDataset(dataset), nil
}

// trimDataset trims the dataset to 500 items
func trimDataset(dataset []DatasetItem) []DatasetItem {
	if len(dataset) > MAX_DATASET_SIZE {
		return dataset[:MAX_DATASET_SIZE]
	}
	return dataset
}
