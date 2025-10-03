package benchmark

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/google/uuid"
)

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
