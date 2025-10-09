package classifier

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/FrenchMajesty/consistent-classifier/utils/disjoint_set"
)

// FileDSUPersistence implements DSUPersistence using file-based storage
type FileDSUPersistence struct {
	filepath string
}

// NewFileDSUPersistence creates a new file-based DSU persistence handler
func NewFileDSUPersistence(filepath string) *FileDSUPersistence {
	return &FileDSUPersistence{
		filepath: filepath,
	}
}

// Load loads the DSU from the file. If the file doesn't exist, returns a new empty DSU.
func (f *FileDSUPersistence) Load() (*disjoint_set.DSU, error) {
	// Check if file exists
	if _, err := os.Stat(f.filepath); os.IsNotExist(err) {
		// File doesn't exist, return empty DSU
		return disjoint_set.NewDSU(), nil
	}

	// Read file contents
	data, err := os.ReadFile(f.filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read DSU from file %s: %w", f.filepath, err)
	}

	// Unmarshal into new DSU
	dsu := disjoint_set.NewDSU()
	if err := json.Unmarshal(data, dsu); err != nil {
		return nil, fmt.Errorf("failed to unmarshal DSU from file %s: %w", f.filepath, err)
	}

	return dsu, nil
}

// Save saves the DSU to the file
func (f *FileDSUPersistence) Save(dsu *disjoint_set.DSU) error {
	// Marshal DSU to JSON
	data, err := json.Marshal(dsu)
	if err != nil {
		return fmt.Errorf("failed to marshal DSU: %w", err)
	}

	// Write to file
	if err := os.WriteFile(f.filepath, data, 0644); err != nil {
		return fmt.Errorf("failed to write DSU to file %s: %w", f.filepath, err)
	}

	return nil
}
