package classifier

import (
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
	dsu := disjoint_set.NewDSU()

	// Check if file exists
	if _, err := os.Stat(f.filepath); os.IsNotExist(err) {
		// File doesn't exist, return empty DSU
		return dsu, nil
	}

	// File exists, load it
	loadedDSU, err := dsu.ReadFromFile(f.filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to load DSU from file %s: %w", f.filepath, err)
	}

	return loadedDSU, nil
}

// Save saves the DSU to the file
func (f *FileDSUPersistence) Save(dsu *disjoint_set.DSU) error {
	err := dsu.WriteToFile(f.filepath)
	if err != nil {
		return fmt.Errorf("failed to save DSU to file %s: %w", f.filepath, err)
	}
	return nil
}
