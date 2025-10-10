package classifier_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/FrenchMajesty/consistent-classifier/pkg/classifier"
	"github.com/FrenchMajesty/consistent-classifier/utils/disjoint_set"
)

func TestFileDSUPersistence_Load_NonExistentFile(t *testing.T) {
	tempDir := t.TempDir()
	filepath := filepath.Join(tempDir, "non_existent.bin")

	persistence := classifier.NewFileDSUPersistence(filepath)
	dsu, err := persistence.Load()

	if err != nil {
		t.Fatalf("Expected no error when loading non-existent file, got: %v", err)
	}

	if dsu == nil {
		t.Fatal("Expected DSU to be non-nil")
	}

	// Should return empty DSU
	if dsu.Size() != 0 {
		t.Errorf("Expected empty DSU, got size: %d", dsu.Size())
	}
}

func TestFileDSUPersistence_Load_ValidFile(t *testing.T) {
	tempDir := t.TempDir()
	filepath := filepath.Join(tempDir, "valid.bin")

	// Create a DSU and save it
	originalDSU := disjoint_set.NewDSU()
	originalDSU.Add("label1")
	originalDSU.Add("label2")
	originalDSU.Add("label3")
	originalDSU.Union(0, 1) // Merge label1 and label2

	// Marshal and write
	data, err := json.Marshal(originalDSU)
	if err != nil {
		t.Fatalf("Failed to marshal DSU: %v", err)
	}
	err = os.WriteFile(filepath, data, 0644)
	if err != nil {
		t.Fatalf("Failed to write file: %v", err)
	}

	// Now load it
	persistence := classifier.NewFileDSUPersistence(filepath)
	loadedDSU, err := persistence.Load()

	if err != nil {
		t.Fatalf("Failed to load DSU: %v", err)
	}

	if loadedDSU.Size() != 3 {
		t.Errorf("Expected DSU size 3, got: %d", loadedDSU.Size())
	}

	// Check that labels were loaded
	labels := loadedDSU.Labels()
	expectedLabels := map[string]bool{
		"label1": true,
		"label2": true,
		"label3": true,
	}
	for _, label := range labels {
		if !expectedLabels[label] {
			t.Errorf("Unexpected label: %s", label)
		}
	}
}

func TestFileDSUPersistence_Load_CorruptedJSON(t *testing.T) {
	tempDir := t.TempDir()
	filepath := filepath.Join(tempDir, "corrupted.bin")

	// Write invalid JSON
	err := os.WriteFile(filepath, []byte("{invalid json}"), 0644)
	if err != nil {
		t.Fatalf("Failed to write file: %v", err)
	}

	persistence := classifier.NewFileDSUPersistence(filepath)
	_, err = persistence.Load()

	if err == nil {
		t.Error("Expected error when loading corrupted JSON, got nil")
	}
}

func TestFileDSUPersistence_Load_ReadError(t *testing.T) {
	tempDir := t.TempDir()
	filepath := filepath.Join(tempDir, "unreadable.bin")

	// Create file
	err := os.WriteFile(filepath, []byte("{}"), 0644)
	if err != nil {
		t.Fatalf("Failed to write file: %v", err)
	}

	// Make it unreadable
	err = os.Chmod(filepath, 0000)
	if err != nil {
		t.Fatalf("Failed to chmod file: %v", err)
	}
	defer os.Chmod(filepath, 0644) // Cleanup

	persistence := classifier.NewFileDSUPersistence(filepath)
	_, err = persistence.Load()

	if err == nil {
		t.Error("Expected error when loading unreadable file, got nil")
	}
}

func TestFileDSUPersistence_Save_Success(t *testing.T) {
	tempDir := t.TempDir()
	filepath := filepath.Join(tempDir, "save_test.bin")

	dsu := disjoint_set.NewDSU()
	dsu.Add("label1")
	dsu.Add("label2")

	persistence := classifier.NewFileDSUPersistence(filepath)
	err := persistence.Save(dsu)

	if err != nil {
		t.Fatalf("Failed to save DSU: %v", err)
	}

	// Verify file was created
	_, err = os.Stat(filepath)
	if err != nil {
		t.Errorf("Expected file to exist: %v", err)
	}

	// Verify content is valid JSON
	data, err := os.ReadFile(filepath)
	if err != nil {
		t.Fatalf("Failed to read file: %v", err)
	}

	var loadedDSU disjoint_set.DSU
	err = json.Unmarshal(data, &loadedDSU)
	if err != nil {
		t.Errorf("Saved file is not valid JSON: %v", err)
	}
}

func TestFileDSUPersistence_Save_WriteError(t *testing.T) {
	// Try to write to a directory that doesn't exist
	filepath := "/nonexistent/directory/file.bin"

	dsu := disjoint_set.NewDSU()
	dsu.Add("label1")

	persistence := classifier.NewFileDSUPersistence(filepath)
	err := persistence.Save(dsu)

	if err == nil {
		t.Error("Expected error when saving to nonexistent directory, got nil")
	}
}

func TestFileDSUPersistence_RoundTrip(t *testing.T) {
	tempDir := t.TempDir()
	filepath := filepath.Join(tempDir, "roundtrip.bin")

	// Create original DSU with some structure
	originalDSU := disjoint_set.NewDSU()
	originalDSU.Add("technical_question")
	originalDSU.Add("tech_query")
	originalDSU.Add("asking_technical")
	originalDSU.Add("expressing_gratitude")
	originalDSU.Add("saying_thanks")

	// Merge some labels
	idx1 := originalDSU.FindOrCreate("technical_question")
	idx2 := originalDSU.FindOrCreate("tech_query")
	idx3 := originalDSU.FindOrCreate("asking_technical")
	originalDSU.Union(idx1, idx2)
	originalDSU.Union(idx2, idx3)

	idx4 := originalDSU.FindOrCreate("expressing_gratitude")
	idx5 := originalDSU.FindOrCreate("saying_thanks")
	originalDSU.Union(idx4, idx5)

	// Save
	persistence := classifier.NewFileDSUPersistence(filepath)
	err := persistence.Save(originalDSU)
	if err != nil {
		t.Fatalf("Failed to save DSU: %v", err)
	}

	// Load
	loadedDSU, err := persistence.Load()
	if err != nil {
		t.Fatalf("Failed to load DSU: %v", err)
	}

	// Verify structure is preserved
	if loadedDSU.Size() != 5 {
		t.Errorf("Expected size 5, got: %d", loadedDSU.Size())
	}

	// Verify the clusters are preserved (should have 2 clusters)
	if loadedDSU.CountSets() != 2 {
		t.Errorf("Expected 2 clusters, got: %d", loadedDSU.CountSets())
	}

	// Verify specific labels are connected
	idx1Loaded := loadedDSU.FindOrCreate("technical_question")
	idx2Loaded := loadedDSU.FindOrCreate("tech_query")
	if !loadedDSU.Connected(idx1Loaded, idx2Loaded) {
		t.Error("Expected 'technical_question' and 'tech_query' to be connected")
	}

	idx4Loaded := loadedDSU.FindOrCreate("expressing_gratitude")
	idx5Loaded := loadedDSU.FindOrCreate("saying_thanks")
	if !loadedDSU.Connected(idx4Loaded, idx5Loaded) {
		t.Error("Expected 'expressing_gratitude' and 'saying_thanks' to be connected")
	}

	// Verify they are NOT connected across clusters
	if loadedDSU.Connected(idx1Loaded, idx4Loaded) {
		t.Error("Expected different clusters to not be connected")
	}
}

func TestFileDSUPersistence_NewFileDSUPersistence(t *testing.T) {
	filepath := "/test/path/file.bin"
	persistence := classifier.NewFileDSUPersistence(filepath)

	if persistence == nil {
		t.Fatal("Expected non-nil persistence")
	}

	// The constructor is simple, just verify it creates an instance
	// Actual functionality is tested in other tests
}
