package dsu

import (
	"encoding/json"
	"os"
)

// ReadFromFile reads a DSU from a file.
func (d *dsu) ReadFromFile(filename string) (*dsu, error) {
	d.lock.Lock()
	defer d.lock.Unlock()

	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	err = d.UnmarshalJSON(data)
	if err != nil {
		return nil, err
	}

	return d, nil
}

// WriteToFile writes the DSU to a file.
func (d *dsu) WriteToFile(filename string) error {
	d.lock.Lock()
	defer d.lock.Unlock()

	data, err := d.MarshalJSON()
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

// MarshalJSON implements json.Marshaler interface
func (d *dsu) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]interface{}{
		"root":   d.root,
		"rank":   d.rank,
		"labels": d.labels,
	})
}

// UnmarshalJSON implements json.Unmarshaler interface
func (d *dsu) UnmarshalJSON(data []byte) error {
	var temp struct {
		Root   []int          `json:"root"`
		Rank   []int          `json:"rank"`
		Labels map[string]int `json:"labels"`
	}

	if err := json.Unmarshal(data, &temp); err != nil {
		return err
	}

	d.root = temp.Root
	d.rank = temp.Rank
	d.labels = temp.Labels

	return nil
}
