package disjoint_set

import (
	"encoding/json"
)

// MarshalJSON implements json.Marshaler interface
func (d *dsu) MarshalJSON() ([]byte, error) {
	d.lock.RLock()
	defer d.lock.RUnlock()

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
