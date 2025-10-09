package disjoint_set

import (
	"sync"
)

// DSU represents a Disjoint Set Union data structure
type DSU = dsu

type dsu struct {
	root       []int
	rank       []int
	labels     map[string]int
	labelIndex map[int]string
	lock       sync.RWMutex
}

// NewDSU creates a new DSU with the given size.
func NewDSU() *dsu {
	return &dsu{
		root:       make([]int, 0),
		rank:       make([]int, 0),
		labels:     make(map[string]int),
		labelIndex: make(map[int]string),
		lock:       sync.RWMutex{},
	}
}

// Add adds a new group to the DSU. Returns the index of the new group.
func (d *dsu) Add(label string) int {
	d.lock.Lock()
	defer d.lock.Unlock()

	return d.add(label)
}

// add adds a new group to the DSU. Returns the index of the new group. (internal, unlocked, caller must hold lock)
func (d *dsu) add(label string) int {
	d.root = append(d.root, len(d.root))
	d.rank = append(d.rank, 0)
	d.labels[label] = len(d.root) - 1
	d.labelIndex[len(d.root)-1] = label
	return d.labels[label]
}

// find finds the root of the set (internal, unlocked - caller must hold lock)
func (d *dsu) find(x int) int {
	if d.root[x] == x {
		return x
	}

	d.root[x] = d.find(d.root[x]) // Path compression
	return d.root[x]
}

// FindOrCreate finds the root of the set by label, or adds it if it doesn't exist
func (d *dsu) FindOrCreate(label string) int {
	d.lock.Lock()
	defer d.lock.Unlock()

	idx, ok := d.labels[label]
	if !ok {
		return d.add(label)
	}

	return d.find(idx)
}

// Union merges two sets
func (d *dsu) Union(x int, y int) {
	d.lock.Lock()
	defer d.lock.Unlock()

	rootX := d.find(x)
	rootY := d.find(y)

	if rootX == rootY {
		return
	}

	if d.rank[rootX] > d.rank[rootY] {
		d.root[rootY] = rootX
	} else if d.rank[rootX] < d.rank[rootY] {
		d.root[rootX] = rootY
	} else {
		d.root[rootY] = rootX
		d.rank[rootX]++
	}
}

// Connected checks if two elements are in the same set
func (d *dsu) Connected(x int, y int) bool {
	d.lock.RLock()
	defer d.lock.RUnlock()

	return d.find(x) == d.find(y)
}

// Size returns the number of elements in the DSU
func (d *dsu) Size() int {
	d.lock.RLock()
	defer d.lock.RUnlock()

	return len(d.labels)
}

// Labels returns all labels in the DSU
func (d *dsu) Labels() []string {
	d.lock.RLock()
	defer d.lock.RUnlock()

	labels := make([]string, 0, len(d.labels))
	for label := range d.labels {
		labels = append(labels, label)
	}
	return labels
}

// CountSets returns the number of unique sets in the DSU
func (d *dsu) CountSets() int {
	d.lock.RLock()
	defer d.lock.RUnlock()

	rootSet := make(map[int]bool)
	for i := range d.root {
		root := d.find(i)
		rootSet[root] = true
	}

	return len(rootSet)
}

// FindLabel finds the label by a root index
func (d *dsu) FindLabel(idx int) string {
	d.lock.RLock()
	defer d.lock.RUnlock()

	if label, ok := d.labelIndex[idx]; ok {
		return label
	}

	return ""
}
