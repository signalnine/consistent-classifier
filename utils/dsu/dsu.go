package dsu

import "sync"

type dsu struct {
	root   []int
	rank   []int
	labels map[string]int
	lock   sync.RWMutex
}

// NewDSU creates a new DSU with the given size.
func NewDSU(size int) *dsu {
	return &dsu{
		root:   make([]int, size),
		rank:   make([]int, size),
		labels: make(map[string]int),
		lock:   sync.RWMutex{},
	}
}

// Add adds a new group to the DSU. Returns the index of the new group.
func (d *dsu) Add(label string) int {
	d.lock.Lock()
	defer d.lock.Unlock()

	d.root = append(d.root, len(d.root))
	d.rank = append(d.rank, 0)
	d.labels[label] = len(d.root) - 1
	return d.labels[label]
}

// Find finds the root of the set
func (d *dsu) Find(x int) int {
	d.lock.RLock()
	defer d.lock.RUnlock()

	if d.root[x] == x {
		return x
	}
	d.root[x] = d.Find(d.root[x]) // Path compression
	return d.root[x]
}

// FindByLabel finds the root of the set by label
func (d *dsu) FindByLabel(label string) int {
	d.lock.RLock()
	defer d.lock.RUnlock()

	idx, ok := d.labels[label]
	if !ok {
		return -1
	}

	return d.Find(idx)
}

// Union merges two sets
func (d *dsu) Union(x int, y int) {
	d.lock.Lock()
	defer d.lock.Unlock()

	rootX := d.Find(x)
	rootY := d.Find(y)
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
	return d.Find(x) == d.Find(y)
}

// Size returns the number of elements in the DSU
func (d *dsu) Size() int {
	d.lock.RLock()
	defer d.lock.RUnlock()

	return len(d.root)
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
