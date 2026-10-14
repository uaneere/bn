package learning

import (
	"fmt"

	"bnproto/cpt"
	"bnproto/graph"
)

type Sample map[string]string

type Options struct {
	Alpha float64
}

func FitCPTs(g *graph.DAG, data []Sample, opt Options) (map[string]*cpt.Table, error) {
	if opt.Alpha < 0 {
		return nil, fmt.Errorf("alpha must be >= 0")
	}
	tables := make(map[string]*cpt.Table, len(g.Nodes()))
	for _, n := range g.Nodes() {
		t, err := cpt.NewTable(g, n.Name)
		if err != nil {
			return nil, err
		}
		for i := range t.Values {
			for j := range t.Values[i] {
				t.Values[i][j] = opt.Alpha
			}
		}
		tables[n.Name] = t
	}
	for i, row := range data {
		for _, n := range g.Nodes() {
			child, ok := row[n.Name]
			if !ok {
				return nil, fmt.Errorf("row %d: missing %s", i, n.Name)
			}
			t := tables[n.Name]
			si := -1
			for k, st := range t.States {
				if st == child {
					si = k
					break
				}
			}
			if si < 0 {
				return nil, fmt.Errorf("row %d: unknown state %q of %s", i, child, n.Name)
			}
			idx, err := t.ConfigIndex(row)
			if err != nil {
				return nil, fmt.Errorf("row %d: %w", i, err)
			}
			t.Values[idx][si]++
		}
	}
	for name, t := range tables {
		if err := t.NormalizeAll(); err != nil {
			return nil, fmt.Errorf("normalize %s: %w", name, err)
		}
	}
	return tables, nil
}