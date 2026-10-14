package synthetic

import (
	"fmt"
	"math/rand"

	"bnproto/cpt"
	"bnproto/graph"
	"bnproto/learning"
)

func Generate(g *graph.DAG, tables map[string]*cpt.Table, n int, rng *rand.Rand) ([]learning.Sample, error) {
	if n < 0 {
		return nil, fmt.Errorf("n must be >= 0")
	}
	order := g.TopologicalOrder()
	out := make([]learning.Sample, n)
	for i := 0; i < n; i++ {
		row := make(learning.Sample, len(order))
		for _, name := range order {
			t, ok := tables[name]
			if !ok {
				return nil, fmt.Errorf("missing CPT for %s", name)
			}
			idx, err := t.ConfigIndex(row)
			if err != nil {
				return nil, err
			}
			row[name] = sampleState(rng, t.States, t.Values[idx])
		}
		out[i] = row
	}
	return out, nil
}

func sampleState(rng *rand.Rand, states []string, probs []float64) string {
	u := rng.Float64()
	acc := 0.0
	for i, p := range probs {
		acc += p
		if u <= acc {
			return states[i]
		}
	}
	return states[len(states)-1]
}