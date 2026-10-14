package domain

import (
	"bnproto/cpt"
	"bnproto/graph"
	"bnproto/inference"
)

func NewGraph() (*graph.DAG, error) {
	nodes := []graph.Node{
		{Name: "NYHA", States: []string{"I_II", "III_IV"}},
		{Name: "cfDNA", States: []string{"low", "high"}},
		{Name: "VIS", States: []string{"low", "high"}},
		{Name: "IVL", States: []string{"le24h", "gt24h"}},
	}
	edges := []graph.Edge{
		{From: "NYHA", To: "cfDNA"},
		{From: "NYHA", To: "VIS"},
		{From: "cfDNA", To: "VIS"},
		{From: "cfDNA", To: "IVL"},
		{From: "VIS", To: "IVL"},
	}
	return graph.NewDAG(nodes, edges)
}

func TrueTables(g *graph.DAG) (map[string]*cpt.Table, error) {
	out := make(map[string]*cpt.Table, 4)

	nyha, err := cpt.NewTable(g, "NYHA")
	if err != nil {
		return nil, err
	}
	if err := nyha.SetRow(nil, []float64{0.65, 0.35}); err != nil {
		return nil, err
	}
	out["NYHA"] = nyha

	cfdna, err := cpt.NewTable(g, "cfDNA")
	if err != nil {
		return nil, err
	}
	if err := cfdna.SetRow(map[string]string{"NYHA": "I_II"}, []float64{0.80, 0.20}); err != nil {
		return nil, err
	}
	if err := cfdna.SetRow(map[string]string{"NYHA": "III_IV"}, []float64{0.30, 0.70}); err != nil {
		return nil, err
	}
	out["cfDNA"] = cfdna

	vis, err := cpt.NewTable(g, "VIS")
	if err != nil {
		return nil, err
	}
	if err := vis.SetRow(map[string]string{"NYHA": "I_II", "cfDNA": "low"}, []float64{0.85, 0.15}); err != nil {
		return nil, err
	}
	if err := vis.SetRow(map[string]string{"NYHA": "I_II", "cfDNA": "high"}, []float64{0.45, 0.55}); err != nil {
		return nil, err
	}
	if err := vis.SetRow(map[string]string{"NYHA": "III_IV", "cfDNA": "low"}, []float64{0.50, 0.50}); err != nil {
		return nil, err
	}
	if err := vis.SetRow(map[string]string{"NYHA": "III_IV", "cfDNA": "high"}, []float64{0.20, 0.80}); err != nil {
		return nil, err
	}
	out["VIS"] = vis

	ivl, err := cpt.NewTable(g, "IVL")
	if err != nil {
		return nil, err
	}
	if err := ivl.SetRow(map[string]string{"cfDNA": "low", "VIS": "low"}, []float64{0.90, 0.10}); err != nil {
		return nil, err
	}
	if err := ivl.SetRow(map[string]string{"cfDNA": "low", "VIS": "high"}, []float64{0.45, 0.55}); err != nil {
		return nil, err
	}
	if err := ivl.SetRow(map[string]string{"cfDNA": "high", "VIS": "low"}, []float64{0.55, 0.45}); err != nil {
		return nil, err
	}
	if err := ivl.SetRow(map[string]string{"cfDNA": "high", "VIS": "high"}, []float64{0.15, 0.85}); err != nil {
		return nil, err
	}
	out["IVL"] = ivl
	return out, nil
}

func TrueNetwork(g *graph.DAG) (inference.Network, error) {
	t, err := TrueTables(g)
	if err != nil {
		return inference.Network{}, err
	}
	return inference.Network{Graph: g, Tables: t}, nil
}