package graph

import "testing"

func TestRejectsCycle(t *testing.T) {
	_, err := NewDAG(
		[]Node{{Name: "A", States: []string{"0", "1"}}, {Name: "B", States: []string{"0", "1"}}},
		[]Edge{{From: "A", To: "B"}, {From: "B", To: "A"}},
	)
	if err == nil {
		t.Fatal("expected cycle error")
	}
}

func TestTopoOrder(t *testing.T) {
	g, err := NewDAG(
		[]Node{
			{Name: "A", States: []string{"0", "1"}},
			{Name: "B", States: []string{"0", "1"}},
			{Name: "C", States: []string{"0", "1"}},
		},
		[]Edge{{From: "A", To: "C"}, {From: "B", To: "C"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	order := g.TopologicalOrder()
	pos := map[string]int{}
	for i, n := range order {
		pos[n] = i
	}
	if pos["A"] > pos["C"] || pos["B"] > pos["C"] {
		t.Fatalf("bad order %v", order)
	}
}