package graph

import (
	"fmt"
	"sort"
	"strings"
)

type Node struct {
	Name   string
	States []string
}

type Edge struct {
	From string
	To   string
}

type DAG struct {
	nodes  map[string]Node
	order  []string
	edges  []Edge
	parent map[string][]string
	child  map[string][]string
}

func NewDAG(nodes []Node, edges []Edge) (*DAG, error) {
	g := &DAG{
		nodes:  make(map[string]Node, len(nodes)),
		order:  make([]string, 0, len(nodes)),
		edges:  make([]Edge, 0, len(edges)),
		parent: make(map[string][]string, len(nodes)),
		child:  make(map[string][]string, len(nodes)),
	}
	seen := make(map[string]struct{}, len(nodes))
	for _, n := range nodes {
		if n.Name == "" {
			return nil, fmt.Errorf("empty node name")
		}
		if len(n.States) < 2 {
			return nil, fmt.Errorf("node %q needs at least two states", n.Name)
		}
		if _, ok := seen[n.Name]; ok {
			return nil, fmt.Errorf("duplicate node %q", n.Name)
		}
		stateSeen := make(map[string]struct{}, len(n.States))
		for _, s := range n.States {
			if s == "" {
				return nil, fmt.Errorf("node %q has empty state", n.Name)
			}
			if _, ok := stateSeen[s]; ok {
				return nil, fmt.Errorf("node %q has duplicate state %q", n.Name, s)
			}
			stateSeen[s] = struct{}{}
		}
		seen[n.Name] = struct{}{}
		g.nodes[n.Name] = n
		g.order = append(g.order, n.Name)
		g.parent[n.Name] = nil
		g.child[n.Name] = nil
	}
	edgeSeen := make(map[string]struct{}, len(edges))
	for _, e := range edges {
		if _, ok := g.nodes[e.From]; !ok {
			return nil, fmt.Errorf("unknown parent %q", e.From)
		}
		if _, ok := g.nodes[e.To]; !ok {
			return nil, fmt.Errorf("unknown child %q", e.To)
		}
		if e.From == e.To {
			return nil, fmt.Errorf("self-loop on %q", e.From)
		}
		key := e.From + "->" + e.To
		if _, ok := edgeSeen[key]; ok {
			return nil, fmt.Errorf("duplicate edge %s", key)
		}
		edgeSeen[key] = struct{}{}
		g.edges = append(g.edges, e)
		g.parent[e.To] = append(g.parent[e.To], e.From)
		g.child[e.From] = append(g.child[e.From], e.To)
	}
	if err := g.assertAcyclic(); err != nil {
		return nil, err
	}
	return g, nil
}

func (g *DAG) assertAcyclic() error {
	indeg := make(map[string]int, len(g.nodes))
	for name := range g.nodes {
		indeg[name] = len(g.parent[name])
	}
	q := make([]string, 0, len(g.nodes))
	for _, name := range g.order {
		if indeg[name] == 0 {
			q = append(q, name)
		}
	}
	seen := 0
	for len(q) > 0 {
		u := q[0]
		q = q[1:]
		seen++
		for _, v := range g.child[u] {
			indeg[v]--
			if indeg[v] == 0 {
				q = append(q, v)
			}
		}
	}
	if seen != len(g.nodes) {
		return fmt.Errorf("graph contains a cycle")
	}
	return nil
}

func (g *DAG) Node(name string) (Node, bool) {
	n, ok := g.nodes[name]
	return n, ok
}

func (g *DAG) Nodes() []Node {
	out := make([]Node, 0, len(g.order))
	for _, name := range g.order {
		out = append(out, g.nodes[name])
	}
	return out
}

func (g *DAG) Edges() []Edge {
	out := make([]Edge, len(g.edges))
	copy(out, g.edges)
	return out
}

func (g *DAG) Parents(name string) []string {
	p := g.parent[name]
	out := make([]string, len(p))
	copy(out, p)
	return out
}

func (g *DAG) Children(name string) []string {
	c := g.child[name]
	out := make([]string, len(c))
	copy(out, c)
	return out
}

func (g *DAG) TopologicalOrder() []string {
	indeg := make(map[string]int, len(g.nodes))
	for name := range g.nodes {
		indeg[name] = len(g.parent[name])
	}
	q := make([]string, 0, len(g.nodes))
	for _, name := range g.order {
		if indeg[name] == 0 {
			q = append(q, name)
		}
	}
	out := make([]string, 0, len(g.nodes))
	for len(q) > 0 {
		u := q[0]
		q = q[1:]
		out = append(out, u)
		next := append([]string{}, g.child[u]...)
		sort.Slice(next, func(i, j int) bool {
			return indexOf(g.order, next[i]) < indexOf(g.order, next[j])
		})
		for _, v := range next {
			indeg[v]--
			if indeg[v] == 0 {
				q = append(q, v)
			}
		}
	}
	return out
}

func indexOf(xs []string, v string) int {
	for i, x := range xs {
		if x == v {
			return i
		}
	}
	return len(xs)
}

func (g *DAG) String() string {
	parts := make([]string, 0, len(g.edges))
	for _, e := range g.edges {
		parts = append(parts, e.From+" -> "+e.To)
	}
	if len(parts) == 0 {
		return "(no edges)"
	}
	return strings.Join(parts, "; ")
}