package cpt

import (
	"fmt"
	"math"
	"strings"

	"bnproto/graph"
)

const eps = 1e-12

type Table struct {
	Variable string
	States   []string
	Parents  []string
	PStates  [][]string
	Values   [][]float64
	strides  []int
}

func NewTable(g *graph.DAG, variable string) (*Table, error) {
	node, ok := g.Node(variable)
	if !ok {
		return nil, fmt.Errorf("unknown variable %q", variable)
	}
	parents := g.Parents(variable)
	pStates := make([][]string, len(parents))
	card := make([]int, len(parents))
	for i, p := range parents {
		pn, ok := g.Node(p)
		if !ok {
			return nil, fmt.Errorf("unknown parent %q", p)
		}
		pStates[i] = append([]string{}, pn.States...)
		card[i] = len(pn.States)
	}
	nCfg := 1
	for _, c := range card {
		nCfg *= c
	}
	values := make([][]float64, nCfg)
	for i := range values {
		values[i] = make([]float64, len(node.States))
	}
	return &Table{
		Variable: variable,
		States:   append([]string{}, node.States...),
		Parents:  append([]string{}, parents...),
		PStates:  pStates,
		Values:   values,
		strides:  parentStrides(card),
	}, nil
}

func parentStrides(card []int) []int {
	s := make([]int, len(card))
	if len(card) == 0 {
		return s
	}
	s[len(card)-1] = 1
	for i := len(card) - 2; i >= 0; i-- {
		s[i] = s[i+1] * card[i+1]
	}
	return s
}

func (t *Table) ConfigIndex(parentStates map[string]string) (int, error) {
	idx := 0
	for i, p := range t.Parents {
		st, ok := parentStates[p]
		if !ok {
			return 0, fmt.Errorf("missing parent %q for %s", p, t.Variable)
		}
		si := indexOf(t.PStates[i], st)
		if si < 0 {
			return 0, fmt.Errorf("unknown state %q of parent %s", st, p)
		}
		idx += si * t.strides[i]
	}
	return idx, nil
}

func (t *Table) SetRow(parentStates map[string]string, probs []float64) error {
	if len(probs) != len(t.States) {
		return fmt.Errorf("%s: expected %d probabilities, got %d", t.Variable, len(t.States), len(probs))
	}
	sum := 0.0
	for _, p := range probs {
		if p < -eps {
			return fmt.Errorf("%s: negative probability %g", t.Variable, p)
		}
		sum += p
	}
	if math.Abs(sum-1) > 1e-8 {
		return fmt.Errorf("%s: probabilities sum to %g, want 1", t.Variable, sum)
	}
	idx, err := t.ConfigIndex(parentStates)
	if err != nil {
		return err
	}
	copy(t.Values[idx], probs)
	return nil
}

func (t *Table) Prob(childState string, parentStates map[string]string) (float64, error) {
	si := indexOf(t.States, childState)
	if si < 0 {
		return 0, fmt.Errorf("unknown state %q of %s", childState, t.Variable)
	}
	idx, err := t.ConfigIndex(parentStates)
	if err != nil {
		return 0, err
	}
	return t.Values[idx][si], nil
}

func (t *Table) NormalizeRow(idx int) error {
	if idx < 0 || idx >= len(t.Values) {
		return fmt.Errorf("row %d out of range", idx)
	}
	sum := 0.0
	for _, v := range t.Values[idx] {
		if v < 0 {
			return fmt.Errorf("%s: negative count/prob", t.Variable)
		}
		sum += v
	}
	if sum == 0 {
		u := 1.0 / float64(len(t.States))
		for i := range t.Values[idx] {
			t.Values[idx][i] = u
		}
		return nil
	}
	for i := range t.Values[idx] {
		t.Values[idx][i] /= sum
	}
	return nil
}

func (t *Table) NormalizeAll() error {
	for i := range t.Values {
		if err := t.NormalizeRow(i); err != nil {
			return err
		}
	}
	return nil
}

func (t *Table) ParentAssignment(idx int) map[string]string {
	out := make(map[string]string, len(t.Parents))
	rem := idx
	for i, p := range t.Parents {
		card := len(t.PStates[i])
		var si int
		if t.strides[i] == 0 {
			si = 0
		} else {
			si = rem / t.strides[i]
			rem %= t.strides[i]
			_ = card
		}
		out[p] = t.PStates[i][si]
	}
	return out
}

func (t *Table) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "CPT %s | [%s]\n", t.Variable, strings.Join(t.Parents, ", "))
	for i, row := range t.Values {
		cfg := t.ParentAssignment(i)
		parts := make([]string, 0, len(t.Parents))
		for _, p := range t.Parents {
			parts = append(parts, p+"="+cfg[p])
		}
		probs := make([]string, len(t.States))
		for j, st := range t.States {
			probs[j] = fmt.Sprintf("P(%s)=%.4f", st, row[j])
		}
		if len(parts) == 0 {
			fmt.Fprintf(&b, "  %s\n", strings.Join(probs, " "))
		} else {
			fmt.Fprintf(&b, "  [%s] %s\n", strings.Join(parts, ", "), strings.Join(probs, " "))
		}
	}
	return b.String()
}

func indexOf(xs []string, v string) int {
	for i, x := range xs {
		if x == v {
			return i
		}
	}
	return -1
}