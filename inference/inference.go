package inference

import (
	"fmt"
	"math"
	"sort"

	"bnproto/cpt"
	"bnproto/graph"
)

type Evidence map[string]string

type Distribution struct {
	Variable string
	States   []string
	Probs    []float64
}

func (d Distribution) Prob(state string) float64 {
	for i, s := range d.States {
		if s == state {
			return d.Probs[i]
		}
	}
	return 0
}

type Network struct {
	Graph  *graph.DAG
	Tables map[string]*cpt.Table
}

func (n Network) validate() error {
	for _, node := range n.Graph.Nodes() {
		t, ok := n.Tables[node.Name]
		if !ok {
			return fmt.Errorf("missing CPT for %s", node.Name)
		}
		if t.Variable != node.Name {
			return fmt.Errorf("CPT %s stored under %s", t.Variable, node.Name)
		}
	}
	return nil
}

func (n Network) checkEvidence(ev Evidence) error {
	for k, v := range ev {
		node, ok := n.Graph.Node(k)
		if !ok {
			return fmt.Errorf("evidence for unknown variable %q", k)
		}
		found := false
		for _, s := range node.States {
			if s == v {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("unknown state %q of %s", v, k)
		}
	}
	return nil
}

func Enumerate(n Network, query string, ev Evidence) (Distribution, error) {
	if err := n.validate(); err != nil {
		return Distribution{}, err
	}
	if err := n.checkEvidence(ev); err != nil {
		return Distribution{}, err
	}
	qn, ok := n.Graph.Node(query)
	if !ok {
		return Distribution{}, fmt.Errorf("unknown query %q", query)
	}
	if _, observed := ev[query]; observed {
		probs := make([]float64, len(qn.States))
		for i, s := range qn.States {
			if s == ev[query] {
				probs[i] = 1
			}
		}
		return Distribution{Variable: query, States: qn.States, Probs: probs}, nil
	}

	hidden := make([]graph.Node, 0)
	for _, node := range n.Graph.Nodes() {
		if node.Name == query {
			continue
		}
		if _, ok := ev[node.Name]; ok {
			continue
		}
		hidden = append(hidden, node)
	}

	unnorm := make([]float64, len(qn.States))
	assign := make(map[string]string, len(n.Graph.Nodes()))
	for k, v := range ev {
		assign[k] = v
	}
	var rec func(int)
	rec = func(hi int) {
		if hi == len(hidden) {
			for i, st := range qn.States {
				assign[query] = st
				unnorm[i] += joint(n, assign)
			}
			return
		}
		h := hidden[hi]
		for _, st := range h.States {
			assign[h.Name] = st
			rec(hi + 1)
		}
	}
	rec(0)

	return normalize(query, qn.States, unnorm)
}

func joint(n Network, assign map[string]string) float64 {
	p := 1.0
	for _, node := range n.Graph.Nodes() {
		t := n.Tables[node.Name]
		pi, err := t.Prob(assign[node.Name], assign)
		if err != nil {
			return 0
		}
		p *= pi
	}
	return p
}

func VariableElimination(n Network, query string, ev Evidence) (Distribution, error) {
	if err := n.validate(); err != nil {
		return Distribution{}, err
	}
	if err := n.checkEvidence(ev); err != nil {
		return Distribution{}, err
	}
	qn, ok := n.Graph.Node(query)
	if !ok {
		return Distribution{}, fmt.Errorf("unknown query %q", query)
	}

	factors := make([]*factor, 0, len(n.Graph.Nodes()))
	for _, node := range n.Graph.Nodes() {
		f, err := tableToFactor(n.Graph, n.Tables[node.Name], ev)
		if err != nil {
			return nilDist(), err
		}
		if f != nil {
			factors = append(factors, f)
		}
	}

	elim := make([]string, 0)
	for _, name := range n.Graph.TopologicalOrder() {
		if name == query {
			continue
		}
		if _, ok := ev[name]; ok {
			continue
		}
		elim = append(elim, name)
	}
	sort.SliceStable(elim, func(i, j int) bool {
		ni, _ := n.Graph.Node(elim[i])
		nj, _ := n.Graph.Node(elim[j])
		return len(ni.States) < len(nj.States)
	})

	for _, v := range elim {
		related := make([]*factor, 0)
		rest := make([]*factor, 0, len(factors))
		for _, f := range factors {
			if contains(f.vars, v) {
				related = append(related, f)
			} else {
				rest = append(rest, f)
			}
		}
		if len(related) == 0 {
			factors = rest
			continue
		}
		prod := related[0]
		for i := 1; i < len(related); i++ {
			prod = multiply(prod, related[i])
		}
		factors = append(rest, sumOut(prod, v, n.Graph))
	}

	var result *factor
	if len(factors) == 0 {
		node, _ := n.Graph.Node(query)
		result = uniformFactor(query, node.States)
	} else {
		result = factors[0]
		for i := 1; i < len(factors); i++ {
			result = multiply(result, factors[i])
		}
	}
	if len(result.vars) == 0 {
		probs := make([]float64, len(qn.States))
		if st, ok := ev[query]; ok {
			for i, s := range qn.States {
				if s == st {
					probs[i] = 1
				}
			}
		} else {
			u := 1.0 / float64(len(qn.States))
			for i := range probs {
				probs[i] = u
			}
		}
		return Distribution{Variable: query, States: qn.States, Probs: probs}, nil
	}
	if len(result.vars) != 1 || result.vars[0] != query {
		return Distribution{}, fmt.Errorf("elimination left variables %v, want [%s]", result.vars, query)
	}
	return normalize(query, result.states[0], append([]float64{}, result.values...))
}

func nilDist() Distribution { return Distribution{} }

func normalize(variable string, states []string, unnorm []float64) (Distribution, error) {
	sum := 0.0
	for _, v := range unnorm {
		if v < -1e-15 {
			return Distribution{}, fmt.Errorf("negative unnormalized mass")
		}
		sum += v
	}
	if sum <= 0 {
		return Distribution{}, fmt.Errorf("evidence has probability 0")
	}
	probs := make([]float64, len(unnorm))
	for i, v := range unnorm {
		probs[i] = v / sum
	}
	return Distribution{Variable: variable, States: states, Probs: probs}, nil
}

type factor struct {
	vars   []string
	states [][]string
	card   []int
	stride []int
	values []float64
}

func tableToFactor(g *graph.DAG, t *cpt.Table, ev Evidence) (*factor, error) {
	vars := append([]string{}, t.Parents...)
	vars = append(vars, t.Variable)
	f, err := newFactorFromVars(g, vars)
	if err != nil {
		return nil, err
	}
	assign := make(map[string]string, len(vars))
	for i := range f.values {
		decode(f, i, assign)
		p, err := t.Prob(assign[t.Variable], assign)
		if err != nil {
			return nil, err
		}
		f.values[i] = p
	}
	return reduceEvidence(f, ev)
}

func newFactorFromVars(g *graph.DAG, vars []string) (*factor, error) {
	f := &factor{
		vars:   append([]string{}, vars...),
		states: make([][]string, len(vars)),
		card:   make([]int, len(vars)),
	}
	n := 1
	for i, v := range vars {
		node, ok := g.Node(v)
		if !ok {
			return nil, fmt.Errorf("unknown variable %q", v)
		}
		f.states[i] = append([]string{}, node.States...)
		f.card[i] = len(node.States)
		n *= f.card[i]
	}
	f.stride = strides(f.card)
	f.values = make([]float64, n)
	return f, nil
}

func strides(card []int) []int {
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

func decode(f *factor, idx int, assign map[string]string) {
	rem := idx
	for i, v := range f.vars {
		si := 0
		if f.stride[i] != 0 {
			si = rem / f.stride[i]
			rem %= f.stride[i]
		}
		assign[v] = f.states[i][si]
	}
}

func indexOfState(states []string, s string) int {
	for i, x := range states {
		if x == s {
			return i
		}
	}
	return -1
}

func contains(xs []string, v string) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}

func reduceEvidence(f *factor, ev Evidence) (*factor, error) {
	keep := make([]int, 0, len(f.vars))
	fixed := make(map[int]int)
	for i, v := range f.vars {
		if st, ok := ev[v]; ok {
			si := indexOfState(f.states[i], st)
			if si < 0 {
				return nil, fmt.Errorf("unknown evidence state %q of %s", st, v)
			}
			fixed[i] = si
			continue
		}
		keep = append(keep, i)
	}
	if len(keep) == len(f.vars) {
		return f, nil
	}
	if len(keep) == 0 {
		idx := 0
		for i, si := range fixed {
			idx += si * f.stride[i]
		}
		return &factor{values: []float64{f.values[idx]}}, nil
	}
	out := &factor{
		vars:   make([]string, len(keep)),
		states: make([][]string, len(keep)),
		card:   make([]int, len(keep)),
	}
	for j, i := range keep {
		out.vars[j] = f.vars[i]
		out.states[j] = append([]string{}, f.states[i]...)
		out.card[j] = f.card[i]
	}
	out.stride = strides(out.card)
	n := 1
	for _, c := range out.card {
		n *= c
	}
	out.values = make([]float64, n)
	assign := make([]int, len(f.vars))
	for i, si := range fixed {
		assign[i] = si
	}
	var rec func(int)
	rec = func(j int) {
		if j == len(keep) {
			src := 0
			for i := range f.vars {
				src += assign[i] * f.stride[i]
			}
			dst := 0
			for k, i := range keep {
				dst += assign[i] * out.stride[k]
			}
			out.values[dst] = f.values[src]
			return
		}
		i := keep[j]
		for s := 0; s < f.card[i]; s++ {
			assign[i] = s
			rec(j + 1)
		}
	}
	rec(0)
	return out, nil
}

func multiply(a, b *factor) *factor {
	varSet := make(map[string][]string)
	order := make([]string, 0)
	add := func(f *factor) {
		for i, v := range f.vars {
			if _, ok := varSet[v]; !ok {
				varSet[v] = f.states[i]
				order = append(order, v)
			}
		}
	}
	add(a)
	add(b)
	out := &factor{
		vars:   order,
		states: make([][]string, len(order)),
		card:   make([]int, len(order)),
	}
	n := 1
	for i, v := range order {
		out.states[i] = varSet[v]
		out.card[i] = len(varSet[v])
		n *= out.card[i]
	}
	out.stride = strides(out.card)
	out.values = make([]float64, n)
	aIdx := make([]int, len(a.vars))
	bIdx := make([]int, len(b.vars))
	for i, v := range a.vars {
		aIdx[i] = indexOf(order, v)
	}
	for i, v := range b.vars {
		bIdx[i] = indexOf(order, v)
	}
	coord := make([]int, len(order))
	var rec func(int)
	rec = func(k int) {
		if k == len(order) {
			oi := 0
			for i := range order {
				oi += coord[i] * out.stride[i]
			}
			ai := 0
			for i := range a.vars {
				if len(a.stride) == 0 {
					break
				}
				ai += coord[aIdx[i]] * a.stride[i]
			}
			bi := 0
			for i := range b.vars {
				if len(b.stride) == 0 {
					break
				}
				bi += coord[bIdx[i]] * b.stride[i]
			}
			av, bv := 1.0, 1.0
			if len(a.values) > 0 {
				if len(a.vars) == 0 {
					av = a.values[0]
				} else {
					av = a.values[ai]
				}
			}
			if len(b.values) > 0 {
				if len(b.vars) == 0 {
					bv = b.values[0]
				} else {
					bv = b.values[bi]
				}
			}
			out.values[oi] = av * bv
			return
		}
		for s := 0; s < out.card[k]; s++ {
			coord[k] = s
			rec(k + 1)
		}
	}
	rec(0)
	return out
}

func sumOut(f *factor, v string, g *graph.DAG) *factor {
	pos := indexOf(f.vars, v)
	if pos < 0 {
		return f
	}
	keep := make([]int, 0, len(f.vars)-1)
	for i := range f.vars {
		if i != pos {
			keep = append(keep, i)
		}
	}
	if len(keep) == 0 {
		sum := 0.0
		for _, x := range f.values {
			sum += x
		}
		return &factor{values: []float64{sum}}
	}
	out := &factor{
		vars:   make([]string, len(keep)),
		states: make([][]string, len(keep)),
		card:   make([]int, len(keep)),
	}
	for j, i := range keep {
		out.vars[j] = f.vars[i]
		out.states[j] = f.states[i]
		out.card[j] = f.card[i]
	}
	out.stride = strides(out.card)
	n := 1
	for _, c := range out.card {
		n *= c
	}
	out.values = make([]float64, n)
	coord := make([]int, len(f.vars))
	var rec func(int)
	rec = func(k int) {
		if k == len(f.vars) {
			src := 0
			for i := range f.vars {
				src += coord[i] * f.stride[i]
			}
			dst := 0
			for j, i := range keep {
				dst += coord[i] * out.stride[j]
			}
			out.values[dst] += f.values[src]
			return
		}
		for s := 0; s < f.card[k]; s++ {
			coord[k] = s
			rec(k + 1)
		}
	}
	rec(0)
	_ = g
	return out
}

func uniformFactor(name string, states []string) *factor {
	u := 1.0 / float64(len(states))
	vals := make([]float64, len(states))
	for i := range vals {
		vals[i] = u
	}
	return &factor{
		vars:   []string{name},
		states: [][]string{states},
		card:   []int{len(states)},
		stride: []int{1},
		values: vals,
	}
}

func indexOf(xs []string, v string) int {
	for i, x := range xs {
		if x == v {
			return i
		}
	}
	return -1
}

func MaxAbsDiff(a, b Distribution) float64 {
	m := 0.0
	for _, st := range a.States {
		d := math.Abs(a.Prob(st) - b.Prob(st))
		if d > m {
			m = d
		}
	}
	return m
}