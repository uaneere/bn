package inference_test

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"bnproto/domain"
	"bnproto/inference"
	"bnproto/learning"
	"bnproto/patients"
	"bnproto/synthetic"
)

func TestEnumerateMatchesVE(t *testing.T) {
	g, err := domain.NewGraph()
	if err != nil {
		t.Fatal(err)
	}
	net, err := domain.TrueNetwork(g)
	if err != nil {
		t.Fatal(err)
	}
	ev := inference.Evidence{"VIS": "high", "cfDNA": "high"}
	a, err := inference.Enumerate(net, "IVL", ev)
	if err != nil {
		t.Fatal(err)
	}
	b, err := inference.VariableElimination(net, "IVL", ev)
	if err != nil {
		t.Fatal(err)
	}
	if inference.MaxAbsDiff(a, b) > 1e-12 {
		t.Fatalf("enum=%v ve=%v", a, b)
	}
	if math.Abs(a.Prob("gt24h")+a.Prob("le24h")-1) > 1e-12 {
		t.Fatalf("not normalized: %v", a)
	}
}

func TestLearningRecoversCPT(t *testing.T) {
	g, err := domain.NewGraph()
	if err != nil {
		t.Fatal(err)
	}
	trueNet, err := domain.TrueNetwork(g)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	data, err := synthetic.Generate(g, trueNet.Tables, 8000, rng)
	if err != nil {
		t.Fatal(err)
	}
	learned, err := learning.FitCPTs(g, data, learning.Options{Alpha: 1})
	if err != nil {
		t.Fatal(err)
	}
	ivl := learned["IVL"]
	p, err := ivl.Prob("gt24h", map[string]string{"cfDNA": "high", "VIS": "high"})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(p-0.85) > 0.05 {
		t.Fatalf("P(IVL=gt24h | high,high)=%g, want ~0.85", p)
	}
}

func TestFitOnPatientsCSV(t *testing.T) {
	g, err := domain.NewGraph()
	if err != nil {
		t.Fatal(err)
	}
	data, st, err := patients.LoadCSV(filepath.Join("..", "data", "patients.csv"), patients.DefaultThresholds())
	if err != nil {
		t.Fatal(err)
	}
	if st.Complete != len(data) {
		t.Fatalf("complete=%d len(data)=%d", st.Complete, len(data))
	}
	learned, err := learning.FitCPTs(g, data, learning.Options{Alpha: 1})
	if err != nil {
		t.Fatal(err)
	}
	net := inference.Network{Graph: g, Tables: learned}
	post, err := inference.Enumerate(net, "IVL", inference.Evidence{"VIS": "high", "cfDNA": "high"})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(post.Prob("gt24h")+post.Prob("le24h")-1) > 1e-9 {
		t.Fatalf("not normalized: %+v", post)
	}
}

func TestSprinklerStyleVStructure(t *testing.T) {
	g, err := domain.NewGraph()
	if err != nil {
		t.Fatal(err)
	}
	net, err := domain.TrueNetwork(g)
	if err != nil {
		t.Fatal(err)
	}
	prior, err := inference.Enumerate(net, "IVL", nil)
	if err != nil {
		t.Fatal(err)
	}
	post, err := inference.Enumerate(net, "IVL", inference.Evidence{"VIS": "high"})
	if err != nil {
		t.Fatal(err)
	}
	if post.Prob("gt24h") <= prior.Prob("gt24h") {
		t.Fatalf("high VIS should raise P(IVL>24); prior=%.3f post=%.3f", prior.Prob("gt24h"), post.Prob("gt24h"))
	}
}