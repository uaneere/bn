package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"math/rand"

	"bnproto/domain"
	"bnproto/inference"
	"bnproto/learning"
	"bnproto/patients"
	"bnproto/synthetic"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	n := flag.Int("n", 4000, "number of synthetic training samples (ignored if -csv is set)")
	alpha := flag.Float64("alpha", 1.0, "Dirichlet pseudo-count (0 = pure MLE, 1 = Laplace)")
	seed := flag.Int64("seed", 42, "RNG seed for ancestral sampling")
	query := flag.String("query", "IVL", "query variable")
	evidence := flag.String("evidence", "VIS=high,cfDNA=high", "comma-separated VAR=state pairs")
	csvPath := flag.String("csv", "", "patients CSV (semicolon, header first); empty = synthetic data")
	cfdnaCut := flag.Float64("cfdna-cut", 50, "cfdna_t0 (ng/ml) threshold for state high")
	visCut := flag.Float64("vis-cut", 5, "VIS threshold for state high")
	flag.Parse()

	g, err := domain.NewGraph()
	if err != nil {
		return err
	}

	var data []learning.Sample
	var source string
	var trueNet *inference.Network
	if *csvPath != "" {
		th := patients.DefaultThresholds()
		th.CfDNAHigh = *cfdnaCut
		th.VISHigh = *visCut
		loaded, st, err := patients.LoadCSV(*csvPath, th)
		if err != nil {
			return err
		}
		data = loaded
		source = fmt.Sprintf("csv %s  rows=%d complete=%d skipped=%d reasons=%v  cuts: cfDNA>=%.3g VIS>=%.3g IVL>%.0fh",
			*csvPath, st.Rows, st.Complete, st.Skipped, st.SkipReasons, th.CfDNAHigh, th.VISHigh, th.IVLHours)
	} else {
		tn, err := domain.TrueNetwork(g)
		if err != nil {
			return err
		}
		trueNet = &tn
		rng := rand.New(rand.NewSource(*seed))
		data, err = synthetic.Generate(g, tn.Tables, *n, rng)
		if err != nil {
			return err
		}
		source = fmt.Sprintf("synthetic ancestral sampling n=%d seed=%d", *n, *seed)
	}

	learned, err := learning.FitCPTs(g, data, learning.Options{Alpha: *alpha})
	if err != nil {
		return err
	}
	learnedNet := inference.Network{Graph: g, Tables: learned}

	ev, err := parseEvidence(*evidence)
	if err != nil {
		return err
	}

	t0 := time.Now()
	postEnum, err := inference.Enumerate(learnedNet, *query, ev)
	if err != nil {
		return fmt.Errorf("enumerate: %w", err)
	}
	postVE, err := inference.VariableElimination(learnedNet, *query, ev)
	if err != nil {
		return fmt.Errorf("variable elimination: %w", err)
	}
	var truePost inference.Distribution
	if trueNet != nil {
		truePost, err = inference.Enumerate(*trueNet, *query, ev)
		if err != nil {
			return fmt.Errorf("true query: %w", err)
		}
	}
	elapsed := time.Since(t0)

	fmt.Println("Bayesian network prototype (discrete, Dirichlet smoothing)")
	fmt.Println("=======================================================")
	fmt.Printf("Structure: %s\n", g)
	fmt.Printf("Samples:   %s\n", source)
	fmt.Printf("Learning:  Dirichlet alpha=%.3f\n", *alpha)
	fmt.Println()
	fmt.Println("Learned CPTs")
	fmt.Println("------------")
	for _, node := range g.Nodes() {
		fmt.Print(learned[node.Name])
	}
	if trueNet != nil {
		fmt.Println("Parameter recovery (max |P_hat - P_true| per CPT)")
		fmt.Println("-------------------------------------------------")
		for _, node := range g.Nodes() {
			d := maxTableDiff(learned[node.Name].Values, trueNet.Tables[node.Name].Values)
			fmt.Printf("  %-6s  %.4f\n", node.Name, d)
		}
		fmt.Println()
	}
	fmt.Printf("Query: P(%s | %s)\n", *query, formatEvidence(ev))
	fmt.Println("------")
	printDist("learned (enumeration)", postEnum)
	printDist("learned (var. elim.)", postVE)
	if trueNet != nil {
		printDist("true generative CPT", truePost)
		fmt.Printf("\n|enum - VE|_∞ = %.2e\n", inference.MaxAbsDiff(postEnum, postVE))
		fmt.Printf("|learned - true|_∞ = %.4f\n", inference.MaxAbsDiff(postEnum, truePost))
	} else {
		fmt.Printf("\n|enum - VE|_∞ = %.2e\n", inference.MaxAbsDiff(postEnum, postVE))
	}
	fmt.Printf("inference wall time: %s\n", elapsed)
	return nil
}

func parseEvidence(s string) (inference.Evidence, error) {
	ev := inference.Evidence{}
	s = strings.TrimSpace(s)
	if s == "" {
		return ev, nil
	}
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		k, v, ok := strings.Cut(part, "=")
		if !ok || k == "" || v == "" {
			return nil, fmt.Errorf("bad evidence fragment %q (want VAR=state)", part)
		}
		ev[k] = v
	}
	return ev, nil
}

func formatEvidence(ev inference.Evidence) string {
	if len(ev) == 0 {
		return "∅"
	}
	parts := make([]string, 0, len(ev))
	for k, v := range ev {
		parts = append(parts, k+"="+v)
	}
	return strings.Join(parts, ", ")
}

func printDist(label string, d inference.Distribution) {
	fmt.Printf("  %s:\n", label)
	for i, st := range d.States {
		fmt.Printf("    P(%s=%s) = %.4f\n", d.Variable, st, d.Probs[i])
	}
}

func maxTableDiff(a, b [][]float64) float64 {
	m := 0.0
	for i := range a {
		for j := range a[i] {
			d := math.Abs(a[i][j] - b[i][j])
			if d > m {
				m = d
			}
		}
	}
	return m
}