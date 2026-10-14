package patients

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"bnproto/learning"
)

type Thresholds struct {
	CfDNAHigh float64
	VISHigh float64
	IVLHours float64
}

func DefaultThresholds() Thresholds {
	return Thresholds{CfDNAHigh: 50, VISHigh: 5, IVLHours: 24}
}

type Stats struct {
	Rows        int
	Complete    int
	Skipped     int
	SkipReasons map[string]int
}

func LoadCSV(path string, th Thresholds) ([]learning.Sample, Stats, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Stats{}, err
	}
	defer f.Close()
	return Read(f, th)
}

func Read(r io.Reader, th Thresholds) ([]learning.Sample, Stats, error) {
	cr := csv.NewReader(r)
	cr.Comma = ';'
	cr.TrimLeadingSpace = true
	cr.ReuseRecord = false
	cr.FieldsPerRecord = -1

	header, err := cr.Read()
	if err != nil {
		return nil, Stats{}, fmt.Errorf("read header: %w", err)
	}
	idx := colIndex(header)
	for _, need := range []string{"nyha", "vis", "cfdna_t0", "ivl_hours"} {
		if _, ok := idx[need]; !ok {
			return nil, Stats{}, fmt.Errorf("csv missing column %q", need)
		}
	}

	st := Stats{SkipReasons: map[string]int{}}
	out := make([]learning.Sample, 0, 64)
	line := 1
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, st, fmt.Errorf("row after header line %d: %w", line, err)
		}
		line++
		st.Rows++
		sample, reason := discretizeRow(rec, idx, th)
		if reason != "" {
			st.Skipped++
			st.SkipReasons[reason]++
			continue
		}
		st.Complete++
		out = append(out, sample)
	}
	if st.Complete == 0 {
		return nil, st, fmt.Errorf("no complete cases (need nyha, vis, cfdna_t0, ivl_hours)")
	}
	return out, st, nil
}

func colIndex(header []string) map[string]int {
	m := make(map[string]int, len(header))
	for i, h := range header {
		m[strings.ToLower(strings.TrimSpace(h))] = i
	}
	return m
}

func field(rec []string, idx map[string]int, name string) string {
	i, ok := idx[name]
	if !ok || i >= len(rec) {
		return ""
	}
	return strings.TrimSpace(rec[i])
}

func discretizeRow(rec []string, idx map[string]int, th Thresholds) (learning.Sample, string) {
	nyha := field(rec, idx, "nyha")
	if nyha != "I_II" && nyha != "III_IV" {
		return nil, "bad_nyha"
	}
	vis, ok := parseFloat(field(rec, idx, "vis"))
	if !ok {
		return nil, "missing_vis"
	}
	cfdna, ok := parseFloat(field(rec, idx, "cfdna_t0"))
	if !ok {
		return nil, "missing_cfdna"
	}
	ivl, ok := parseFloat(field(rec, idx, "ivl_hours"))
	if !ok {
		return nil, "missing_ivl"
	}

	visState := "low"
	if vis >= th.VISHigh {
		visState = "high"
	}
	cfState := "low"
	if cfdna >= th.CfDNAHigh {
		cfState = "high"
	}
	ivlState := "le24h"
	if ivl > th.IVLHours {
		ivlState = "gt24h"
	}
	return learning.Sample{
		"NYHA":  nyha,
		"cfDNA": cfState,
		"VIS":   visState,
		"IVL":   ivlState,
	}, ""
}

func parseFloat(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	if s == "" || strings.EqualFold(s, "NaN") || s == "-" {
		return 0, false
	}
	s = strings.ReplaceAll(s, ",", ".")
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}