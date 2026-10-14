package patients

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestReadCommaDecimalsAndNaN(t *testing.T) {
	raw := `patient_id;group;age_days;weight_kg;nyha;pge1;vis;cfdna_t0;mtdna_frac_t0;delta_cfdna;ivl_hours;aki;death;vap;sepsis
1;1;30;4;I_II;0;0;NaN;NaN;NaN;7,6;0;0;0;0
2;1;30;4,32;I_II;0;0;28,92;6,172;44,88;11,5;0;0;0;0
23;0;5;2,8;III_IV;1;0;112,2;0,477;58,2;71;0;0;0;0
`
	data, st, err := Read(strings.NewReader(raw), DefaultThresholds())
	if err != nil {
		t.Fatal(err)
	}
	if st.Rows != 3 || st.Complete != 2 || st.Skipped != 1 {
		t.Fatalf("stats %+v", st)
	}
	if data[0]["NYHA"] != "I_II" || data[0]["cfDNA"] != "low" || data[0]["VIS"] != "low" || data[0]["IVL"] != "le24h" {
		t.Fatalf("row0 %+v", data[0])
	}
	if data[1]["NYHA"] != "III_IV" || data[1]["cfDNA"] != "high" || data[1]["IVL"] != "gt24h" {
		t.Fatalf("row1 %+v", data[1])
	}
}

func TestLoadRealCSV(t *testing.T) {
	path := filepath.Join("..", "data", "patients.csv")
	data, st, err := LoadCSV(path, DefaultThresholds())
	if err != nil {
		t.Fatal(err)
	}
	if st.Rows != 52 {
		t.Fatalf("rows=%d want 52", st.Rows)
	}
	if len(data) < 20 {
		t.Fatalf("too few complete cases: %d (skipped %+v)", len(data), st.SkipReasons)
	}
	for i, s := range data {
		for _, k := range []string{"NYHA", "cfDNA", "VIS", "IVL"} {
			if s[k] == "" {
				t.Fatalf("row %d missing %s", i, k)
			}
		}
	}
}