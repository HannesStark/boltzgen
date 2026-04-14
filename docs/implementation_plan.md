# Implementation Plan: Developability Filtering for BoltzGen

## Context

BoltzGen generates protein binder designs and filters them by structural quality (RMSD, confidence, interactions). However, designs also need to be assessed for practical developability before experimental validation. Two features are being added:

1. **Selectivity against decoy targets** — ensure binders are specific to the true target and don't cross-react with decoys. Approach: reuse Boltz2 folding confidence (iptm/pae) as an affinity proxy by folding designs against each decoy target.
2. **HIS-tag / expression tag effects** — assess whether expression tags (e.g. 6xHis) would perturb binding or stability. Two-tier approach: fast sequence-level screen, then explicit re-folding with tag appended for designs that pass initial filters.

---

## Feature 1: Selectivity Against Decoy Targets

### Step 1.1 — Extend YAML Design Spec Schema

**File:** `src/boltzgen/data/parse/schema.py`

- Add `"decoys"` to `yaml_keys` list (line ~289)
- Add parsing logic in `parse_boltzgen_schema()` (line ~1322) to handle:
  ```yaml
  decoys:
    - id: decoy_1
      protein:
        id: A
        sequence: MKVL...
    - id: decoy_2
      file: decoy2.pdb
  ```
- Each decoy is parsed as a separate `Target` structure
- Store decoys in a new field on the main `Target` dataclass: `decoys: Optional[list[Target]]`

### Step 1.2 — Add `decoy_folding` Pipeline Step

**File:** `src/boltzgen/cli/boltzgen.py`

- In `BinderDesignPipeline.__init__()` (line ~858), after the main `folding` step (line ~1125):
  - Check if design spec contains decoys
  - For each decoy, add a `PipelineStep` named `decoy_folding_{decoy_id}` using the same `fold.yaml` config but with the decoy structure as target
- The folding step config needs a modified data module that swaps the true target for the decoy target while keeping the designed binder chain

**New/modified files:**
- `src/boltzgen/task/predict/data_from_generated.py` — extend `FromGeneratedDataModule` to accept an alternate target structure (decoy) to fold against
- OR create a thin wrapper `DecoyFoldingDataModule` that loads the designed sequence but pairs it with a decoy target

**Output:** `refold_decoy_{decoy_id}_cif/` subdirectories containing refolded complex structures with each decoy

### Step 1.3 — Add Selectivity Metrics to Analysis

**File:** `src/boltzgen/task/analyze/analyze.py`

In `compute_metrics()` (line ~521):
- After computing main folding metrics, iterate over decoy folding results
- For each decoy, extract: `iptm`, `pae` (design-to-decoy interface), `ptm`
- Compute aggregate metrics:
  ```
  decoy_{id}_iptm          — per-decoy iptm
  max_decoy_iptm           — worst-case cross-reactivity
  mean_decoy_iptm          — average cross-reactivity
  selectivity_score        — design_to_target_iptm - max_decoy_iptm
  selectivity_ratio        — design_to_target_iptm / (max_decoy_iptm + eps)
  ```

**File:** `src/boltzgen/task/analyze/analyze_utils.py`
- Add `compute_selectivity_metrics(target_metrics, decoy_metrics_list)` helper function

**File:** `src/boltzgen/resources/config/analysis.yaml`
- Add `compute_selectivity: true` flag (default false, enabled when decoys present)

### Step 1.4 — Add Selectivity Filters and Ranking

**File:** `src/boltzgen/task/filter/filter.py`

- Add hard filter: `max_decoy_iptm < threshold` (default 0.5 — reject designs that fold well against any decoy)
- Add hard filter: `selectivity_score > threshold` (default 0.1 — require meaningful selectivity margin)
- Add ranking metric: `selectivity_score` with weight 2 in `self.metrics` dict (line ~203)

**File:** `src/boltzgen/resources/config/filtering.yaml`
- Add default selectivity filter thresholds

**File:** `src/boltzgen/cli/boltzgen.py`
- Add `--decoy_iptm_threshold` CLI argument
- Add `--selectivity_weight` CLI argument for ranking weight

---

## Feature 2: HIS-Tag / Expression Tag Assessment

### Step 2.1 — Tier 1: Sequence-Level Tag Screen (in Analysis)

**File:** `src/boltzgen/task/analyze/analyze_utils.py`

Add `compute_tag_compatibility(structure, design_mask, tag_type="his", tag_position="C")`:
- Identify terminal residues of the designed chain
- Compute distance from tag attachment point (N/C terminus) to nearest interface residue
- Compute solvent accessibility of the terminus (from SASA)
- Flag if terminus is buried at the interface (tag would clash)
- Output metrics:
  ```
  tag_terminus_to_interface_dist   — Å distance from tag end to nearest interface residue
  tag_terminus_sasa                — solvent accessibility at attachment point
  tag_clash_risk                   — boolean: True if terminus is <8Å from interface and buried
  ```

**File:** `src/boltzgen/task/analyze/analyze.py`

In `compute_metrics()`:
- Call `compute_tag_compatibility()` after structural metrics
- Add results to metrics dict

**File:** `src/boltzgen/resources/config/analysis.yaml`
- Add `tag_analysis: true` and `tag_type: "his"` and `tag_position: "C"` config options

### Step 2.2 — Tier 2: Explicit Tag Modeling (new pipeline step)

**File:** `src/boltzgen/cli/boltzgen.py`

- Add a `tag_folding` step after `design_folding`:
  - Appends tag sequence (e.g. `HHHHHH` for 6xHis) to the design chain
  - Runs folding on tagged construct + target
  - Compares metrics with vs without tag

**File:** `src/boltzgen/task/predict/data_from_generated.py`
- Add option to append a tag sequence to the design chain before folding
- New parameter: `append_tag: Optional[str] = None`

### Step 2.3 — Tag-Aware Metrics and Filtering

**File:** `src/boltzgen/task/analyze/analyze.py`

Additional metrics from tagged folding:
```
tagged_iptm                          — complex confidence with tag present
tag_delta_iptm                       — iptm(tagged) - iptm(untagged)
tagged_rmsd                          — structural deviation with tag
tag_stability_impact                 — plddt(tagged design) - plddt(untagged)
```

**File:** `src/boltzgen/task/filter/filter.py`
- Hard filter: `tag_clash_risk == False` (Tier 1)
- Hard filter: `tag_delta_iptm > -0.1` (Tier 2 — tag doesn't drop iptm by more than 0.1)
- Ranking metric: `tag_terminus_to_interface_dist` with weight 1

**File:** `src/boltzgen/cli/boltzgen.py`
- Add `--tag_type {his, strep, flag, custom}` CLI argument
- Add `--tag_sequence` for custom tag
- Add `--tag_position {N, C}` (default C)
- Add `--skip_tag_analysis` flag

---

## Implementation Order

1. **Feature 1.1–1.2** (YAML schema + decoy folding step) — foundational
2. **Feature 2.1** (sequence-level tag screen) — can parallel with above
3. **Feature 1.3** (selectivity metrics in analysis)
4. **Feature 1.4** (selectivity filters)
5. **Feature 2.2** (explicit tag modeling)
6. **Feature 2.3** (tag metrics and filtering)

---

## Critical Files Summary

| File | Changes |
|------|---------|
| `src/boltzgen/data/parse/schema.py` | Add `decoys` YAML field, parse decoy targets |
| `src/boltzgen/data/data.py` | Extend `Target` with `decoys` field |
| `src/boltzgen/cli/boltzgen.py` | New CLI args, `decoy_folding` and `tag_folding` pipeline steps |
| `src/boltzgen/task/predict/data_from_generated.py` | Support alternate target (decoy) and tag appending |
| `src/boltzgen/task/analyze/analyze.py` | Selectivity + tag metrics in `compute_metrics()` |
| `src/boltzgen/task/analyze/analyze_utils.py` | New helpers: `compute_selectivity_metrics()`, `compute_tag_compatibility()` |
| `src/boltzgen/task/filter/filter.py` | New hard filters + ranking weights for selectivity and tag |
| `src/boltzgen/resources/config/analysis.yaml` | New config flags |
| `src/boltzgen/resources/config/filtering.yaml` | New filter defaults |

---

## Verification Plan

1. **Unit test**: Create a test design YAML with a decoys section, verify parsing produces correct `Target.decoys` list
2. **Integration test**: Run `boltzgen configure` with decoy YAML, verify `decoy_folding_*` steps appear in `steps.yaml`
3. **Metrics test**: Run analysis on a pre-computed folding output with mock decoy folding results, verify `selectivity_score` and tag metrics appear in CSV
4. **Filter test**: Run filtering with selectivity hard filter, verify designs with high decoy iptm are removed
5. **End-to-end**: Run `boltzgen run` on a small test case with 1 decoy target and HIS-tag analysis enabled, verify full pipeline completes and `final_ranked_designs/` contains selectivity + tag columns
