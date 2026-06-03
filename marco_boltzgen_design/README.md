# MARCO / Marco Nanobody Design with BoltzGen

De novo VHH (nanobody) design against the SRCR (Scavenger Receptor Cysteine-Rich) domain of **human MARCO** (UniProt Q9UEW3) and **mouse Marco** (PDB 2OYA).

---

## Table of Contents

1. [Quick-Start](#1-quick-start)
2. [Project Structure](#2-project-structure)
3. [Setup](#3-setup)
4. [Interface Sets](#4-interface-sets)
5. [Pipeline Overview](#5-pipeline-overview)
6. [Step 1 — Validate Specs](#step-1--validate-specs)
7. [Step 2 — Local Pilot](#step-2--local-pilot)
8. [Step 3 — HPC Production](#step-3--hpc-production)
9. [Step 4 — Collect Metrics](#step-4--collect-metrics)
10. [Step 5 — Rank & Filter](#step-5--rank--filter)
11. [Step 6 — AF2 Validation](#step-6--af2-validation)
12. [Full Production Workflow](#full-production-workflow)
13. [Reference: Numbering Systems](#reference-numbering-systems)
14. [Spec Files Reference](#14-spec-files-reference)
15. [Reference: MARCO CLI Defaults](#reference-marco-cli-defaults)
16. [Troubleshooting](#troubleshooting)
17. [Appendices](#appendices)

---

## 1. Quick-Start

```bash
# Clone on HPC
git clone https://github.com/jxshi/boltzgen.git
cd boltzgen/marco_boltzgen_design

# Validate all specs (pick one or run all)
conda activate boltzgen
boltzgen check specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml

# Run a local pilot (50 designs, 10 final)
./runs/run_nanobody_campaign.sh specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/pilot

# Run production on HPC (2000 designs → 200 final)
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_prod

# ── Speed mode (2–4× faster, for screening / large batches) ─────────────────
# Sets SPEED_MODE=1 to apply: fold sampling_steps=100, recycling_steps=1,
# diffusion_samples=1, design torch.compile, inverse_fold bf16, diffusion_batch_size=8
SPEED_MODE=1 NUM_DESIGNS=5000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_fast
```

---

## 2. Project Structure

```
marco_boltzgen_design/
├── targets/
│   ├── human_MARCO_input.cif     # Human MARCO full sequence (Q9UEW3, 1–520 aa)
│   └── mouse_marco_srcr.cif     # Mouse Marco SRCR domain (PDB 2OYA, 102 aa)
├── specs/
│   ├── _defaults.md             # MARCO runtime defaults (CLI flags reference)
│   ├── mouse_marco_nanobody_hotspot.yaml           # Original SO4-site DSSP hotspots
│   ├── human_marco_nanobody_hotspot.yaml
│   ├── crossreactive_marco_nanobody_hotspot.yaml
│   ├── mouse_marco_nanobody_setA_so4_pocket.yaml  # ← SET A
│   ├── human_marco_nanobody_setA_so4_pocket.yaml  # ← SET A
│   ├── human_marco_nanobody_setB_patent_epitope.yaml  # ← SET B
│   ├── crossreactive_marco_nanobody_setC_hybrid.yaml # ← SET C
│   ├── mouse_marco_nanobody_setD_beta_pairing.yaml   # ← SET D
│   ├── human_marco_nanobody_setD_beta_pairing.yaml  # ← SET D
│   └── crossreactive_marco_nanobody_setD_beta_pairing.yaml # ← SET D
├── scripts/
│   ├── run_hpc_campaign.sh       # SLURM submission (primary production script)
│   ├── collect_campaign.sh       # Merge metrics from multiple campaigns
│   ├── rank_and_validate.sh      # Stage 3 (rank) + Stage 4 (AF2 validation)
│   ├── rank_designs.py           # Score & rank by metrics + conservation
│   └── validate_designs.py       # AF2 backfold validation
├── runs/                         # Campaign output (designs, metrics)
├── results/                      # Merged metrics, ranked lists, AF2 reports
└── README.md
```

---

## 3. Setup

### 3.1 Clone the Repository

```bash
git clone https://github.com/jxshi/boltzgen.git
cd boltzgen/marco_boltzgen_design
```

### 3.2 Activate Conda Environment

```bash
conda activate boltzgen
# or
source ~/miniconda3/etc/profile.d/conda.sh && conda activate boltzgen

# boltzgen CLI is at: ~/miniconda3/envs/boltzgen/bin/boltzgen
```

### 3.3 Download Models (first time only)

```bash
boltzgen run --force_download specs/mouse_marco_nanobody_hotspot.yaml \
  --output /tmp/test_model_dl --num_designs 1 --budget 1
```

---

## 4. Interface Sets

Four strategy groups cover distinct SRCR surfaces and species scopes. All use `--protocol nanobody-anything` with MARCO-specific filtering defaults.

| Set | Strategy | Species | Specs | Priority |
|-----|----------|---------|-------|----------|
| **D** | Beta-edge strand targeting — polar beta-sheet face | Both / Human / Mouse | `*_setD_beta_pairing.yaml` | 🔴 Highest |
| **C** | Hybrid interface — Sets A + B union | Cross-reactive | `crossreactive_*_setC_hybrid.yaml` | 🔴 Highest |
| **A** | SO₄/pocket blocking — ligand-binding crevice | Mouse / Human | `*_setA_so4_pocket.yaml` | 🟡 High |
| **B** | Patent antibody epitope | Human only | `*_setB_patent_epitope.yaml` | 🟡 High |
| Hotspot | ARG-rich basic patch (conserved) | Various | `*_hotspot.yaml`, `*_conserved_surface.yaml` | 🟢 Medium |
| Anywhere | Unconstrained surface exploration | Various | `*_anywhere.yaml` | 🔵 Exploratory |

> 💡 **Detailed per-spec instructions** (binding residues, validate/pilot/HPC commands) are in [Section 14 — Spec Files Reference](#14-spec-files-reference).

### Set A — SO₄ / Ligand-Blocking Pocket
Block the ligand-binding crevice (LDL, oxLDL, bacteria, apoptotic cells).
- **Mouse:** 2OYA label_seq 12,14,21,50,56,58,78,89
- **Human:** Q9UEW3 label_seq 429,431,438,467,473,475,495,506

### Set B — Patent Antibody Epitope
Target the PI-3010/PI-3035 antibody epitope (human-only; Q452 differs in mouse).
- **Human:** Q9UEW3 label_seq 450,452,472,473,487,499,505,507,509,511

> ⚠️ Patent-reported amino acid identities were re-verified against mmCIF — several differed.

### Set C — Hybrid Interface (Cross-Reactive)
Combine Sets A + B for maximum paratope breadth and interface stability.
- **Human:** 17 residues (union of Sets A + B)
- **Mouse:** 8 residues from the crevice entrance

### Set D — Beta-Pairing SRCR Edge-Strand Targeting
Exploit exposed beta-strand edges for backbone-like, H-bond-rich VHH contacts on the polar beta-sheet face. Best for hydrophilic SRCR surfaces where hydrophobic filtering underperforms.
- **Human:** Q9UEW3 label_seq 423,424,426,428,430,431,433,436,437,438,466,468,470,514,517,518
- **Mouse:** 2OYA label_seq 7,8,10,12,14,15,17,20,21,22,50,52,54,98,101,102

---

## Step 1 — Validate Specs

Always validate before submitting jobs. This checks that all CIF residue IDs exist and writes a `.cif` visualization file.

```bash
conda activate boltzgen

# Validate all Set D specs
boltzgen check specs/mouse_marco_nanobody_setD_beta_pairing.yaml
boltzgen check specs/human_marco_nanobody_setD_beta_pairing.yaml
boltzgen check specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml

# Validate Sets A, B, C
boltzgen check specs/mouse_marco_nanobody_setA_so4_pocket.yaml
boltzgen check specs/human_marco_nanobody_setA_so4_pocket.yaml
boltzgen check specs/human_marco_nanobody_setB_patent_epitope.yaml
boltzgen check specs/crossreactive_marco_nanobody_setC_hybrid.yaml
boltzgen check specs/mouse_marco_nanobody_setD_beta_pairing.yaml
boltzgen check specs/human_marco_nanobody_setD_beta_pairing.yaml
boltzgen check specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml
```

All specs should write a `.cif` visualization file. Exact designed-residue counts vary by scaffold and targeted residue set.

---

## Step 2 — Local Pilot

Run a small batch locally to verify the pipeline works before submitting an HPC job. Use the `run_nanobody_campaign.sh` wrapper, which applies all MARCO defaults automatically:

### Local pilot (quick test, 10–50 designs)

The wrapper now applies the recommended MARCO defaults automatically: `--protocol nanobody-anything`, `--diffusion_batch_size 2`, `plip_hbonds_refolded=0.2`, `delta_sasa_refolded=0.5`, and `--refolding_rmsd_threshold 3.0`.

```bash
# Set D cross-reactive pilot (recommended first run)
NUM_DESIGNS=50 BUDGET=10 \
  ./runs/run_nanobody_campaign.sh specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_pilot
# → Output: runs/setD_pilot/final_ranked_designs/all_designs_metrics.csv
```

Equivalent explicit command:

```bash
boltzgen run specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml \
  --protocol nanobody-anything \
  --output runs/setD_pilot \
  --num_designs 50 \
  --diffusion_batch_size 2 \
  --budget 10 \
  --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 \
  --refolding_rmsd_threshold 3.0
# → Output: runs/setD_pilot/final_ranked_designs/all_designs_metrics.csv
```

Set C hybrid pilot (maximum interface breadth):

```bash
NUM_DESIGNS=50 BUDGET=10 \
  ./runs/run_nanobody_campaign.sh specs/crossreactive_marco_nanobody_setC_hybrid.yaml runs/setC_pilot
# → Output: runs/setC_pilot/final_ranked_designs/all_designs_metrics.csv
```

**Output:** `runs/setD_pilot/final_ranked_designs/all_designs_metrics.csv`

To permit Cysteine in CDRs (disabled by default in nanobody-anything):
```bash
EXTRA_ARGS='--inverse_fold_avoid ""' ./runs/run_nanobody_campaign.sh \
  specs/... runs/...
```

---

## Step 3 — HPC Production

### 3.1 Standard Production (quality mode)

```bash
# Set D — beta-pairing cross-reactive
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml \
  runs/setD_beta_pairing

# Set C — hybrid cross-reactive
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setC_hybrid.yaml \
  runs/setC
```

### 3.2 Speed Mode (2–4× faster, for screening / large batches)

Set `SPEED_MODE=1` to apply these optimizations automatically:

| Step | Parameter | Default → Speed Mode | Est. speedup |
|------|-----------|----------------------|--------------|
| **fold** | `sampling_steps` | 200 → **100** | ~2× |
| **fold** | `recycling_steps` | 3 → **1** | ~3× per forward |
| **fold** | `diffusion_samples` | 5 → **1** | 5× fewer passes |
| **design** | `compile_pairformer` | false → **true** | ~20–40% |
| **design** | `compile_structure` | false → **true** | ~20–40% |
| **inverse_fold** | `precision` | FP32 → **bf16-mixed** | ~2× |
| **global** | `diffusion_batch_size` | 2 → **8** | better GPU util. |

```bash
# Speed mode — recommended for 5000+ design batches
SPEED_MODE=1 NUM_DESIGNS=5000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml \
  runs/setD_fast
```

### 3.3 All-Sets Production (Example Full Batch)

```bash
# ── Set A ──
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_setA_so4_pocket.yaml runs/setA_mouse &
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setA_so4_pocket.yaml runs/setA_human &

# ── Set B ──
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setB_patent_epitope.yaml runs/setB &

# ── Set C ──
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setC_hybrid.yaml runs/setC &

# ── Set D ──
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD &

wait
```

**Output location:** `runs/<name>/final_ranked_designs/all_designs_metrics.csv`

---

## Step 4 — Collect Metrics

After all HPC jobs complete, merge metrics from all campaigns into a single CSV:

```bash
./scripts/collect_campaign.sh \
  --runs runs/setA_mouse runs/setA_human runs/setB runs/setC runs/setD_beta_pairing \
  --out results/all_metrics.csv
```

This adds `source_run` and `source_spec` columns to each row and deduplicates designs that may appear across runs.

**Expected output:** `results/all_metrics.csv` with one row per design.

---

## Step 5 — Rank & Filter

Rank by composite score (pLDDT, ipTM, interface metrics) and apply conservation and developability filters:

```bash
./scripts/rank_and_validate.sh \
  --metrics results/all_metrics.csv \
  --human-conserved "A:423,A:424,A:431,A:460,A:466,A:468,A:488,A:499" \
  --mouse-conserved "A:6,A:8,A:15,A:44,A:50,A:52,A:72,A:83" \
  --top_n 50 \
  --out_rank results/ranked_candidates.csv
```

For **Set C** (cross-reactive hybrid):
```bash
./scripts/rank_and_validate.sh \
  --metrics results/all_metrics.csv \
  --human-conserved "A:35,A:50,A:56,A:58,A:70,A:78,A:82,A:88,A:89,A:90,A:92,A:94" \
  --mouse-conserved "A:12,A:14,A:21,A:50,A:56,A:58,A:78,A:89" \
  --top_n 50 \
  --out_rank results/ranked_candidates.csv
```

For **Set B** (human-only patent epitope):
```bash
./scripts/rank_and_validate.sh \
  --metrics results/all_metrics.csv \
  --human-conserved "A:33,A:35,A:56,A:70,A:82,A:88,A:90,A:92,A:94" \
  --top_n 50 \
  --out_rank results/ranked_candidates.csv
```

**Output:** `results/ranked_candidates.csv`

---

## Step 6 — AF2 Validation

AF2 backfold validation checks whether the designed sequences refold to the predicted structures. The `rank_and_validate.sh` script calls `validate_designs.py` which runs AF2 on the top-50 ranked CIFs:

```bash
# After rank_and_validate.sh completes, run AF2 validation separately:
python scripts/validate_designs.py \
  --complexes results/candidate_cifs \
  --metrics results/ranked_candidates.csv \
  --top_n 50 \
  --method colabfold \
  --out results/af_validation.csv
```

If CIFs are still on HPC, copy them first:
```bash
rsync -avz hpc:path/to/runs/*/final_ranked_designs/*.cif results/candidate_cifs/
```

**Output:** `results/af_validation.csv` with per-design pLDDT/RMSD metrics for backfold validation.

---

## Full Production Workflow

```bash
# 1. Clone & validate
git clone https://github.com/jxshi/boltzgen.git
cd boltzgen/marco_boltzgen_design
conda activate boltzgen
boltzgen check specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml

# 2. Local pilot
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/pilot

Legacy `binder` and `peptide` filenames in `specs/` have been converted to VHH scaffold specs and should also be run with `--protocol nanobody-anything`.

The original DSSP-derived hotspot specs remain available:
- `specs/mouse_marco_nanobody_hotspot.yaml` — label_seq 6,8,15,44,50,52,72,83
- `specs/human_marco_nanobody_hotspot.yaml` — label_seq 6,8,15,44,50,52,72,83 (offset-applied)
- `specs/crossreactive_marco_nanobody_hotspot.yaml` — cross-reactive union

---

## Reference: Numbering Systems

| System | Offset | Range | Used for |
|--------|--------|-------|---------|
| **Q9UEW3** (human) | — | 1–520 | Full sequence positions |
| mmCIF `label_seq` (human) | Q9UEW3 − 417 | 1–103 | Human spec `binding:` field |
| 2OYA `label_seq` (mouse) | Direct (1–102) | 1–102 | Mouse spec `binding:` field |

**Q9UEW3 → label_seq conversion:**
- Human: `label_seq = Q9UEW3_position − 417`
- Mouse (2OYA): use 2OYA `label_seq` directly (1–102)

**Example:** Q9UEW3 position 452 → `label_seq = 452 − 417 = 35`

> ⚠️ **Common error:** Do NOT use Q9UEW3 positions directly in the YAML `binding:` field. The mmCIF uses `label_seq`, not the full-sequence Q9UEW3 position. For mouse (2OYA), the offset is **+416** (label_seq 1 = Q9UEW3 417), not +417.

---
## 14. Spec Files Reference

All 19 YAML spec files target distinct design strategies. All accept the same MARCO runtime defaults (`--protocol nanobody-anything`, `--diffusion_batch_size 2`, etc.) applied automatically by the wrapper scripts. Pick the spec matching your biological goal.

### Spec naming key

| Pattern | Meaning |
|---------|---------|
| `crossreactive_*` | Targets both human MARCO (Q9UEW3) and mouse Marco (2OYA) simultaneously |
| `*_nanobody_*` | VHH scaffold (7EOW 21-aa CDR3, 8COH 19-aa CDR3) |
| `*_binder_*` | General binder scaffold (non-VHH-specific) |
| `*_peptide_*` | Peptide-length scaffold |
| `setA` | SO₄/pocket blocking — ligand-binding crevice |
| `setB` | Patent antibody epitope — human-only |
| `setC` | Hybrid (Sets A + B union) — cross-reactive |
| `setD` | Beta-edge strand targeting — exposed beta-sheet face |
| `*_hotspot` | ARG-rich conserved basic patch (verified from SO₄ sites) |
| `*_anywhere` | No hotspot constraint — unconstrained surface exploration |

### Priority recommendation

**Start here:**
- **Set D** (`crossreactive_marco_nanobody_setD_beta_pairing.yaml`) — highest-priority exploratory set for the hydrophilic SRCR surface
- **Set C** (`crossreactive_marco_nanobody_setC_hybrid.yaml`) — maximum paratope breadth for therapeutic applications
- **Set A** (`*_setA_so4_pocket.yaml`) — ligand-blocking for LDL/oxLDL/bacteria/apoptotic cell interference

---

### Cross-reactive specs (human + mouse)

Design against both species simultaneously. Requires nanobodies that bind both human MARCO and mouse Marco.

| Spec | Residues | Strategy | Priority |
|------|----------|----------|----------|
| `crossreactive_marco_nanobody_setD_beta_pairing.yaml` | Human: 423,424,426,428,430,431,433,436,437,438,466,468,470,514,517,518 · Mouse: 7,8,10,12,14,15,17,20,21,22,50,52,54,98,101,102 | Beta-edge targeting (16/16 residues) | 🔴 Highest |
| `crossreactive_marco_nanobody_setC_hybrid.yaml` | Human: 429,431,438,450,452,467,472,473,475,487,495,499,505,506,507,509,511 · Mouse: 12,14,21,50,56,58,78,89 | Hybrid interface — Sets A + B union | 🔴 Highest |
| `crossreactive_marco_nanobody_hotspot.yaml` | Human: 423,425,432,461,467,469,489,500 · Mouse: 6,8,15,44,50,52,72,83 | ARG-rich conserved basic patch | 🟢 Medium |
| `crossreactive_conserved_surface.yaml` | Same as `_hotspot.yaml` | Broad conserved-surface targeting | 🟢 Medium |

**Set D (beta-pairing):** Targets exposed beta-strand edges on the SRCR fold for backbone-like H-bond contacts on the polar beta-sheet face. Best when hydrophobic filtering may underperform on hydrophilic surfaces.

**Set C (hybrid):** Combines the SO₄/pocket (Set A) with the patent epitope (Set B) for maximum interface breadth. Best for therapeutic applications requiring broad MARCO interference.

**Notes:**
- Patent-based position claims in Set C were verified against mmCIF — several patent-reported amino acid identities were corrected.
- Set C human Q452 = TYR (not the patent's claimed residue at that position).

---

### Human MARCO specs

| Spec | Residues (Q9UEW3 label_seq) | Strategy | Priority |
|------|----------------------------|----------|----------|
| `human_marco_nanobody_setD_beta_pairing.yaml` | 423,424,426,428,430,431,433,436,437,438,466,468,470,514,517,518 | Beta-edge targeting | 🟡 High |
| `human_marco_nanobody_setA_so4_pocket.yaml` | 429,431,438,467,473,475,495,506 | SO₄/pocket blocking | 🟡 High |
| `human_marco_nanobody_setB_patent_epitope.yaml` | 450,452,472,473,487,499,505,507,509,511 | Patent PI-3010/PI-3035 epitope | 🟡 High |
| `human_marco_nanobody_hotspot.yaml` | 423,425,432,461,467,469,489,500 | ARG-rich conserved basic patch | 🟢 Medium |
| `human_marco_nanobody_anywhere.yaml` | None (unconstrained) | Unconstrained surface exploration | 🔵 Exploratory |
| `human_marco_binder_hotspot.yaml` | 423,425,432,461,467,469,489,500 | Hotspot targeting (general binder) | 🟢 Medium |
| `human_marco_binder_anywhere.yaml` | None (unconstrained) | Unconstrained (general binder) | 🔵 Exploratory |
| `human_marco_peptide_anywhere.yaml` | None (unconstrained) | Peptide-length unconstrained | 🔵 Exploratory |

**Set B (patent epitope):** Human-only. Q452 = TYR in human (mouse = D452). Do not use for cross-reactive designs. Patent text contained incorrect amino acid claims — all positions re-verified against `human_MARCO_input.cif`.

**`binder_*` specs:** Use general binder scaffolds rather than VHH-specific ones. Less CDR geometry constraint; consider `*_nanobody_*` variants for tighter VHH geometry.

---

### Mouse Marco specs

| Spec | Residues (2OYA label_seq) | Strategy | Priority |
|------|---------------------------|----------|----------|
| `mouse_marco_nanobody_setA_so4_pocket.yaml` | 12,14,21,50,56,58,78,89 | SO₄/pocket blocking | 🟡 High |
| `mouse_marco_nanobody_setD_beta_pairing.yaml` | 7,8,10,12,14,15,17,20,21,22,50,52,54,98,101,102 | Beta-edge targeting | 🟡 High |
| `mouse_marco_nanobody_hotspot.yaml` | 6,8,15,44,50,52,72,83 | ARG-rich conserved basic patch | 🟢 Medium |
| `mouse_marco_nanobody_anywhere.yaml` | None (unconstrained) | Unconstrained surface exploration | 🔵 Exploratory |
| `mouse_marco_binder_hotspot.yaml` | 6,8,15,44,50,52,72,83 | Hotspot targeting (general binder) | 🟢 Medium |
| `mouse_marco_binder_anywhere.yaml` | None (unconstrained) | Unconstrained (general binder) | 🔵 Exploratory |
| `mouse_marco_peptide_anywhere.yaml` | None (unconstrained) | Peptide-length unconstrained | 🔵 Exploratory |

**Set A vs hotspot:** Set A uses 12,14,21,50,56,58,78,89 (crevice entrance / SO₄ pocket). The original hotspot spec uses 6,8,15,44,50,52,72,83 (deeper ARG-rich basic patch at SO₄ sites). Both are valid — choose based on whether you want deep-site or crevice-entrance targeting.

**Hotspot residues verified:** All from PDB 2OYA SO₄ binding sites AC1/AC2/AC3. 6(R), 8(R), 15(R), 44(R), 50(R), 52(R), 72(R), 83(K).

---

### Validate all specs

```bash
conda activate boltzgen

# All Set D specs
boltzgen check specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml
boltzgen check specs/human_marco_nanobody_setD_beta_pairing.yaml
boltzgen check specs/mouse_marco_nanobody_setD_beta_pairing.yaml

# All Set C / A / B specs
boltzgen check specs/crossreactive_marco_nanobody_setC_hybrid.yaml
boltzgen check specs/human_marco_nanobody_setA_so4_pocket.yaml
boltzgen check specs/mouse_marco_nanobody_setA_so4_pocket.yaml
boltzgen check specs/human_marco_nanobody_setB_patent_epitope.yaml

# Hotspot / cross-reactive
boltzgen check specs/crossreactive_marco_nanobody_hotspot.yaml
boltzgen check specs/crossreactive_conserved_surface.yaml
boltzgen check specs/human_marco_nanobody_hotspot.yaml
boltzgen check specs/mouse_marco_nanobody_hotspot.yaml

# Anywhere (unconstrained — discovery mode)
boltzgen check specs/human_marco_nanobody_anywhere.yaml
boltzgen check specs/mouse_marco_nanobody_anywhere.yaml
```

---

### Local pilot commands

```bash
# Set D — cross-reactive (recommended first run)
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_pilot

# Set C — hybrid cross-reactive
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_marco_nanobody_setC_hybrid.yaml runs/setC_pilot

# Set A — mouse SO₄/pocket
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/mouse_marco_nanobody_setA_so4_pocket.yaml runs/setA_mouse_pilot

# Set A — human SO₄/pocket
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_nanobody_setA_so4_pocket.yaml runs/setA_human_pilot

# Set B — patent epitope (human-only)
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_nanobody_setB_patent_epitope.yaml runs/setB_pilot

# Set D — human-only
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_nanobody_setD_beta_pairing.yaml runs/setD_human_pilot

# Set D — mouse-only
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/mouse_marco_nanobody_setD_beta_pairing.yaml runs/setD_mouse_pilot

# Hotspot specs
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_hotspot_pilot

NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml runs/human_hotspot_pilot

NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_marco_nanobody_hotspot.yaml runs/hotspot_pilot

# Anywhere (unconstrained)
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_nanobody_anywhere.yaml runs/human_anywhere_pilot

NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/mouse_marco_nanobody_anywhere.yaml runs/mouse_anywhere_pilot
```

**Permit Cysteine in CDRs** (disabled by default in `nanobody-anything`):
```bash
EXTRA_ARGS='--inverse_fold_avoid ""' ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD
```

---

### HPC production commands

```bash
# ── SET D ──
# Beta-pairing — highest priority exploratory
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_prod

NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setD_beta_pairing.yaml runs/setD_human_prod

NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_setD_beta_pairing.yaml runs/setD_mouse_prod

# ── SET C ──
# Hybrid — maximum paratope breadth
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setC_hybrid.yaml runs/setC_prod

# ── SET A ──
# SO₄/pocket blocking
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_setA_so4_pocket.yaml runs/setA_mouse_prod

NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setA_so4_pocket.yaml runs/setA_human_prod

# ── SET B ──
# Patent epitope — human-only
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setB_patent_epitope.yaml runs/setB_prod

# ── HOTSPOT / CROSS-REACTIVE ──
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_hotspot.yaml runs/hotspot_prod

NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_conserved_surface.yaml runs/conserved_surface_prod

NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml runs/human_hotspot_prod

NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_hotspot_prod

# ── ANYWHERE (unconstrained) ──
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_anywhere.yaml runs/human_anywhere_prod

NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_anywhere.yaml runs/mouse_anywhere_prod
```

**Run multiple specs in parallel** (dual-GPU or SLURM array):
```bash
# Parallel submission
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_setA_so4_pocket.yaml runs/setA_mouse &

NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setA_so4_pocket.yaml runs/setA_human &

wait  # wait for all parallel jobs
```

**Direct bash (no SLURM):**
```bash
NUM_DESIGNS=2000 BUDGET=200 GPUS=1 bash scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_direct
```

**Output:** `runs/<name>/final_ranked_designs/all_designs_metrics.csv`


## Reference: MARCO CLI Defaults

All MARCO nanobody designs should use these flags:

```bash
--protocol nanobody-anything
--diffusion_batch_size 2
--metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5
--refolding_rmsd_threshold 3.0
```

| Flag | Value | Rationale |
|------|-------|-----------|
| `--protocol` | `nanobody-anything` | VHH-specific bias; avoids non-native Cys; favors CDR loop geometry |
| `--diffusion_batch_size` | `2` | Designs in the same batch share length — keep small (1–2) for length diversity |
| `plip_hbonds_refolded` | `0.2` | Lower = more important. Up-weights buried H-bonds — critical for MARCO's polar SRCR surface |
| `delta_sasa_refolded` | `0.5` | Emphasizes interface burial area, not hydrophobic core burial |
| `--refolding_rmsd_threshold` | `3.0` | Deliberately relaxed for VHH loop flexibility during refolding |

These flags are applied automatically by `runs/run_nanobody_campaign.sh` and `scripts/run_hpc_campaign.sh`.

---

## Troubleshooting

### boltzgen: error: unrecognized arguments
Diffusion parameters (e.g. `--diffusion.num_steps`, `--diffusion.guidance_scale`) are YAML-only, **not** CLI arguments. Do not pass them as EXTRA_ARGS. To change diffusion settings, edit the spec YAML directly.

### OOM / killed on local Mac
Mac local runs OOM-kill at ~50 designs. Always use HPC for production runs (2000+ designs).

### No metrics CSV after job finishes
Check the SLURM log: `logs/<spec_name>_<jobid>_run.log`. A killed job (time limit or OOM) may produce no output. Increase `--time` or reduce `NUM_DESIGNS`/`BUDGET`.

### boltzgen check passes but HPC job fails
Check that the working directory is `marco_boltzgen_design/` and that relative paths in the YAML (e.g. `../targets/mouse_marco_srcr.cif`) resolve correctly from that location.

### Missing residue IDs in boltzgen check
The `binding:` positions in the YAML don't match the mmCIF `label_seq` numbering. Verify with gemmi:
```python
import gemmi
doc = gemmi.read_file('targets/mouse_marco_srcr.cif')
# inspect label_seq values
```

### Wrong amino acid residues
Always verify AA identities against the actual mmCIF structure with gemmi, not from patent text or sequence alignments. The patent claims are frequently incorrect (e.g. Q450 → actual VAL at that position in AF-Q9UEW3-F1).

---

## Appendices

### Appendix A: Original DSSP-Derived Hotspot Specs

The original SO₄²⁻ binding site hotspots remain available:
- `specs/mouse_marco_nanobody_hotspot.yaml` — label_seq 6,8,15,44,50,52,72,83 (all ARG except 83=LYS)
- `specs/human_marco_nanobody_hotspot.yaml`
- `specs/crossreactive_marco_nanobody_hotspot.yaml`

These target the conserved basic patch and are less specific than Sets A–D.

### Appendix B: Allow Cysteine in CDRs

By default, `nanobody-anything` avoids Cysteine. To permit Cys:
```bash
EXTRA_ARGS='--inverse_fold_avoid ""' ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD
```

### Appendix C: SLURM Log Locations

After HPC jobs, SLURM logs are at:
```
logs/<spec_name>_<jobid>_start.log   # environment at start
logs/<spec_name>_<jobid>_run.log    # boltzgen output
logs/<spec_name>_<jobid>_done.log    # completion summary
```

### Appendix D: Hotspot Discovery

To identify new interfaces from antibody complex structures:
```bash
python scripts/find_marco_srcr_hotspots.py \
  --human-structure targets/human_MARCO_input.cif --human-chain A \
  --mouse-structure targets/mouse_marco_srcr.cif --mouse-chain A \
  --human-complexes runs/human_complexes/*.cif \
  --mouse-complexes runs/mouse_complexes/*.cif \
  --human-binder-chains H,L --mouse-binder-chains H,L \
  --out results/marco_hotspots.csv
```
