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
14. [Reference: MARCO CLI Defaults](#reference-marco-cli-defaults)
15. [Troubleshooting](#troubleshooting)
16. [Appendices](#appendices)

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

Four distinct interface sets have been designed based on structural, patent, and beta-pairing design logic. All are intended to be run with `--protocol nanobody-anything` and MARCO-specific filtering flags.

### Set A — SO4 / Ligand-Blocking Pocket
**Block the ligand-binding crevice** (LDL, oxLDL, bacteria, apoptotic cells).

| Spec | label_seq | Notes |
|------|-----------|-------|
| `specs/mouse_marco_nanobody_setA_so4_pocket.yaml` | 12,14,21,50,56,58,78,89 | Mouse SRCR (2OYA) |
| `specs/human_marco_nanobody_setA_so4_pocket.yaml` | 429,431,438,467,473,475,495,506 | Human MARCO (Q9UEW3 full-length numbering) |

### Set B — Patent Antibody Epitope
**Target the PI-3010/PI-3035 antibody epitope** (human-only; cross-species conservation differs).

| Spec | label_seq | Notes |
|------|-----------|-------|
| `specs/human_marco_nanobody_setB_patent_epitope.yaml` | 450,452,472,473,487,499,505,507,509,511 | Human only |

> ⚠️ Human Q452 = mouse D452 (species差异). Set B is human-only.

### Set C — Hybrid Interface (Cross-Reactive)
**Combine Sets A + B** for maximum paratope breadth and interface stability.

| Spec | Human label_seq | Mouse label_seq | Notes |
|------|----------------|-----------------|-------|
| `specs/crossreactive_marco_nanobody_setC_hybrid.yaml` | 429,431,438,450,452,467,472,473,475,487,495,499,505,506,507,509,511 | 12,14,21,50,56,58,78,89 | Cross-reactive |

**Key conserved residue: 473** (K in both human and mouse) — appears in Sets A–C.

### Set D — Beta-Pairing SRCR Edge-Strand Targeting
**Purpose:** Exploit exposed beta-strand edges on the SRCR fold to favor backbone-like, hydrogen-bond-rich VHH contacts on a polar beta-sheet surface.
**Use case:** Highest-priority exploratory set for hydrophilic SRCR surfaces where generic hydrophobic binder filtering may underperform.

| Spec | Human label_seq | Mouse label_seq | Notes |
|------|----------------|-----------------|-------|
| `specs/human_marco_nanobody_setD_beta_pairing.yaml` | 423,424,426,428,430,431,433,436,437,438,466,468,470,514,517,518 | — | Human beta-edge |
| `specs/mouse_marco_nanobody_setD_beta_pairing.yaml` | — | 7,8,10,12,14,15,17,20,21,22,50,52,54,98,101,102 | Mouse 2OYA beta-edge |
| `specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml` | 423,424,426,428,430,431,433,436,437,438,466,468,470,514,517,518 | 7,8,10,12,14,15,17,20,21,22,50,52,54,98,101,102 | Cross-reactive beta-pairing |

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
  ./runs/run_nanobody_campaign.sh specs/crossreactive_marco_nanobody_setC_hybrid.yaml runs/setC_pilot
```

Equivalent explicit command:

```bash
boltzgen run specs/crossreactive_marco_nanobody_setC_hybrid.yaml \
  --protocol nanobody-anything \
  --output runs/setC_pilot \
  --num_designs 50 \
  --diffusion_batch_size 2 \
  --budget 10 \
  --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 \
  --refolding_rmsd_threshold 3.0
```

Set D beta-pairing pilot:

```bash
boltzgen run specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml \
  --protocol nanobody-anything \
  --output runs/setD_beta_pairing_pilot \
  --num_designs 50 \
  --diffusion_batch_size 2 \
  --budget 10 \
  --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 \
  --refolding_rmsd_threshold 3.0
```

**Output:** `runs/setD_pilot/final_ranked_designs/all_designs_metrics.csv`

To permit Cysteine in CDRs (disabled by default in nanobody-anything):
```bash
EXTRA_ARGS='--inverse_fold_avoid ""' ./runs/run_nanobody_campaign.sh \
  specs/... runs/...
```

---

## Step 3 — HPC Production

### 3.1 Single-GPU Job (RTX 5000, 16 GB VRAM)

```bash
# Set D — beta-pairing cross-reactive
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml \
  runs/setD_beta_pairing

# Set C — hybrid cross-reactive
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setC_hybrid.yaml \
  runs/setC

# Set A — mouse
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_setA_so4_pocket.yaml \
  runs/setA_mouse

# Set A — human
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setA_so4_pocket.yaml \
  runs/setA_human

# Set B — patent epitope (human-only)
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setB_patent_epitope.yaml \
  runs/setB

# Set D — beta-pairing cross-reactive
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_beta_pairing &

wait
```


**Bash CLI (no `sbatch`, run directly):**
```bash
# Run directly from shell on a GPU node (or local workstation with GPUs).
# This uses the same script without SLURM submission.
NUM_DESIGNS=2000 BUDGET=200 GPUS=1 bash scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setC_hybrid.yaml runs/setC_direct

# Run Set D directly
NUM_DESIGNS=2000 BUDGET=200 GPUS=1 bash scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_beta_pairing_direct
```

**Dual-GPU (both RTX 5000s, parallel jobs):**
```bash
# Job 1 — GPU 1
GPUS=1 NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setA_so4_pocket.yaml \
  runs/setA_human_gpu1 &

# Job 2 — GPU 2
GPUS=1 NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_setA_so4_pocket.yaml \
  runs/setA_mouse_gpu2 &

wait
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

|| System | Offset | Range | Used for |
|--------|--------|-------|---------|
|| **Q9UEW3** (human) | — | 1–520 | Full sequence positions |
|| mmCIF `label_seq` (human) | Q9UEW3 − 417 | 1–103 | Human spec `binding:` field |
|| 2OYA `label_seq` (mouse) | Direct (1–102) | 1–102 | Mouse spec `binding:` field |

**Q9UEW3 → label_seq conversion:**
- Human: `label_seq = Q9UEW3_position − 417`
- Mouse (2OYA): use 2OYA `label_seq` directly (1–102)

**Example:** Q9UEW3 position 452 → `label_seq = 452 − 417 = 35`

> ⚠️ **Common error:** Do NOT use Q9UEW3 positions directly in the YAML `binding:` field. The mmCIF uses `label_seq`, not the full-sequence Q9UEW3 position. For mouse (2OYA), the offset is **+416** (label_seq 1 = Q9UEW3 417), not +417.

---

## Reference: Per-Spec File Instructions

Each YAML file targets a distinct design strategy. All specs accept the same MARCO runtime defaults and are run with the same wrapper scripts. Pick the spec that matches your biological goal.

> **TL;DR** — Most users should start with **Set D** (`crossreactive_marco_nanobody_setD_beta_pairing.yaml`) or **Set C** (`crossreactive_marco_nanobody_setC_hybrid.yaml`) for cross-reactive designs, or **Set A** (`*_setA_so4_pocket.yaml`) for ligand-blocking.

### Spec naming key
- `nanobody_*` — VHH scaffolds (7EOW 21-aa CDR3, 8COH 19-aa CDR3). Use `--protocol nanobody-anything`.
- `binder_*` — general binder scaffolds (same CLI flags, less VHH-specific bias).
- `peptide_*` — peptide scaffold targets.
- `hotspot` — targets the ARG-rich conserved SO₄/basic patch (Sets A/B/C/D precursor).
- `setA` — SO₄/pocket blocking (ligand-binding crevice).
- `setB` — patent antibody epitope (human-only).
- `setC` — hybrid cross-reactive (Sets A + B union).
- `setD` — beta-edge strand targeting (exposed beta-sheet face).
- `crossreactive_*` — designs against both human MARCO (Q9UEW3) and mouse Marco (2OYA) simultaneously; requires a cross-reactive nanobody that binds both.
- `*_anywhere` — no hotspot constraint; designs against any accessible surface of the target.
- `*_hotspot` — targets the conserved ARG-rich basic patch (hotspot positions only).

---

### crossreactive_conserved_surface.yaml

**Strategy:** Cross-reactive design targeting the conserved ARG-rich basic patch shared between human and mouse MARCO. This is the same functional surface as `crossreactive_marco_nanobody_hotspot.yaml` but uses a broader definition of conserved residues.

**Binding residues:**
- Human (Q9UEW3): 423, 425, 432, 461, 467, 469, 489, 500
- Mouse (2OYA): 6, 8, 15, 44, 50, 52, 72, 83

**Scaffolds:** 7EOW (21-aa CDR3) + 8COH (19-aa CDR3)

**When to use:** Broad cross-reactivity screening with a focused hotspot set. Good for testing whether cross-reactive nanobodies can simultaneously engage both species.

**Validate:**
```bash
boltzgen check specs/crossreactive_conserved_surface.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_conserved_surface.yaml runs/conserved_surface_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_conserved_surface.yaml runs/conserved_surface_prod
```

---

### crossreactive_marco_nanobody_hotspot.yaml

**Strategy:** Cross-reactive nanobody design targeting the conserved ARG-rich basic patch in both human and mouse. Uses the same hotspot set as `crossreactive_conserved_surface.yaml` with VHH-specific scaffold bias.

**Binding residues:**
- Human (Q9UEW3): 423, 425, 432, 461, 467, 469, 489, 500
- Mouse (2OYA): 6, 8, 15, 44, 50, 52, 72, 83

**Scaffolds:** 7EOW (21-aa CDR3, recommended) + 8COH (19-aa CDR3)

**When to use:** Cross-reactive screening with VHH-specific CDR constraints. Preferred over `crossreactive_conserved_surface.yaml` when using `nanobody-anything` protocol.

**Validate:**
```bash
boltzgen check specs/crossreactive_marco_nanobody_hotspot.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_marco_nanobody_hotspot.yaml runs/hotspot_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_hotspot.yaml runs/hotspot_prod
```

---

### crossreactive_marco_nanobody_setC_hybrid.yaml

**Strategy:** SET C — Hybrid cross-reactive interface. Combines the SO₄/pocket residues (Set A) with the patent antibody epitope (Set B) for maximum paratope breadth. The widest functional surface of any Set C spec.

**Binding residues:**
- Human (Q9UEW3): 429, 431, 438, 450, 452, 467, 472, 473, 475, 487, 495, 499, 505, 506, 507, 509, 511
- Mouse (2OYA): 12, 14, 21, 50, 56, 58, 78, 89

**Scaffolds:** 7EOW (21-aa CDR3, recommended for large interface) + 8COH (19-aa CDR3)

**When to use:** Highest-priority cross-reactive spec. Maximizes interface contacts by targeting both the ligand pocket and the patent epitope simultaneously. Best for therapeutic applications requiring broad interference with MARCO function.

**Validate:**
```bash
boltzgen check specs/crossreactive_marco_nanobody_setC_hybrid.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_marco_nanobody_setC_hybrid.yaml runs/setC_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setC_hybrid.yaml runs/setC_prod
```

> **Note:** The original patent-based position claims were verified against mmCIF structures. Several patent-reported amino acid identities were corrected (e.g., Q452 = TYR, not the patent's claimed residue).

---

### crossreactive_marco_nanobody_setD_beta_pairing.yaml

**Strategy:** SET D — Beta-pairing cross-reactive. Targets exposed beta-strand edges on the SRCR fold to favor backbone-like, hydrogen-bond-rich VHH contacts on the polar beta-sheet surface. Distinct from Sets A/B/C which target the crevice/pocket.

**Binding residues:**
- Human (Q9UEW3): 423, 424, 426, 428, 430, 431, 433, 436, 437, 438, 466, 468, 470, 514, 517, 518
- Mouse (2OYA): 7, 8, 10, 12, 14, 15, 17, 20, 21, 22, 50, 52, 54, 98, 101, 102

**Scaffolds:** 7EOW (21-aa CDR3) + 8COH (19-aa CDR3)

**When to use:** Highest-priority exploratory set for the hydrophilic SRCR surface. Best when generic hydrophobic binder filtering may underperform. Distinct strategy from Set A/B/C — designed to complement, not compete with, crevice-targeting designs.

**Validate:**
```bash
boltzgen check specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_prod
```

---

### human_marco_nanobody_anywhere.yaml

**Strategy:** Human MARCO only. No hotspot constraint — designs VHH CDRs against any accessible surface of the human SRCR domain (Q9UEW3 positions 418–520). Unconstrained exploration for novel epitopes.

**Target:** Human MARCO SRCR (Q9UEW3, chain A), full SRCR domain

**Scaffolds:** 7EOW (21-aa CDR3) + 8COH (19-aa CDR3)

**When to use:** Discovery-mode exploration without prior epitope assumptions. May find novel sites not covered by Sets A–D. Higher risk of off-target binding — always validate top designs with AF2 backfolding.

**Validate:**
```bash
boltzgen check specs/human_marco_nanobody_anywhere.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_nanobody_anywhere.yaml runs/human_anywhere_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_anywhere.yaml runs/human_anywhere_prod
```

---

### human_marco_nanobody_hotspot.yaml

**Strategy:** Human MARCO only. Targets the conserved ARG-rich basic patch (Q9UEW3 positions 423, 425, 432, 461, 467, 469, 489, 500) — the functional ligand-binding surface identified from SO₄ binding site analysis.

**Target:** Human MARCO SRCR (Q9UEW3, chain A)

**Binding residues:** 423, 425, 432, 461, 467, 469, 489, 500

**Scaffolds:** 7EOW (21-aa CDR3) + 8COH (19-aa CDR3)

**When to use:** Human-specific screening where cross-reactivity is not required. The hotspot set is the same functional surface as the mouse DSSP-derived hotspots.

**Validate:**
```bash
boltzgen check specs/human_marco_nanobody_hotspot.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml runs/human_hotspot_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml runs/human_hotspot_prod
```

---

### human_marco_nanobody_setA_so4_pocket.yaml

**Strategy:** SET A — Human MARCO SO₄/pocket blocking. Targets the ligand-binding crevice residues (Q9UEW3 positions 429, 431, 438, 467, 473, 475, 495, 506) to block LDL, oxLDL, bacteria, and apoptotic cell binding.

**Target:** Human MARCO SRCR (Q9UEW3, chain A)

**Binding residues:** 429, 431, 438, 467, 473, 475, 495, 506

**Scaffolds:** 7EOW (21-aa CDR3, recommended for deep-pocket reach) + 8COH (19-aa CDR3)

**When to use:** Ligand-blocking therapeutic applications. Specifically blocks the ligand-binding crevice rather than the patent epitope (Set B) or beta-edge (Set D).

**Validate:**
```bash
boltzgen check specs/human_marco_nanobody_setA_so4_pocket.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_nanobody_setA_so4_pocket.yaml runs/setA_human_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setA_so4_pocket.yaml runs/setA_human_prod
```

---

### human_marco_nanobody_setB_patent_epitope.yaml

**Strategy:** SET B — Human MARCO patent antibody epitope. Targets the PI-3010/PI-3035 antibody epitope (Q9UEW3 positions 450, 452, 472, 473, 487, 499, 505, 507, 509, 511) — a functional epitope distinct from the SO₄ pocket.

**Target:** Human MARCO SRCR (Q9UEW3, chain A) — human-only

**Binding residues:** 450, 452, 472, 473, 487, 499, 505, 507, 509, 511

**Scaffolds:** 7EOW (21-aa CDR3) + 8COH (19-aa CDR3)

**When to use:** Targeting the same epitope space as the PI-3010/PI-3035 antibody path. Human-only due to species-specific residues (e.g., Q452 = mouse D452). Not suitable for cross-reactive designs.

> ⚠️ **Patent accuracy note:** The patent text contained amino acid claims that did not match the actual mmCIF structure. All positions were re-verified against `human_MARCO_input.cif`.

**Validate:**
```bash
boltzgen check specs/human_marco_nanobody_setB_patent_epitope.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_nanobody_setB_patent_epitope.yaml runs/setB_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setB_patent_epitope.yaml runs/setB_prod
```

---

### human_marco_nanobody_setD_beta_pairing.yaml

**Strategy:** SET D — Human MARCO beta-pairing SRCR edge-strand targeting. Targets exposed beta-strand edges on the human SRCR beta-sheet face (Q9UEW3 positions 423, 424, 426, 428, 430, 431, 433, 436, 437, 438, 466, 468, 470, 514, 517, 518) to favor backbone-like H-bond contacts.

**Target:** Human MARCO SRCR (Q9UEW3, chain A)

**Binding residues:** 423, 424, 426, 428, 430, 431, 433, 436, 437, 438, 466, 468, 470, 514, 517, 518

**Scaffolds:** 7EOW (21-aa CDR3) + 8COH (19-aa CDR3)

**When to use:** Human-only version of the Set D beta-pairing strategy. Use when cross-reactivity is not required and you want to specifically target the polar beta-sheet face rather than the crevice (Set A) or epitope (Set B).

**Validate:**
```bash
boltzgen check specs/human_marco_nanobody_setD_beta_pairing.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_nanobody_setD_beta_pairing.yaml runs/setD_human_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setD_beta_pairing.yaml runs/setD_human_prod
```

---

### mouse_marco_nanobody_anywhere.yaml

**Strategy:** Mouse Marco only. No hotspot constraint — designs VHH CDRs against any accessible surface of the mouse SRCR domain (2OYA, 102 aa). Unconstrained exploration.

**Target:** Mouse Marco SRCR (PDB 2OYA, chain A)

**Scaffolds:** 7EOW (21-aa CDR3) + 8COH (19-aa CDR3)

**When to use:** Discovery-mode exploration of the mouse Marco surface. Useful for in vivo mouse model applications where cross-reactivity is not needed.

**Validate:**
```bash
boltzgen check specs/mouse_marco_nanobody_anywhere.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/mouse_marco_nanobody_anywhere.yaml runs/mouse_anywhere_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_anywhere.yaml runs/mouse_anywhere_prod
```

---

### mouse_marco_nanobody_hotspot.yaml

**Strategy:** Mouse Marco only. Targets the conserved ARG-rich basic patch (2OYA label_seq 6, 8, 15, 44, 50, 52, 72, 83) identified from SO₄ binding sites AC1/AC2/AC3. These are the verified ligand-binding residues.

**Target:** Mouse Marco SRCR (PDB 2OYA, chain A)

**Binding residues:** 6, 8, 15, 44, 50, 52, 72, 83 (2OYA label_seq)

**Scaffolds:** 7EOW (21-aa CDR3, recommended for deep-pocket reach) + 8COH (19-aa CDR3)

**When to use:** Mouse-specific screening. The original DSSP-derived hotspot spec that informed Sets A–D. Good for understanding which positions drive the strongest H-bond interface.

**Validate:**
```bash
boltzgen check specs/mouse_marco_nanobody_hotspot.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_hotspot_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_hotspot_prod
```

---

### mouse_marco_nanobody_setA_so4_pocket.yaml

**Strategy:** SET A — Mouse Marco SO₄/pocket blocking. Targets the ligand-binding crevice (2OYA label_seq 12, 14, 21, 50, 56, 58, 78, 89) to block ligand (LDL, oxLDL, bacteria, apoptotic cells) binding.

**Target:** Mouse Marco SRCR (PDB 2OYA, chain A)

**Binding residues:** 12, 14, 21, 50, 56, 58, 78, 89 (2OYA label_seq)

**Scaffolds:** 7EOW (21-aa CDR3, recommended for deep-pocket reach) + 8COH (19-aa CDR3)

**When to use:** Ligand-blocking applications in mouse systems. The mouse counterpart to `human_marco_nanobody_setA_so4_pocket.yaml`.

> **Note:** This uses a different residue set than `mouse_marco_nanobody_hotspot.yaml` (which uses 6,8,15,44,50,52,72,83). Set A targets the crevice entrance; hotspot targets the deep SO₄ site.

**Validate:**
```bash
boltzgen check specs/mouse_marco_nanobody_setA_so4_pocket.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/mouse_marco_nanobody_setA_so4_pocket.yaml runs/setA_mouse_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_setA_so4_pocket.yaml runs/setA_mouse_prod
```

---

### mouse_marco_nanobody_setD_beta_pairing.yaml

**Strategy:** SET D — Mouse Marco beta-pairing SRCR edge-strand targeting. Targets exposed beta-strand edges (2OYA label_seq 7, 8, 10, 12, 14, 15, 17, 20, 21, 22, 50, 52, 54, 98, 101, 102) to favor backbone-like H-bond contacts on the polar beta-sheet.

**Target:** Mouse Marco SRCR (PDB 2OYA, chain A)

**Binding residues:** 7, 8, 10, 12, 14, 15, 17, 20, 21, 22, 50, 52, 54, 98, 101, 102 (2OYA label_seq)

**Scaffolds:** 7EOW (21-aa CDR3) + 8COH (19-aa CDR3)

**When to use:** Mouse-only beta-edge exploration. Use when cross-reactivity is not needed and the beta-sheet face is the target.

**Validate:**
```bash
boltzgen check specs/mouse_marco_nanobody_setD_beta_pairing.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/mouse_marco_nanobody_setD_beta_pairing.yaml runs/setD_mouse_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_setD_beta_pairing.yaml runs/setD_mouse_prod
```

---

### human_marco_binder_anywhere.yaml

**Strategy:** General binder scaffolds (not VHH-specific) against human MARCO with no hotspot constraint. Explores any accessible surface. Less CDR geometry bias than `*_nanobody_anywhere`.

**Target:** Human MARCO SRCR (Q9UEW3, chain A)

**Scaffolds:** 7EOW (21-aa CDR3) + 8COH (19-aa CDR3) — same scaffolds as nanobody specs but without `--protocol nanobody-anything` specificity

**When to use:** General binder discovery when VHH-specific constraints are not required. Consider `*_nanobody_anywhere` instead for tighter CDR geometry.

> ⚠️ **Chain ID:** This spec has `# TODO: replace with target chain` comment. Verify the chain ID in `human_MARCO_input.cif` before running.

**Validate:**
```bash
boltzgen check specs/human_marco_binder_anywhere.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_binder_anywhere.yaml runs/human_binder_anywhere_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_binder_anywhere.yaml runs/human_binder_anywhere_prod
```

---

### mouse_marco_binder_anywhere.yaml

**Strategy:** General binder scaffolds against mouse Marco with no hotspot constraint. Explores any accessible surface.

**Target:** Mouse Marco SRCR (PDB 2OYA, chain A)

**Scaffolds:** 7EOW (21-aa CDR3) + 8COH (19-aa CDR3)

**When to use:** General binder discovery in mouse systems. Consider `mouse_marco_nanobody_anywhere.yaml` for VHH-specific CDR constraints.

> ⚠️ **Chain ID:** Verify chain ID in `mouse_marco_srcr.cif` before running (default is A).

**Validate:**
```bash
boltzgen check specs/mouse_marco_binder_anywhere.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/mouse_marco_binder_anywhere.yaml runs/mouse_binder_anywhere_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_binder_anywhere.yaml runs/mouse_binder_anywhere_prod
```

---

### human_marco_binder_hotspot.yaml

**Strategy:** General binder scaffolds targeting the conserved ARG-rich basic patch on human MARCO (Q9UEW3 positions 423, 425, 432, 461, 467, 469, 489, 500).

**Target:** Human MARCO SRCR (Q9UEW3, chain A)

**Binding residues:** 423, 425, 432, 461, 467, 469, 489, 500

**Scaffolds:** 7EOW (21-aa CDR3) + 8COH (19-aa CDR3)

**When to use:** Human-only hotspot targeting with general binder scaffolds. Use `human_marco_nanobody_hotspot.yaml` for VHH-specific CDR constraints.

**Validate:**
```bash
boltzgen check specs/human_marco_binder_hotspot.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_binder_hotspot.yaml runs/human_binder_hotspot_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_binder_hotspot.yaml runs/human_binder_hotspot_prod
```

---

### mouse_marco_binder_hotspot.yaml

**Strategy:** General binder scaffolds targeting the conserved ARG-rich basic patch on mouse Marco (2OYA label_seq 6, 8, 15, 44, 50, 52, 72, 83).

**Target:** Mouse Marco SRCR (PDB 2OYA, chain A)

**Binding residues:** 6, 8, 15, 44, 50, 52, 72, 83 (2OYA label_seq)

**Scaffolds:** 7EOW (21-aa CDR3) + 8COH (19-aa CDR3)

**When to use:** Mouse-only hotspot targeting with general binder scaffolds. Use `mouse_marco_nanobody_hotspot.yaml` for VHH-specific CDR constraints.

**Validate:**
```bash
boltzgen check specs/mouse_marco_binder_hotspot.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/mouse_marco_binder_hotspot.yaml runs/mouse_binder_hotspot_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_binder_hotspot.yaml runs/mouse_binder_hotspot_prod
```

---

### human_marco_peptide_anywhere.yaml

**Strategy:** Peptide scaffold design against human MARCO with no hotspot constraint. Uses peptide-length scaffolds rather than full nanobody/binder scaffolds.

**Target:** Human MARCO SRCR (Q9UEW3, chain A)

**Scaffolds:** 7EOW + 8COH (peptide-length configuration)

**When to use:** Peptide-based blocking agents. Less common than nanobody designs; consider `human_marco_nanobody_anywhere.yaml` for VHH-format alternatives.

> ⚠️ **Chain ID:** Verify chain ID in `human_MARCO_input.cif` before running.

**Validate:**
```bash
boltzgen check specs/human_marco_peptide_anywhere.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/human_marco_peptide_anywhere.yaml runs/human_peptide_anywhere_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_peptide_anywhere.yaml runs/human_peptide_anywhere_prod
```

---

### mouse_marco_peptide_anywhere.yaml

**Strategy:** Peptide scaffold design against mouse Marco with no hotspot constraint.

**Target:** Mouse Marco SRCR (PDB 2OYA, chain A)

**Scaffolds:** 7EOW + 8COH (peptide-length configuration)

**When to use:** Peptide-based blocking agents in mouse systems. Consider `mouse_marco_nanobody_anywhere.yaml` for VHH-format alternatives.

**Validate:**
```bash
boltzgen check specs/mouse_marco_peptide_anywhere.yaml
```

**Local pilot:**
```bash
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/mouse_marco_peptide_anywhere.yaml runs/mouse_peptide_anywhere_pilot
```

**HPC production:**
```bash
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_peptide_anywhere.yaml runs/mouse_peptide_anywhere_prod
```

---

### Spec quick-reference table

| Spec | Species | Strategy | Hotspots | Scaffolds | Priority |
|------|---------|----------|----------|-----------|----------|
| `crossreactive_marco_nanobody_setD_beta_pairing.yaml` | Both | Beta-edge targeting | 16/16 | VHH | 🔴 Highest |
| `crossreactive_marco_nanobody_setC_hybrid.yaml` | Both | Hybrid (A+B union) | 17/8 | VHH | 🔴 Highest |
| `mouse_marco_nanobody_setA_so4_pocket.yaml` | Mouse | SO₄/pocket blocking | 8 | VHH | 🟡 High |
| `human_marco_nanobody_setA_so4_pocket.yaml` | Human | SO₄/pocket blocking | 8 | VHH | 🟡 High |
| `human_marco_nanobody_setB_patent_epitope.yaml` | Human | Patent epitope | 10 | VHH | 🟡 High |
| `human_marco_nanobody_setD_beta_pairing.yaml` | Human | Beta-edge targeting | 16 | VHH | 🟡 High |
| `mouse_marco_nanobody_setD_beta_pairing.yaml` | Mouse | Beta-edge targeting | 16 | VHH | 🟡 High |
| `crossreactive_marco_nanobody_hotspot.yaml` | Both | ARG-rich basic patch | 8/8 | VHH | 🟢 Medium |
| `crossreactive_conserved_surface.yaml` | Both | ARG-rich basic patch | 8/8 | VHH | 🟢 Medium |
| `human_marco_nanobody_hotspot.yaml` | Human | ARG-rich basic patch | 8 | VHH | 🟢 Medium |
| `mouse_marco_nanobody_hotspot.yaml` | Mouse | ARG-rich basic patch | 8 | VHH | 🟢 Medium |
| `human_marco_nanobody_anywhere.yaml` | Human | Unconstrained | None | VHH | 🔵 Exploratory |
| `mouse_marco_nanobody_anywhere.yaml` | Mouse | Unconstrained | None | VHH | 🔵 Exploratory |
| `human_marco_binder_hotspot.yaml` | Human | Hotspot (general) | 8 | Binder | 🟢 Medium |
| `mouse_marco_binder_hotspot.yaml` | Mouse | Hotspot (general) | 8 | Binder | 🟢 Medium |
| `human_marco_binder_anywhere.yaml` | Human | Unconstrained (general) | None | Binder | 🔵 Exploratory |
| `mouse_marco_binder_anywhere.yaml` | Mouse | Unconstrained (general) | None | Binder | 🔵 Exploratory |
| `human_marco_peptide_anywhere.yaml` | Human | Unconstrained (peptide) | None | Peptide | 🔵 Exploratory |
| `mouse_marco_peptide_anywhere.yaml` | Mouse | Unconstrained (peptide) | None | Peptide | 🔵 Exploratory |

---

---

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
