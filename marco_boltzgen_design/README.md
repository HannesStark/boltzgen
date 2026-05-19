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
│   ├── mouse_marco_nanobody_setA_so4_pocket.yaml   # SET A — ligand-blocking pocket
│   ├── human_marco_nanobody_setA_so4_pocket.yaml
│   ├── human_marco_nanobody_setB_patent_epitope.yaml # SET B — patent epitope
│   ├── crossreactive_marco_nanobody_setC_hybrid.yaml # SET C — hybrid (A+B union)
│   ├── mouse_marco_nanobody_setD_beta_pairing.yaml  # SET D — beta-edge strand
│   ├── human_marco_nanobody_setD_beta_pairing.yaml
│   └── crossreactive_marco_nanobody_setD_beta_pairing.yaml
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

Four interface sets target distinct structural features of the MARCO SRCR domain. **All sets use `--protocol nanobody-anything`** and the MARCO-specific filtering defaults.

### Set A — SO4 / Ligand-Blocking Pocket
**Block the ligand-binding crevice** (LDL, oxLDL, bacteria, apoptotic cells).

| Spec | Species | label_seq | Notes |
|------|---------|-----------|-------|
| `mouse_marco_nanobody_setA_so4_pocket.yaml` | Mouse | 12,14,21,50,56,58,78,89 | 2OYA label_seq |
| `human_marco_nanobody_setA_so4_pocket.yaml` | Human | 12,14,21,50,56,58,78,89 | Q9UEW3 positions |

### Set B — Patent Antibody Epitope
**Target the PI-3010/PI-3035 antibody epitope** (human-only; cross-species conservation differs).

| Spec | label_seq | Notes |
|------|-----------|-------|
| `human_marco_nanobody_setB_patent_epitope.yaml` | 33,35,56,70,82,88,90,92,94 | Human only |

### Set C — Hybrid Interface (Cross-Reactive)
**Combine Sets A + B** for maximum paratope breadth and interface stability.

| Spec | Human label_seq | Mouse label_seq | Notes |
|------|----------------|-----------------|-------|
| `crossreactive_marco_nanobody_setC_hybrid.yaml` | 35,50,56,58,70,78,82,88,89,90,92,94 | 12,14,21,50,56,58,78,89 | Cross-reactive |

### Set D — Beta-Pairing SRCR Edge-Strand Targeting
**Target exposed beta-strand edges** for backbone-like, hydrogen-bond-rich VHH contacts. Targets a distinct face from the ligand crevice — recommended as the primary exploratory set for the polar SRCR surface.

| Spec | Human label_seq | Mouse label_seq | Notes |
|------|----------------|-----------------|-------|
| `mouse_marco_nanobody_setD_beta_pairing.yaml` | — | 7,8,10,12,14,15,17,20,21,22,50,52,54,98,101,102 | 2OYA beta-edge |
| `human_marco_nanobody_setD_beta_pairing.yaml` | 423,424,426,428,430,431,433,436,437,438,466,468,470,514,517,518 | — | Human beta-edge |
| `crossreactive_marco_nanobody_setD_beta_pairing.yaml` | 423,424,426,428,430,431,433,436,437,438,466,468,470,514,517,518 | 7,8,10,12,14,15,17,20,21,22,50,52,54,98,101,102 | Cross-reactive |

**Key conserved residue: 473** (K in both human and mouse) appears in Sets A–C.

---

## 5. Pipeline Overview

```
Step 1          Step 2          Step 3           Step 4          Step 5         Step 6
Validate  ───▶  Local     ───▶  HPC Production ──▶  Collect  ───▶  Rank &   ───▶  AF2
Specs          Pilot             (2000 designs)        Metrics     Filter          Validate
                    50 designs   sbatch × N            merge CSV   top-50 by       backfold
                                                                       score         top-50
BoltzGen pipeline steps (within each campaign):
  [1] design → [2] inverse_folding → [3] folding → [4] analysis → [5] filtering
```

**Total time estimates (RTX 5000 × 2):**
- 2000 designs, BUDGET=200: ~2–4 hours per campaign
- AF2 validation (top-50): ~1–2 hours

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
```

**Expected output:** `Total designed residues: 26–46` and a `.cif` visualization file. If you see errors about missing residue IDs, the `binding:` positions in the YAML do not match the mmCIF.

---

## Step 2 — Local Pilot

Run a small batch locally to verify the pipeline works before submitting an HPC job. Use the `run_nanobody_campaign.sh` wrapper, which applies all MARCO defaults automatically:

```bash
# Set D cross-reactive pilot (recommended first run)
NUM_DESIGNS=50 BUDGET=10 \
  ./runs/run_nanobody_campaign.sh \
    specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml \
    runs/setD_pilot
```

Or run `boltzgen` directly with explicit MARCO flags:

```bash
boltzgen run specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml \
  --protocol nanobody-anything \
  --output runs/setD_pilot \
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

# Run all in parallel
wait
```

### 3.2 Dual-GPU Job (2 × RTX 5000, 16 GB each)

The HPC script reserves 2 GPUs by default. Submit two single-GPU jobs simultaneously to fill both cards:

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
  --runs runs/setA_mouse runs/setA_human runs/setB runs/setC runs/setD \
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

# 3. HPC production (submit all sets in parallel)
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD &

NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setC_hybrid.yaml runs/setC &

NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_setA_so4_pocket.yaml runs/setA_mouse &

NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setA_so4_pocket.yaml runs/setA_human &

NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setB_patent_epitope.yaml runs/setB &

wait   # wait for all SLURM jobs to finish

# 4. Collect & rank
./scripts/collect_campaign.sh \
  --runs runs/setA_mouse runs/setA_human runs/setB runs/setC runs/setD \
  --out results/all_metrics.csv

./scripts/rank_and_validate.sh \
  --metrics results/all_metrics.csv \
  --human-conserved "A:35,A:50,A:56,A:58,A:70,A:78,A:82,A:88,A:89,A:90,A:92,A:94" \
  --mouse-conserved "A:12,A:14,A:21,A:50,A:56,A:58,A:78,A:89" \
  --top_n 50

# 5. AF2 validation
python scripts/validate_designs.py \
  --complexes results/candidate_cifs \
  --metrics results/ranked_candidates.csv \
  --top_n 50 \
  --method colabfold \
  --out results/af_validation.csv
```

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

### Appendix E: Key Files for Downstream Use

After the full pipeline, these are the key outputs:
- `results/ranked_candidates.csv` — all ranked designs with scores
- `results/af_validation.csv` — AF2 backfold validation results
- `results/candidate_cifs/` — top-50 design CIF files for experimental testing
- `runs/<name>/final_ranked_designs/` — full design outputs per campaign