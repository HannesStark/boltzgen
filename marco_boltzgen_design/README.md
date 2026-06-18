# MARCO Nanobody Design with BoltzGen

De novo VHH (nanobody) design against the SRCR domain of **human MARCO** (UniProt Q9UEW3) and **mouse Marco** (PDB 2OYA), powered by [BoltzGen](https://github.com/jxshi/boltzgen) and aligned with the **BoltzProt-1 Technical Report** (arXiv:2512.00000).

---

## Table of Contents

1. [Quick-Start](#1-quick-start)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Step 1 — Clone & Setup](#step-1--clone--setup)
4. [Step 2 — Validate Specs](#step-2--validate-specs)
5. [Step 3 — Local Pilot](#step-3--local-pilot)
6. [Step 4 — HPC Production](#step-4--hpc-production)
7. [Step 5 — Collect & Merge Metrics](#step-5--collect--merge-metrics)
8. [Step 6 — Rank & Filter](#step-6--rank--filter)
9. [Step 7 — CDR3 Novelty Check](#step-7--cdr3-novelty-check)
10. [Step 8 — AF2 Validation](#step-8--af2-validation)
11. [Interface Strategy Sets](#interface-strategy-sets)
12. [Key Scripts Reference](#key-scripts-reference)
13. [BoltzProt-1 Developability Flags](#boltzprot-1-developability-flags)
14. [SAbDab Novelty Cache](#sabdab-novelty-cache)
15. [Troubleshooting](#troubleshooting)

---

## 1. Quick-Start

```bash
# 1. Clone
git clone https://github.com/jxshi/boltzgen.git
cd boltzgen/marco_boltzgen_design

# 2. Validate
conda activate boltzgen
boltzgen check specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml

# 3. Local pilot (50 designs)
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/pilot

# 4. HPC production (60,000 designs)
NUM_DESIGNS=60000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_prod
```

---

## 2. Pipeline Overview

```
Stage 0          Stage 1          Stage 2          Stage 3          Stage 4          Stage 5
─────────        ─────────        ─────────        ─────────        ─────────        ─────────
Validate   →     HPC Design  →    Collect &   →    Rank &      →    Novelty   →    AF2
(specs)         (SLURM)           Merge            Filter          Check          Validation
```

| Stage | What | Where | Output |
|-------|------|-------|--------|
| **0 — Validate** | Spec check, target check | Local | PASS/FAIL |
| **1 — Design** | `boltzgen run` 5-step pipeline on SLURM | HPC (GPU) | CIF + NPZ in `runs/<name>/` |
| **2 — Collect** | Gather outputs from HPC, merge metrics | Local | `results/all_metrics.csv` |
| **3 — Rank** | `rank_designs.py` + developability filters | Local | `results/ranked_candidates.csv` |
| **4 — Novelty** | `novelty_check.py` — CDR3 edit distance ≥ 4 from SAbDab | Local | `results/novelty_checked.csv` |
| **5 — Validate** | `validate_designs.py` (AF2 backfold) | Local/HPC | `results/af_validation.csv` |

**Automatic post-processing:** Both `run_hpc_campaign.sh` and `run_nanobody_campaign.sh` automatically apply the **N-glycosylation sequon filter** (`filter_nglyc.py`) after generation, removing any design containing NXS/T motifs before ranking.

---

## Step 1 — Clone & Setup

```bash
git clone https://github.com/jxshi/boltzgen.git
cd boltzgen/marco_boltzgen_design
conda activate boltzgen
boltzgen --version
```

**First time on HPC only** — download BoltzGen models:

```bash
boltzgen run --force_download specs/mouse_marco_nanobody_setD_beta_pairing.yaml \
  --output /tmp/test_model_dl --num_designs 1 --budget 1
```

---

## Step 2 — Validate Specs

Always validate before submitting HPC jobs:

```bash
# Recommended: start with Set D (beta-pairing, highest priority)
boltzgen check specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml

# Validate all priority specs
boltzgen check specs/mouse_marco_nanobody_setD_beta_pairing.yaml
boltzgen check specs/human_marco_nanobody_setD_beta_pairing.yaml
boltzgen check specs/crossreactive_marco_nanobody_setC_hybrid.yaml
boltzgen check specs/mouse_marco_nanobody_setA_so4_pocket.yaml
boltzgen check specs/human_marco_nanobody_setA_so4_pocket.yaml
boltzgen check specs/human_marco_nanobody_setB_patent_epitope.yaml
```

---

## Step 3 — Local Pilot

Run a small batch locally to verify the pipeline before submitting HPC jobs.

```bash
# Set D cross-reactive pilot (recommended first run)
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/pilot

# Set C hybrid pilot (maximum interface breadth)
NUM_DESIGNS=50 BUDGET=10 ./runs/run_nanobody_campaign.sh \
  specs/crossreactive_marco_nanobody_setC_hybrid.yaml runs/setC_pilot
```

> ⚠️ **Local Mac OOM risk:** Use `NUM_DESIGNS ≤ 100` locally. Production runs must go to HPC.

**Output:** `runs/<name>/final_ranked_designs/all_designs_metrics.csv`

---

## Step 4 — HPC Production

### Standard quality mode

```bash
# Single spec — 60,000 designs (BoltzProt-1 production standard)
NUM_DESIGNS=60000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml \
  runs/setD_beta_pairing
```

### Speed mode (2–4× faster, for large batches / screening)

```bash
SPEED_MODE=1 NUM_DESIGNS=60000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml \
  runs/setD_fast
```

Speed mode applies: `sampling_steps=100`, `recycling_steps=1`, `diffusion_samples=1`, `compile_pairformer=true`, `compile_structure=true`, `inverse_fold precision=bf16-mixed`, `diffusion_batch_size=8`.

### Multi-spec campaign (submit all in parallel)

```bash
# ── Set A ──
NUM_DESIGNS=60000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_setA_so4_pocket.yaml runs/setA_mouse &
NUM_DESIGNS=60000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setA_so4_pocket.yaml runs/setA_human &

# ── Set B ──
NUM_DESIGNS=60000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setB_patent_epitope.yaml runs/setB &

# ── Set C ──
NUM_DESIGNS=60000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setC_hybrid.yaml runs/setC &

# ── Set D ──
NUM_DESIGNS=60000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD &
wait
```

### Key environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `NUM_DESIGNS` | 60000 | Designs per HPC job |
| `BUDGET` | 150 | Inference steps per design (higher = better quality) |
| `GPUS` | 2 | GPUs per job (RTX 5000 x2) |
| `SPEED_MODE` | 0 | Set to 1 for fast screening mode |
| `EXCLUDE_NGLYC` | 1 | Auto-filter N-glyc sequons post-generation |

**BUDGET on RTX 5000:** `100–150` is the sweet spot. `200` may exceed 96-hour time limit for 60k designs.

### Monitor SLURM jobs

```bash
squeue -u $USER
tail -f logs/boltzgen_<spec>_<JOB_ID>.out
ls runs/<name>/intermediate_designs/*.cif 2>/dev/null | wc -l   # count done so far
```

---

## Step 5 — Collect & Merge Metrics

After all HPC jobs finish, copy results to local machine and merge:

```bash
# Copy from HPC to local
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/ runs/

# Merge all metrics CSVs
./scripts/collect_campaign.sh \
  --runs runs/setA_mouse runs/setA_human runs/setB runs/setC runs/setD \
  --out results/all_metrics.csv
```

---

## Step 6 — Rank & Filter

Rank by confidence, developability, and epitope coverage:

```bash
# Set D / Set C / Set A (cross-reactive or mouse)
python scripts/rank_designs.py \
  --metrics results/all_metrics.csv \
  --human-conserved "A:423,A:424,A:431,A:460,A:466,A:468,A:488,A:499" \
  --mouse-conserved "A:6,A:8,A:15,A:44,A:50,A:52,A:72,A:83" \
  --max-len 120 \
  --out results/ranked_candidates.csv

# Set B (human-only patent epitope)
python scripts/rank_designs.py \
  --metrics results/all_metrics.csv \
  --human-conserved "A:33,A:35,A:56,A:70,A:82,A:88,A:90,A:92,A:94" \
  --max-len 120 \
  --out results/ranked_candidates.csv
```

What this does:
- Sorts by `final_score = base_confidence + 0.5 × crossreactivity_score − developability_penalties`
- Applies 9 developability flag columns (Cys, length, charge, hydrophobicity, aromatic fraction, pI region, Pro in CDR3, N-glyc motifs)
- Auto-detects contact columns in metrics CSV for cross-reactivity scoring

**Inspect top candidates:**

```python
import pandas as pd
df = pd.read_csv('results/ranked_candidates.csv')
print(df[['design_id','final_score','developability_flags','binder_sequence']].head(20))
```

---

## Step 7 — CDR3 Novelty Check

Flags any design whose CDR3 is **edit-distance < 4** from a known SAbDab antibody. Per BoltzProt-1: *"every recovered design has a minimum CDR3 edit distance of at least four to its closest SAbDab match."*

The pre-built cache (4,466 unique CDR3s from 32k SAbDab PDBs) is committed to the repo at `.sabdab_reference.json` — no rebuild needed on most machines.

```bash
# First time: build the cache from local SAbDab zip
# (only needed if .sabdab_reference.json is missing or stale)
python scripts/novelty_check.py \
  --build_cache \
  --sabdab_zip ~/Downloads/all_structures.zip

# Check designs against the reference
python scripts/novelty_check.py \
  --designs results/ranked_candidates.csv \
  --out results/novelty_checked.csv
```

**Output columns:**

| Column | Meaning |
|--------|---------|
| `cdr3_edit_distance` | Min edit distance of CDR3 to any SAbDab entry |
| `cdrs_edit_distance` | Min edit distance of CDR1+CDR2+CDR3 combined |
| `novelty_flag` | `low_novelty` if edit distance < 4 |

**Filter to novel designs only:**

```python
import pandas as pd
df = pd.read_csv('results/novelty_checked.csv')
novel = df[df['novelty_flag'] != 'low_novelty']
novel.to_csv('results/novel_candidates.csv', index=False)
print(f"Novel candidates: {len(novel)} / {len(df)}")
```

---

## Step 8 — AF2 Validation

Validates that designed binder sequences back-fold correctly to the predicted structures:

```bash
python scripts/validate_designs.py \
  --complexes results/candidate_cifs \
  --metrics results/novel_candidates.csv \
  --top_n 50 \
  --method colabfold \
  --out results/af_validation.csv
```

**Thresholds:** CA RMSD < 2.5 Å **AND** mean PAE < 5.0 Å

**Merge AF2 results:**

```python
import pandas as pd
ranked = pd.read_csv('results/novel_candidates.csv')
af2 = pd.read_csv('results/af_validation.csv')
merged = ranked.merge(af2[['design_id','af2_rmsd','af2_pae','af2_plddt','flag_ok']], on='design_id', how='left')
passing = merged[merged['flag_ok'] == True]
print(f"AF2-passing designs: {len(passing)} / {len(merged)}")
print(passing[['design_id','plddt','final_score','af2_rmsd','af2_pae']].head(20))
```

**Designs passing AF2 are ready for experimental characterization.**

---

## Interface Strategy Sets

Four strategy groups cover distinct SRCR surfaces. **Set D** and **Set C** are the highest priority.

| Set | Strategy | Specs | Priority |
|-----|----------|-------|----------|
| **D** | Beta-edge strand targeting — polar beta-sheet face | `*_setD_beta_pairing.yaml` | 🔴 Highest |
| **C** | Hybrid interface — Sets A + B union | `*_setC_hybrid.yaml` | 🔴 Highest |
| **A** | SO₄/pocket blocking — ligand-binding crevice | `*_setA_so4_pocket.yaml` | 🟡 High |
| **B** | Patent antibody epitope — human-only | `*_setB_patent_epitope.yaml` | 🟡 High |
| Hotspot | ARG-rich basic patch (conserved) | `*_hotspot.yaml` | 🟢 Medium |
| Anywhere | Unconstrained surface exploration | `*_anywhere.yaml` | 🔵 Exploratory |

**Set D — Beta-Pairing (highest priority):**

| Species | Residues | Notes |
|---------|----------|-------|
| Mouse | 7,8,10,12,14,15,17,20,21,22,50,52,54,98,101,102 | 2OYA label_seq |
| Human | 423,424,426,428,430,431,433,436,437,438,466,468,470,514,517,518 | Q9UEW3 label_seq |

**Set C — Hybrid (cross-reactive, maximum breadth):**

| Species | Residues | Notes |
|---------|----------|-------|
| Mouse | 12,14,21,50,56,58,78,89 | 2OYA label_seq |
| Human | 429,431,438,450,452,467,472,473,475,487,495,499,505,506,507,509,511 | Q9UEW3 label_seq |

**Set A — SO₄/Pocket Blocking:**

| Species | Residues | Notes |
|---------|----------|-------|
| Mouse | 12,14,21,50,56,58,78,89 | 2OYA label_seq |
| Human | 429,431,438,467,473,475,495,506 | Q9UEW3 label_seq |

**Set B — Patent Epitope (human-only):**

| Species | Residues | Notes |
|---------|----------|-------|
| Human | 450,452,472,473,487,499,505,507,509,511 | Q9UEW3 label_seq |

> ⚠️ **Numbering:** Spec files use **mmCIF `label_seq`** (not Q9UEW3 full-sequence positions). Mouse uses 2OYA `label_seq` directly.

---

## Key Scripts Reference

| Script | What it does |
|--------|-------------|
| `scripts/run_hpc_campaign.sh` | SLURM submission — full 5-step pipeline + N-glyc filter |
| `runs/run_nanobody_campaign.sh` | Local campaign runner (Mac/HPC login node) |
| `scripts/filter_nglyc.py` | Remove designs with NXS/T sequons (32 motifs) |
| `scripts/rank_designs.py` | Rank by confidence + developability + epitope coverage |
| `scripts/novelty_check.py` | Check CDR3 edit distance against SAbDab reference |
| `scripts/validate_designs.py` | AF2 backfold validation |
| `scripts/collect_campaign.sh` | Merge metrics from multiple runs |

---

## BoltzProt-1 Developability Flags

The `rank_designs.py` script applies 9 sequence-based developability flags aligned with the BoltzProt-1 six-assay panel:

| Flag | Threshold | Risk |
|------|-----------|------|
| `has_cys` | Any Cys in sequence | Disulfide scrambling / oxidative aggregation |
| `nglyc_motif` | N[^P][ST] pattern present | Glycan heterogeneity during expression |
| `too_long` | Binder length > 120 aa | High-risk for expression |
| `excess_positive_charge` | Net charge > +8 | Self-association / AC-SINS risk |
| `hydrophobic_patch_flag` | Hydrophobic fraction > 0.42 | HIC retention / aggregation |
| `aromatic_high` | Aromatic fraction > 0.14 | Polyspecificity / BVP ELISA risk |
| `pi_acidic` | Net charge < −5 | HIC retention / acidic pI risk |
| `pi_basic` | Net charge > +5 | HIC retention / basic pI risk |
| `proline_cdr3` | Pro in central 30% of sequence | Thermal stability / Tm disruption |

**Penalty weights in final score:** `nglyc_motif` and `proline_cdr3` incur a **2× penalty** (stronger weight); all others incur 1×.

**Developability tiers (from BoltzProt-1):**

| Tier | Score range | Interpretation |
|------|-------------|----------------|
| Tier-1 | 0 penalties | Best — proceed to experimental validation |
| Tier-2 | 1–2 penalties | Acceptable with characterization |
| Screening-Hit | 3–4 penalties | Needs developability screening assays |
| Problematic | 5–6 penalties | High risk — consider redesign |

---

## SAbDab Novelty Cache

The file `.sabdab_reference.json` contains **4,466 unique CDR3 sequences** (length 6–22 aa) extracted from 32,000+ IMGT-renumbered PDBs in the SAbDab archive.

- **Location:** `.sabdab_reference.json` (committed to repo, 277 KB)
- **Rebuild only if:** The SAbDab version changes or the cache is deleted
- **Rebuild command:** `python scripts/novelty_check.py --build_cache --sabdab_zip ~/Downloads/all_structures.zip`

CDR3 is extracted from IMGT positions 105–117 (inclusive, 1-indexed → Python slice `[104:117]`). Verified against 6xul (15-aa CDR3) and 7tlz (22-aa camelid VHH CDR3).

**Novelty threshold:** min edit distance ≥ 4 (BoltzProt-1 standard). This was the minimum distance observed across all BoltzProt-1 confirmed binders.

---

## Troubleshooting

**"ERROR: spec not found"**
→ Run from `boltzgen/marco_boltzgen_design/` directory.

**OOM / CUDA out of memory on HPC**
→ Reduce `NUM_DESIGNS` or set `SPEED_MODE=1` (lowers `diffusion_batch_size` footprint).

**SLURM job killed before completing**
→ Re-run with `--reuse` flag (already set in `run_hpc_campaign.sh`). Already-generated designs are skipped.

**"conda: command not found"**
→ `eval "$(conda shell.bash hook)" && conda activate boltzgen`

**`all_designs_metrics.csv` not found after job finishes**
→ The job was likely killed early. Check `logs/<spec>_<JOB_ID>.out` for the last completed design and re-run with `--reuse`.

**Novelty check very slow**
→ On first run, `novelty_check.py` parses `~/Downloads/all_structures.zip`. This takes ~30–60 seconds. Subsequent runs use `.sabdab_reference.json` cache (~1 second).

---

## Complete End-to-End Workflow

```bash
# ═══════════════════════════════════════════════════════════════
# STAGE 0: Validate
# ═══════════════════════════════════════════════════════════════
conda activate boltzgen
boltzgen check specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml

# ═══════════════════════════════════════════════════════════════
# STAGE 1: HPC production (60k designs per spec)
# ═══════════════════════════════════════════════════════════════
NUM_DESIGNS=60000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_prod

# ═══════════════════════════════════════════════════════════════
# STAGE 2: Copy & merge (run locally after HPC jobs finish)
# ═══════════════════════════════════════════════════════════════
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/setD_prod/ runs/setD_prod/
./scripts/collect_campaign.sh --runs runs/setD_prod --out results/all_metrics.csv

# ═══════════════════════════════════════════════════════════════
# STAGE 3: Rank & filter
# ═══════════════════════════════════════════════════════════════
python scripts/rank_designs.py \
  --metrics results/all_metrics.csv \
  --human-conserved "A:423,A:424,A:431,A:460,A:466,A:468,A:488,A:499" \
  --mouse-conserved "A:6,A:8,A:15,A:44,A:50,A:52,A:72,A:83" \
  --max-len 120 \
  --out results/ranked_candidates.csv

# ═══════════════════════════════════════════════════════════════
# STAGE 4: CDR3 novelty check
# ═══════════════════════════════════════════════════════════════
python scripts/novelty_check.py \
  --designs results/ranked_candidates.csv \
  --out results/novelty_checked.csv

# Filter to novel candidates only
python3 -c "
import pandas as pd
df = pd.read_csv('results/novelty_checked.csv')
novel = df[df['novelty_flag'] != 'low_novelty']
novel.to_csv('results/novel_candidates.csv', index=False)
print(f'Novel candidates: {len(novel)} / {len(df)}')
"

# ═══════════════════════════════════════════════════════════════
# STAGE 5: AF2 backfold validation
# ═══════════════════════════════════════════════════════════════
python scripts/validate_designs.py \
  --complexes results/candidate_cifs \
  --metrics results/novel_candidates.csv \
  --top_n 50 \
  --method colabfold \
  --out results/af_validation.csv
```