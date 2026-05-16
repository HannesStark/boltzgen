# MARCO Nanobody Design — Step-by-Step Instruction
## Goal: Design 10,000+ nanobodies against MARCO (mouse + human + cross-reactive)

**Hardware:** HPC with **2 × NVIDIA RTX 5000** (16 GB GDDR6 each, Ada Lovelace)
**Throughput:** ~2,000–4,000 designs per 72-hour job (depending on BUDGET)
**Time estimate:** 2–3 weeks total

---

## Hardware Notes — RTX 5000

| Spec | Value |
|------|-------|
| Architecture | Ada Lovelace (RTX 40-series) |
| Memory | 16 GB GDDR6 per card |
| SLURM gres | `gpu:2` (both cards on same node) |
| Memory allocation | 96 GB total node system RAM; **16 GB per GPU (GDDR6)** — the actual VRAM constraint |
| CPU threads | 16 (8 per GPU) |
| Time limit recommended | 96 hours (RTX 5000 is slower than A100/H100) |

> 💡 **Tip:** Run **2 specs in parallel** — each uses 1 GPU within the same 96h job,
> doubling throughput per submitted job. Examples below show this pattern.

---

## Before You Start — One-Time HPC Setup

### 1. SSH into HPC and clone the repo

```bash
ssh your_hpc_login_node
git clone https://github.com/jxshi/boltzgen.git
cd boltzgen/marco_boltzgen_design
```

### 2. Set up conda environment (one time only)

```bash
conda env create -f environment.yml  # or:
conda create -n boltzgen -c conda-forge -c nvidia \
  python=3.12 pytorch torchvision pytorch-cuda=12.1 \
  pip biopython pandas numpy
conda activate boltzgen
pip install boltzgen  # or: pip install -e /path/to/boltzgen/src
```

### 3. Verify BoltzGen installation

```bash
conda activate boltzgen
boltzgen --version
```

---

## Stage 0 — Pre-flight Validation

**Run this every time before starting a new campaign.**

On HPC login node:

```bash
cd ~/boltzgen/marco_boltzgen_design

# Validate all 3 spec files
boltzgen check specs/mouse_marco_nanobody_hotspot.yaml
boltzgen check specs/human_marco_nanobody_hotspot.yaml
boltzgen check specs/crossreactive_marco_nanobody_hotspot.yaml

# Confirm all say "PASS" before proceeding
```

Expected output for each:
```
Configuration check PASSED
```

---

## Stage 1 — HPC Design (SLURM, Dual RTX 5000)

### Strategy: Run 2 specs in parallel, one per GPU

With 2 RTX 5000s, submit **one SLURM job that runs 2 specs simultaneously** — each on its own GPU. This gives the best throughput: ~4,000 designs per job run.

```bash
cd ~/boltzgen/marco_boltzgen_design

# GPU 0 + GPU 1 — Mouse + Human in parallel
(NUM_DESIGNS=2000 BUDGET=150 boltzgen run specs/mouse_marco_nanobody_hotspot.yaml \
  --output runs/mouse_vhh_batch1 --protocol nanobody-anything \
  --num_designs 2000 --budget 150 --devices 1 --reuse &

 NUM_DESIGNS=2000 BUDGET=150 boltzgen run specs/human_marco_nanobody_hotspot.yaml \
  --output runs/human_vhh_batch1 --protocol nanobody-anything \
  --num_designs 2000 --budget 150 --devices 1 --reuse &

 wait) 2>&1 | tee logs/batch1_mouse_human.log
```

But a cleaner approach is to use **two independent SLURM submissions in parallel**, so each GPU job is tracked separately and can be restarted independently:

```bash
# Submit TWO jobs simultaneously — each gets 1 GPU automatically
# Job 1: Mouse on GPU 0
NUM_DESIGNS=2000 BUDGET=150 sbatch --gres=gpu:1 scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_vhh_batch1

# Job 2: Human on GPU 1
NUM_DESIGNS=2000 BUDGET=150 sbatch --gres=gpu:1 scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml runs/human_vhh_batch1

# Job 3: Cross-reactive (will queue until a GPU slot frees up)
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_batch1
```

> **Note:** The main `run_hpc_campaign.sh` now requests `gpu:2` by default. To force 1-GPU mode
> (when you want to run multiple jobs in parallel on the 2-node RTX 5000 system), use:
> ```bash
> GPUS=1 sbatch --gres=gpu:1 scripts/run_hpc_campaign.sh ...
> ```

---

## Stage 2 — Copy Results from HPC to Local Machine

After all SLURM jobs finish, copy everything back:

```bash
# From YOUR LOCAL MACHINE (not HPC)
cd ~/boltzgen/marco_boltzgen_design

rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/mouse_vhh_batch1/ runs/mouse_vhh_batch1/
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/mouse_vhh_batch2/ runs/mouse_vhh_batch2/
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/mouse_vhh_batch3/ runs/mouse_vhh_batch3/
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/human_vhh_batch1/ runs/human_vhh_batch1/
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/human_vhh_batch2/ runs/human_vhh_batch2/
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/human_vhh_batch3/ runs/human_vhh_batch3/
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/cross_vhh_batch1/ runs/cross_vhh_batch1/
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/cross_vhh_batch2/ runs/cross_vhh_batch2/
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/cross_vhh_batch3/ runs/cross_vhh_batch3/
```

Verify each batch has a metrics CSV:
```bash
ls runs/mouse_vhh_batch1/final_ranked_designs/all_designs_metrics.csv
ls runs/human_vhh_batch1/final_ranked_designs/all_designs_metrics.csv
ls runs/cross_vhh_batch1/final_ranked_designs/all_designs_metrics.csv
```

---

## Stage 3 — Collect & Merge All Metrics

```bash
cd ~/boltzgen/marco_boltzgen_design

./scripts/collect_campaign.sh \
  --runs runs/mouse_vhh_batch1 runs/mouse_vhh_batch2 runs/mouse_vhh_batch3 \
          runs/human_vhh_batch1 runs/human_vhh_batch2 runs/human_vhh_batch3 \
          runs/cross_vhh_batch1 runs/cross_vhh_batch2 runs/cross_vhh_batch3 \
  --out results/all_metrics.csv
```

Expected output:
```
Collecting: runs/mouse_vhh_batch1/.../all_designs_metrics.csv  (spec=mouse_marco_nanobody_hotspot)
Collecting: runs/mouse_vhh_batch2/.../all_designs_metrics.csv  (spec=mouse_marco_nanobody_hotspot)
...
Wrote results/all_metrics.csv  (N total rows from 9 campaigns)
```

Check total designs:
```bash
wc -l results/all_metrics.csv
# Should be ~15,000+ rows (excluding header)
```

---

## Stage 4 — Rank by Confidence, Developability & Cross-Reactivity

```bash
cd ~/boltzgen/marco_boltzgen_design

python scripts/rank_designs.py \
  --metrics results/all_metrics.csv \
  --human-conserved "A:423,A:425,A:432,A:461,A:467,A:469,A:489,A:500" \
  --mouse-conserved "A:6,A:8,A:15,A:44,A:50,A:52,A:72,A:83" \
  --max_len 140 \
  --out results/ranked_candidates.csv
```

**What this does:**
- Sorts by `final_score = mean_confidence + 0.5 × crossreactivity_score − penalties`
- Applies developability filters (Cys, length, charge, hydrophobic patches, N-glyc motifs)
- Adds `source_spec` label so you can filter by species

**Inspect top candidates:**
```bash
head -20 results/ranked_candidates.csv

# Filter to only high-confidence, cross-reactive candidates:
python3 -c "
import pandas as pd
df = pd.read_csv('results/ranked_candidates.csv')
print('Total candidates:', len(df))
print('Cross-reactive (score >= 1):', len(df[df.crossreactivity_score >= 1]))
print('High confidence (pLDDT > 80):', len(df[df.get('plddt',0) > 80]))
# Show top 10
print(df.head(10)[['design_id','source_spec','plddt','crossreactivity_score','final_score']])
"
```

---

## Stage 5 — Copy Top CIFs for Downstream Use

```bash
cd ~/boltzgen/marco_boltzgen_design

mkdir -p results/candidate_cifs

python3 -c "
import pandas as pd, shutil, pathlib
df = pd.read_csv('results/ranked_candidates.csv').head(100)

# Build design_id -> CIF path index
cif_map = {}
for run_dir in pathlib.Path('runs').glob('*/final_ranked_designs'):
    for cif in run_dir.glob('*.cif'):
        cif_map[cif.stem] = cif

copied = 0
for _, row in df.iterrows():
    did = str(row['design_id'])
    if did in cif_map:
        dst = pathlib.Path('results/candidate_cifs') / cif_map[did].name
        shutil.copy2(cif_map[did], dst)
        copied += 1
print(f'Copied {copied} CIFs to results/candidate_cifs/')
"
```

---

## Stage 6 — AF2 Backfold Validation (Top 50–100)

Validates that designed binder sequences actually fold correctly when predicted alone:

```bash
cd ~/boltzgen/marco_boltzgen_design

python scripts/validate_designs.py \
  --complexes results/candidate_cifs \
  --metrics results/ranked_candidates.csv \
  --top_n 50 \
  --method colabfold \
  --out results/af_validation.csv
```

**If colabfold is not installed locally**, run on HPC:
```bash
# On HPC
conda activate boltzgen  # or colabfold env
python scripts/validate_designs.py \
  --complexes /path/to/local/results/candidate_cifs \
  --metrics /path/to/local/results/ranked_candidates.csv \
  --top_n 50 \
  --method colabfold \
  --out results/af_validation.csv
```

**Thresholds:**
- ✅ PASS: CA RMSD < 2.5 Å AND mean PAE < 5.0 Å
- ❌ FAIL: Either threshold exceeded

**Merge AF2 results back into ranked candidates:**
```bash
python3 -c "
import pandas as pd
ranked = pd.read_csv('results/ranked_candidates.csv')
af2 = pd.read_csv('results/af_validation.csv')
merged = ranked.merge(af2[['design_id','af2_rmsd','af2_pae','af2_plddt','flag_ok']], on='design_id', how='left')
merged.to_csv('results/ranked_with_af2.csv', index=False)
passing = merged[merged['flag_ok'] == True]
print(f'AF2-passing designs: {len(passing)} / {len(merged)}')
print(passing[['design_id','source_spec','plddt','crossreactivity_score','final_score','af2_rmsd','af2_pae']].head(20))
"
```

---

## Summary — Complete Command List (Dual RTX 5000)

Copy this checklist. The strategy uses **waves** — each wave runs 2 specs in parallel
(one per GPU), then cross-reactive fills the third slot when a GPU frees up.

```bash
# ═══════════════════════════════════════════════════════════════
# STAGE 0 — Validate (on HPC login node)
# ═══════════════════════════════════════════════════════════════
cd ~/boltzgen/marco_boltzgen_design
boltzgen check specs/mouse_marco_nanobody_hotspot.yaml
boltzgen check specs/human_marco_nanobody_hotspot.yaml
boltzgen check specs/crossreactive_marco_nanobody_hotspot.yaml

# ═══════════════════════════════════════════════════════════════
# STAGE 1 — Wave 1 (Mouse + Human first, in parallel on 2 GPUs)
# Cross-reactive waits for a free GPU slot
# ═══════════════════════════════════════════════════════════════
# GPU 0 + GPU 1 — Mouse + Human (2,000 each, simultaneously)
NUM_DESIGNS=2000 BUDGET=150 sbatch --gres=gpu:1 scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_vhh_wave1
NUM_DESIGNS=2000 BUDGET=150 sbatch --gres=gpu:1 scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml runs/human_vhh_wave1

# Cross-reactive (1 GPU, starts when a slot frees up)
NUM_DESIGNS=2000 BUDGET=150 sbatch --gres=gpu:1 scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_wave1

# Monitor:
squeue -u $USER
# When all 3 show CG (completing), Wave 1 is nearly done

# ═══════════════════════════════════════════════════════════════
# STAGE 1 — Wave 2 (after Wave 1 finishes)
# ═══════════════════════════════════════════════════════════════
NUM_DESIGNS=2000 BUDGET=150 sbatch --gres=gpu:1 scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_vhh_wave2
NUM_DESIGNS=2000 BUDGET=150 sbatch --gres=gpu:1 scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml runs/human_vhh_wave2
NUM_DESIGNS=2000 BUDGET=150 sbatch --gres=gpu:1 scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_wave2

# ═══════════════════════════════════════════════════════════════
# STAGE 1 — Wave 3 (final push — 1,000 each for ~12,000 total)
# ═══════════════════════════════════════════════════════════════
NUM_DESIGNS=1000 BUDGET=150 sbatch --gres=gpu:1 scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_vhh_wave3
NUM_DESIGNS=1000 BUDGET=150 sbatch --gres=gpu:1 scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml runs/human_vhh_wave3
NUM_DESIGNS=1000 BUDGET=150 sbatch --gres=gpu:1 scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_wave3

# ═══════════════════════════════════════════════════════════════
# STAGE 2 — Collect (run locally after all HPC jobs finish)
# ═══════════════════════════════════════════════════════════════
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/ runs/

./scripts/collect_campaign.sh \
  --runs runs/mouse_vhh_wave1 runs/mouse_vhh_wave2 runs/mouse_vhh_wave3 \
          runs/human_vhh_wave1 runs/human_vhh_wave2 runs/human_vhh_wave3 \
          runs/cross_vhh_wave1 runs/cross_vhh_wave2 runs/cross_vhh_wave3 \
  --out results/all_metrics.csv

# ═══════════════════════════════════════════════════════════════
# STAGE 3 — Rank
# ═══════════════════════════════════════════════════════════════
python scripts/rank_designs.py \
  --metrics results/all_metrics.csv \
  --human-conserved "A:423,A:425,A:432,A:461,A:467,A:469,A:489,A:500" \
  --mouse-conserved "A:6,A:8,A:15,A:44,A:50,A:52,A:72,A:83" \
  --max_len 140 \
  --out results/ranked_candidates.csv

# ═══════════════════════════════════════════════════════════════
# STAGE 4 — AF2 validate top 50
# ═══════════════════════════════════════════════════════════════
python scripts/validate_designs.py \
  --complexes results/candidate_cifs \
  --metrics results/ranked_candidates.csv \
  --top_n 50 \
  --method colabfold \
  --out results/af_validation.csv
```

---

## Monitoring SLURM Jobs

```bash
# Check all your jobs
squeue -u $USER

# Watch a log in real time
tail -f logs/mouse_marco_nanobody_hotspot_<JOB_ID>.out

# Check how many designs are done so far
ls runs/mouse_vhh_wave1/intermediate_designs/*.cif 2>/dev/null | wc -l

# If job timed out, check last completed design and resume
ls runs/mouse_vhh_wave1/intermediate_designs/ | sort -V | tail -5

# Cancel a job
scancel <JOB_ID>
```

---

## Expected Timeline

| Wave | Jobs | Designs | Duration (RTX 5000) |
|------|------|---------|---------------------|
| Wave 1 | Mouse + Human (parallel) + Cross (sequential) | 6,000 | 4–6 days |
| Wave 2 | Same 3 specs | 6,000 | 4–6 days |
| Wave 3 | Same 3 specs | 3,000 | 2–3 days |
| Stages 2–4 (collect/rank/AF2) | — | — | 1–2 days |
| **Total** | **9 SLURM jobs** | **~15,000** | **11–16 days** |

---

## Key Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `NUM_DESIGNS` | 2000 | Designs per job per spec |
| `BUDGET` | 150 | Inference steps per design (higher = better quality, slower) |
| `GPUS` | 2 | GPUs per job (default 2 for RTX 5000 x2) |

**Recommended BUDGET on RTX 5000:**
- `BUDGET=100` — Fast production (~2,000 designs in ~60h)
- `BUDGET=150` — Standard (recommended, ~2,000 designs in ~90h)
- `BUDGET=200` — High quality (may exceed 96h time limit for 2,000 designs)

**Trade-off:** BUDGET=200 takes ~2× longer than BUDGET=100 but yields marginally better designs. For hotspot-constrained designs with good scaffolds, BUDGET=100–150 is the sweet spot.