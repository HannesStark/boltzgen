# MARCO Nanobody Design — Step-by-Step Instruction
## Goal: Design 10,000+ nanobodies against MARCO (mouse + human + cross-reactive)

**Time estimate:** 2–4 weeks total (depending on HPC GPU queue time)
**Hardware requirement:** HPC with ≥1 GPU (A100/H100) and ≥128 GB CPU memory per job

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

## Stage 1 — HPC Design (Multiple SLURM Batches)

> ⚠️ **A single SLURM job cannot handle 10,000 designs at once.**
> A 72-hour SLURM job can finish ~1,000–2,000 designs (depending on BUDGET).
> **You must split into batches.** This is normal and expected.

### Strategy for 10,000+ nanobodies

| Spec | Batch 1 | Batch 2 | Batch 3 | Total |
|------|---------|---------|---------|-------|
| Mouse | 2,000 | 2,000 | 1,000 | 5,000 |
| Human | 2,000 | 2,000 | 1,000 | 5,000 |
| Cross-reactive | 2,000 | 2,000 | 1,000 | 5,000 |
| **Total** | 6,000 | 6,000 | 3,000 | **15,000** |

**Each batch** = 1 SLURM submission = ~3–5 days on A100.

---

### Batch 1 — Submit first round (all 3 specs simultaneously)

```bash
cd ~/boltzgen/marco_boltzgen_design

# Mouse — Batch 1 (2,000 designs, BUDGET=150)
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_vhh_batch1

# Human — Batch 1 (2,000 designs, BUDGET=150)
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml runs/human_vhh_batch1

# Cross-reactive — Batch 1 (2,000 designs, BUDGET=150)
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_batch1
```

You will see 3 SLURM job IDs:
```
Submitted batch job 12345
Submitted batch job 12346
Submitted batch job 12347
```

**Save these job IDs** — you'll need them to monitor and potentially cancel.

---

### Monitoring SLURM jobs

```bash
# Check status of all your jobs
squeue -u $USER

# Example output:
# JOBID   PARTITION  NAME              USER  STATUS  TIME
# 12345   gpu        marco_vhh_design  jxshi RUNNING  2-13:45:32
# 12346   gpu        marco_vhh_design  jxshi RUNNING  1-08:12:01
# 12347   gpu        marco_vhh_design  jxshi RUNNING  0-15:03:44
```

**Watch a log in real time:**
```bash
tail -f logs/mouse_marco_nanobody_hotspot_12345.out
```

**What to look for — good signs:**
```
[1] design
[2] inverse_folding
[3] folding
[4] analysis
[5] filtering
Configuration complete. Configs written to runs/mouse_vhh_batch1/config
```

**What to look for — problems:**
```
subprocess.CalledProcessError ... died with <Signals.SIGKILL: 9>
```
→ Job ran out of memory or time. Note the last completed design number, then resubmit with `--reuse`.

---

### Batch 2 — Submit after Batch 1 finishes (~3–5 days later)

After all 3 Batch 1 jobs finish (check with `squeue`), submit Batch 2:

```bash
cd ~/boltzgen/marco_boltzgen_design

# Mouse — Batch 2 (2,000 more, resumes from where Batch 1 left off)
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_vhh_batch2

# Human — Batch 2
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml runs/human_vhh_batch2

# Cross-reactive — Batch 2
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_batch2
```

The `--reuse` flag ensures already-designed structures are skipped — no duplication.

---

### Batch 3 — Final push (1,000 each, if you want 15,000 total)

```bash
# Mouse — Batch 3
NUM_DESIGNS=1000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_vhh_batch3

# Human — Batch 3
NUM_DESIGNS=1000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml runs/human_vhh_batch3

# Cross-reactive — Batch 3
NUM_DESIGNS=1000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_batch3
```

---

### If a job hits SLURM time limit before finishing

```bash
# Check how far it got
ls runs/mouse_vhh_batch1/intermediate_designs/ | wc -l

# Resubmit the SAME command — --reuse skips completed designs
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_vhh_batch1
```

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

## Summary — Complete Command List

Copy this checklist for your records:

```bash
# ═══════════════════════════════════════════════════════════════
# STAGE 0 — Validate
# ═══════════════════════════════════════════════════════════════
cd ~/boltzgen/marco_boltzgen_design
boltzgen check specs/mouse_marco_nanobody_hotspot.yaml
boltzgen check specs/human_marco_nanobody_hotspot.yaml
boltzgen check specs/crossreactive_marco_nanobody_hotspot.yaml

# ═══════════════════════════════════════════════════════════════
# STAGE 1 — HPC Batch 1 (submit all 3 at once, then wait)
# ═══════════════════════════════════════════════════════════════
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh specs/mouse_marco_nanobody_hotspot.yaml      runs/mouse_vhh_batch1
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh specs/human_marco_nanobody_hotspot.yaml     runs/human_vhh_batch1
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_batch1

# Wait for all 3 to finish, then:
# STAGE 1 — HPC Batch 2
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh specs/mouse_marco_nanobody_hotspot.yaml      runs/mouse_vhh_batch2
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh specs/human_marco_nanobody_hotspot.yaml     runs/human_vhh_batch2
NUM_DESIGNS=2000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_batch2

# Wait, then:
# STAGE 1 — HPC Batch 3
NUM_DESIGNS=1000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh specs/mouse_marco_nanobody_hotspot.yaml      runs/mouse_vhh_batch3
NUM_DESIGNS=1000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh specs/human_marco_nanobody_hotspot.yaml     runs/human_vhh_batch3
NUM_DESIGNS=1000 BUDGET=150 sbatch scripts/run_hpc_campaign.sh specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_batch3

# ═══════════════════════════════════════════════════════════════
# STAGE 2 — Collect (run locally after HPC jobs finish)
# ═══════════════════════════════════════════════════════════════
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/ runs/
./scripts/collect_campaign.sh \
  --runs runs/mouse_vhh_batch1 runs/mouse_vhh_batch2 runs/mouse_vhh_batch3 \
          runs/human_vhh_batch1 runs/human_vhh_batch2 runs/human_vhh_batch3 \
          runs/cross_vhh_batch1 runs/cross_vhh_batch2 runs/cross_vhh_batch3 \
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

## Expected Timeline

| Phase | Duration | What happens |
|-------|----------|--------------|
| Batch 1 (3 specs × 2,000) | 3–5 days | All 3 run simultaneously on 3 GPUs |
| Batch 2 (3 specs × 2,000) | 3–5 days | After Batch 1 finishes |
| Batch 3 (3 specs × 1,000) | 1–2 days | Final push |
| Stages 2–4 (collect/rank/AF2) | 1–2 days | Local processing |
| **Total** | **8–14 days** | **~15,000 designs** |

---

## Key Parameters Explained

| Parameter | Default | Meaning | Higher = |
|-----------|---------|---------|---------|
| `NUM_DESIGNS` | 1000 | How many binder structures to generate per batch | More candidates |
| `BUDGET` | 150 | Inference steps per design | Better quality, longer runtime |

**Recommended BUDGET values:**
- `BUDGET=50` — Fast pilot (1–2 days for 2,000 designs)
- `BUDGET=100` — Standard production
- `BUDGET=200` — High quality (use if you have time)

**Trade-off:** BUDGET=200 takes ~2× longer than BUDGET=100 but yields marginally better designs. For hotspot-constrained designs with good scaffolds, BUDGET=100–150 is the sweet spot.