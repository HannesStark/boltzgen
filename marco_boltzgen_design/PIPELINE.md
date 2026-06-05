# MARCO Nanobody Design — Formal Pipeline

> **Design nanobodies against mouse Marco and human MARCO SRCR domains using BoltzGen.**
> This pipeline is designed for **HPC (SLURM)** execution. Local Mac execution is only
> suitable for pilots up to ~50 designs; production runs must use the HPC scripts.

---

## Pipeline Overview

```
Stage 0          Stage 1          Stage 2          Stage 3          Stage 4
─────────        ─────────        ─────────        ─────────        ─────────
Validation  →   HPC Design   →   Collect &   →   Rank &       →   AF2
(pre-flight)      (SLURM)          Merge           Filter         Validation
```

| Stage | What | Where | Output |
|-------|------|-------|--------|
| **0 — Validation** | Spec check, target check, env check | Local | PASS/FAIL |
| **1 — Design** | `boltzgen run` 5-step pipeline on SLURM | HPC (GPU node) | CIF + NPZ in `runs/<name>/` |
| **2 — Collect** | Gather outputs from HPC, merge metrics | Local | `results/all_metrics.csv` |
| **3 — Rank** | `rank_designs.py` + developability filters | Local | `results/ranked_candidates.csv` |
| **4 — Validate** | `validate_designs.py` (AF2 backfold) | Local/HPC | `results/af_validation.csv` |

---

## Directory Structure

```
~/boltzgen/marco_boltzgen_design/
├── targets/                          # Input target structures
│   ├── mouse_marco_srcr.cif          # 2OYA SRCR domain (label_seq 1-102)
│   └── human_MARCO_input.cif        # Q9UEW3 full sequence (1-520)
│
├── specs/                            # Design specifications
│   ├── mouse_marco_nanobody_hotspot.yaml
│   ├── human_marco_nanobody_hotspot.yaml
│   ├── crossreactive_marco_nanobody_hotspot.yaml
│   └── _defaults.md                  # Diffusion parameter reference
│
├── scripts/                          # Utility scripts
│   ├── run_hpc_campaign.sh            # ★ HPC campaign runner (5-step pipeline)
│   ├── collect_campaign.sh            # ★ Gather + merge HPC outputs
│   ├── rank_and_validate.sh           # ★ Rank + AF2 validation orchestrator
│   ├── rank_designs.py               # Ranking + developability filter
│   ├── validate_designs.py           # AF2 backfold validation
│   ├── aggregate_campaigns.py         # Cross-campaign metric aggregation
│   └── find_marco_srcr_hotspots.py
│
├── runs/                             # Design run outputs
│   ├── mouse_vhh_pilot/              # ← pilot (local, OOM'd)
│   ├── human_vhh_pilot/              # ← pilot (local, OOM'd)
│   ├── cross_vhh_pilot/              # ← pilot (local, OOM'd)
│   ├── mouse_vhh_prod/               # ★ production run
│   ├── human_vhh_prod/
│   ├── cross_vhh_prod/
│   └── slurm_*/
│
└── results/                          # Post-processing outputs
    ├── all_metrics.csv                # Stage 2: merged from all campaigns
    ├── ranked_candidates.csv          # Stage 3: scored + filtered
    ├── af_validation.csv              # Stage 4: AF2 backfold results
    └── candidate_cifs/                # Top-ranked CIFs for experimental use
```

---

## Stage 0 — Pre-flight Validation (Local)

Always run before submitting an HPC job.

```bash
# 1. Validate all specs
boltzgen check specs/mouse_marco_nanobody_hotspot.yaml
boltzgen check specs/human_marco_nanobody_hotspot.yaml
boltzgen check specs/crossreactive_marco_nanobody_hotspot.yaml

# 2. Check target files exist
ls -la targets/mouse_marco_srcr.cif targets/human_MARCO_input.cif

# 3. Verify conda env on HPC (ssh to HPC node first)
ssh hpc-login
conda activate boltzgen
boltzgen --version
```

---

## Stage 1 — HPC Design (SLURM)

### Single-spec production run

```bash
# From local machine — submits SLURM job for ONE spec
NUM_DESIGNS=1000 BUDGET=200 ./scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml \
  runs/mouse_vhh_prod
```

### Multi-spec campaign (recommended)

Submit all three specs in sequence:

```bash
# Mouse (1000 designs, 200 iters/design)
NUM_DESIGNS=1000 BUDGET=200 ./scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_vhh_prod &

# Human (1000 designs, 200 iters/design)
NUM_DESIGNS=1000 BUDGET=200 ./scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml runs/human_vhh_prod &

# Cross-reactive (1000 designs, 200 iters/design)
NUM_DESIGNS=1000 BUDGET=200 ./scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_prod &

wait  # wait for all three to complete
```

### Speed Mode (2–4× faster for large batches)

Set `SPEED_MODE=1` to apply aggressive folding optimizations:
`sampling_steps` 200→100, `recycling_steps` 3→1, `diffusion_samples` 5→1,
`design compile_pairformer/structure=true`, `inverse_fold precision=bf16-mixed`,
`diffusion_batch_size` 2→8.

```bash
# Standard quality mode (default)
NUM_DESIGNS=1000 BUDGET=200 ./scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_vhh_prod

# Speed mode — recommended for 5000+ design batches
SPEED_MODE=1 NUM_DESIGNS=5000 BUDGET=200 ./scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_fast
```

> **Tip:** Use speed mode for screening to generate more candidates faster.
> Fall back to quality mode for final validation of top-ranked candidates.

### SLURM resource requirements

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--gres=gpu:1` | 1 × A100/H100 | BoltzGen requires dedicated GPU |
| `--mem` | 64–128 GB | OOM killed at < 48 GB |
| `--cpus-per-task` | 8–16 | Parallel data loading |
| `--time` | 24–72 h | Design step is ~10–20 min/design |
| `--partition` | gpu | Use GPU partition |

### Monitoring SLURM jobs

```bash
# Check job status
squeue -u $USER

# Watch log (replace <JOB_ID>)
tail -f logs/boltzgen_marco_<JOB_ID>.out

# Cancel
scancel <JOB_ID>
```

### Understanding the 5-step pipeline

`run_hpc_campaign.sh` runs the full BoltzGen nanobody-anything pipeline:

```
[1] design          Generate novel CDR sequences attached to scaffolds
[2] inverse_folding   Score/filter by structure recovery of scaffold
[3] folding           Predict full binder + target complex structure
[4] analysis          Compute pLDDT, ipTM, PAE, interface metrics
[5] filtering         Apply confidence/diversity filters
```

Outputs land in `runs/<name>/`:
- `intermediate_designs/*.cif` + `*.npz` — per-design complexes after folding
- `final_ranked_designs/all_designs_metrics.csv` — aggregate metrics for all designs
- `final_ranked_designs/*.cif` — top-ranked complex structures

---

## Stage 2 — Collect & Merge (Local)

After HPC jobs finish, pull results back and merge:

```bash
# Copy from HPC to local (if running remote)
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/mouse_vhh_prod/ runs/mouse_vhh_prod/
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/human_vhh_prod/ runs/human_vhh_prod/
rsync -avz hpc:boltzgen/marco_boltzgen_design/runs/cross_vhh_prod/  runs/cross_vhh_prod/

# Merge all metrics CSVs
./scripts/collect_campaign.sh \
  --runs runs/mouse_vhh_prod \
            runs/human_vhh_prod \
            runs/cross_vhh_prod \
  --out results/all_metrics.csv
```

`collect_campaign.sh` aggregates all `all_designs_metrics.csv` files and adds a `source_spec` column.

---

## Stage 3 — Rank & Filter (Local)

```bash
python scripts/rank_designs.py \
  --metrics results/all_metrics.csv \
  --human-conserved "A:423,A:425,A:432,A:461,A:467,A:469,A:489,A:500" \
  --mouse-conserved "A:6,A:8,A:15,A:44,A:50,A:52,A:72,A:83" \
  --max_len 140 \
  --out results/ranked_candidates.csv
```

| Flag | Meaning |
|------|---------|
| `has_cys` | Cys present (usually disfavoured in CDRs) |
| `too_long` | Binder length > 140 aa |
| `excess_positive_charge` | Net charge > +8 |
| `hydrophobic_patch_flag` | > 50% hydrophobic residues |
| `nglyc_motif` | NXS/T sequon present (Nglyc site) |
| `crossreactivity_score` | How many conserved hotspot residues are contacted |
| `final_score` | `mean_confidence + 0.5 × crossreactivity_score − penalties` |

### Filtering thresholds

| Metric | Pass | Warn | Fail |
|--------|------|------|------|
| pLDDT | > 80 | 70–80 | < 70 |
| ipTM | > 0.6 | 0.4–0.6 | < 0.4 |
| Binder length | 110–140 aa | 100–110 or 140–150 | < 100 or > 150 |
| Net charge | −5 to +8 | ±(8–12) | < −15 or > +12 |

---

## Stage 4 — AF2 Validation (Local)

Validates that the designed binder sequences **back-fold correctly** when predicted alone with AF2:

```bash
python scripts/validate_designs.py \
  --complexes results/candidate_cifs \
  --metrics results/ranked_candidates.csv \
  --top_n 50 \
  --method colabfold \
  --out results/af_validation.csv
```

**Thresholds:**
- CA RMSD (AF2 vs BoltzGen design) < 2.5 Å
- Mean PAE < 5.0 Å

Designs passing AF validation are ready for experimental characterization.

---

## Hotspot Reference

> **⚠️ Numbering matters.** All residue IDs in specs must use **mmCIF label_seq_id**.
> See `specs/_defaults.md` for the full mapping.

| Species | Coordinate system | Key residues |
|---------|------------------|--------------|
| Mouse Marco (2OYA) | 2OYA label_seq 1–102 | 6, 8, 15, 44, 50, 52, 72, 83 |
| Human MARCO (Q9UEW3) | Q9UEW3 position = label_seq (offset=417) | 423, 425, 432, 461, 467, 469, 489, 500 |
| Cross-reactive union | Both above | All of the above |

---

## Common Pitfalls

### OOM on local machine
BoltzGen design step requires ≥ 64 GB RAM + dedicated GPU. **Always use HPC for production.** Only pilots up to 50 designs can run locally, and even those risk OOM.

### Wrong residue numbering
The human MARCO spec uses **Q9UEW3 full-sequence positions** (1-520), not label_seq offset. Mouse uses 2OYA label_seq (1-102). Mixing these up will produce wrong hotspots.

### Diffusion params as CLI args
Diffusion settings (`NUM_STEPS`, `GUIDANCE_SCALE`) go in the **spec YAML** as CLI flags, NOT in `EXTRA_ARGS`. Passing them as CLI args causes `boltzgen: error: unrecognized arguments`.

### Using `nanobody-anything` with hotspots
The `nanobody-anything` protocol respects `binding:` residue constraints in the spec YAML. Do NOT override the protocol from the command line. The spec file already specifies the correct protocol in its header comment.

### Incomplete runs (SIGKILL)
If SLURM job was killed before completing, re-run with `--reuse` flag (already set in `run_hpc_campaign.sh`). The design step is resumable — already-generated designs are skipped.

---

## Quick-Start Summary

```bash
# ── STAGE 0: validate ──
boltzgen check specs/mouse_marco_nanobody_hotspot.yaml
boltzgen check specs/human_marco_nanobody_hotspot.yaml
boltzgen check specs/crossreactive_marco_nanobody_hotspot.yaml

# ── STAGE 1: HPC design (submit all three) ──
NUM_DESIGNS=1000 BUDGET=200 ./scripts/run_hpc_campaign.sh specs/mouse_marco_nanobody_hotspot.yaml      runs/mouse_vhh_prod &
NUM_DESIGNS=1000 BUDGET=200 ./scripts/run_hpc_campaign.sh specs/human_marco_nanobody_hotspot.yaml     runs/human_vhh_prod &
NUM_DESIGNS=1000 BUDGET=200 ./scripts/run_hpc_campaign.sh specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_prod &
wait

# ── STAGE 2: collect ──
./scripts/collect_campaign.sh \
  --runs runs/mouse_vhh_prod runs/human_vhh_prod runs/cross_vhh_prod \
  --out results/all_metrics.csv

# ── STAGE 3: rank ──
python scripts/rank_designs.py \
  --metrics results/all_metrics.csv \
  --human-conserved "A:423,A:425,A:432,A:461,A:467,A:469,A:489,A:500" \
  --mouse-conserved "A:6,A:8,A:15,A:44,A:50,A:52,A:72,A:83" \
  --out results/ranked_candidates.csv

# ── STAGE 4: AF2 validate ──
python scripts/validate_designs.py \
  --complexes results/candidate_cifs \
  --metrics results/ranked_candidates.csv \
  --top_n 50 \
  --method colabfold \
  --out results/af_validation.csv
```