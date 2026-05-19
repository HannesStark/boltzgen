# MARCO / Marco BoltzGen Nanobody Design

De novo VHH nanobody design against the SRCR (Scavenger Receptor Cysteine-Rich) domain of human MARCO and mouse Marco.

---

## Project Structure

```
marco_boltzgen_design/
├── targets/
│   ├── human_MARCO_input.cif     # Human MARCO full sequence (Q9UEW3, 1-520)
│   └── mouse_marco_srcr.cif     # Mouse Marco SRCR domain (2OYA, 102 aa)
├── specs/
│   ├── mouse_marco_nanobody_hotspot.yaml          # Original DSSP hotspots (SO4 site)
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
│   ├── run_hpc_campaign.sh      # HPC SLURM launcher (primary production script)
│   ├── collect_campaign.sh      # Merge metrics from multiple runs
│   └── rank_and_validate.sh     # Rank + AF2 backfold validation
├── runs/                        # Design output goes here
├── results/                     # Ranked/validated results
└── README.md
```

---

## 1) Setup

```bash
# Clone on HPC
git clone https://github.com/jxshi/boltzgen.git
cd boltzgen/marco_boltzgen_design

# Activate environment
conda activate boltzgen   # or: source ~/miniconda3/etc/profile.d/conda.sh && conda activate boltzgen
```

---

## 2) Choose an Interface Set

Four distinct interface sets have been designed based on structural, patent, and beta-pairing design logic. All are intended to be run with `--protocol nanobody-anything` and MARCO-specific filtering flags.

### Set A — SO4 / Ligand-Blocking Pocket
**Purpose:** Block MARCO ligand binding (LDL, oxLDL, bacteria, apoptotic cells).
**Residues:** `Q429, Y431, K438, Q467, K473, Q475, W495, H506`

| Spec | label_seq | Notes |
|------|-----------|-------|
| `specs/mouse_marco_nanobody_setA_so4_pocket.yaml` | 12,14,21,50,56,58,78,89 | Mouse SRCR (2OYA) |
| `specs/human_marco_nanobody_setA_so4_pocket.yaml` | 12,14,21,50,56,58,78,89 | Human MARCO (Q9UEW3) |

### Set B — Patent Antibody Epitope
**Purpose:** Mimic functional antibodies PI-3010/PI-3035 (agonist, crosslinking, immune reprogramming).
**Residues:** `Q450, Y452, K473, Q487, T499, H505, D507, S509, E511`

| Spec | label_seq | Notes |
|------|-----------|-------|
| `specs/human_marco_nanobody_setB_patent_epitope.yaml` | 33,35,56,70,82,88,90,92,94 | Human only |

> ⚠️ Human Q452 = mouse D452 (species差异). Set B is human-only.

### Set C — Hybrid Interface (Cross-Reactive)
**Purpose:** Maximize paratope size and stability — combines SO4 pocket + patent epitope.
**Residues:** Union of Sets A and B.

| Spec | Human label_seq | Mouse label_seq | Notes |
|------|----------------|-----------------|-------|
| `specs/crossreactive_marco_nanobody_setC_hybrid.yaml` | 35,50,56,58,70,78,82,88,89,90,92,94 | 12,14,21,50,56,58,78,89 | Cross-reactive |

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

## 3) Validate Specs

Always validate before submitting jobs:

```bash
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

## 4) Run Nanobody Design

### Local pilot (quick test, 10–50 designs)

The wrapper now applies the recommended MARCO defaults automatically: `--protocol nanobody-anything`, `--diffusion_batch_size 2`, `plip_hbonds_refolded=0.2`, `delta_sasa_refolded=0.5`, and `--refolding_rmsd_threshold 3.0`.

```bash
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

### HPC production (1000–3000 designs)

**Single-GPU job:**
```bash
# Set A — mouse
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_setA_so4_pocket.yaml runs/setA_mouse &

# Set A — human
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setA_so4_pocket.yaml runs/setA_human &

# Set B — patent epitope (human-only)
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setB_patent_epitope.yaml runs/setB &

# Set C — hybrid cross-reactive
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setC_hybrid.yaml runs/setC &

# Set D — beta-pairing cross-reactive
NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_beta_pairing &

wait
```

**Dual-GPU (both RTX 5000s, parallel jobs):**
```bash
# Submit two single-GPU jobs simultaneously to fill both cards
GPUS=1 NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/human_marco_nanobody_setA_so4_pocket.yaml runs/setA_human_gpu1 &

GPUS=1 NUM_DESIGNS=2000 BUDGET=200 sbatch scripts/run_hpc_campaign.sh \
  specs/mouse_marco_nanobody_setA_so4_pocket.yaml runs/setA_mouse_gpu2 &

wait
```

> **RTX 5000 notes:** Each card has 16 GB VRAM. `GPUS=1` uses one card with `--devices 1`. Default `DEVICES=2` uses both cards in a single job. Recommended BUDGET: 150–200 for 2000 designs in 96h.

---

## 5) Collect Results

After jobs complete:

```bash
# Merge metrics from all runs
./scripts/collect_campaign.sh \
  --runs runs/setA_mouse runs/setA_human runs/setB runs/setC runs/setD_beta_pairing \
  --out results/all_metrics.csv
```

---

## 6) Rank and AF2 Validate

```bash
./scripts/rank_and_validate.sh \
  --metrics results/all_metrics.csv \
  --human-conserved "A:35,A:50,A:56,A:58,A:70,A:78,A:82,A:88,A:89,A:90,A:92,A:94" \
  --mouse-conserved "A:12,A:14,A:21,A:50,A:56,A:58,A:78,A:89" \
  --top_n 50
```

For Set B (human-only), use:
```bash
./scripts/rank_and_validate.sh \
  --metrics results/all_metrics.csv \
  --human-conserved "A:33,A:35,A:56,A:70,A:82,A:88,A:90,A:92,A:94" \
  --top_n 50
```

---

## 7) Reference: Numbering

| System | Offset | Notes |
|--------|--------|-------|
| Q9UEW3 (human) | −417 → label_seq | Full sequence positions 1-520 |
| Mouse 2OYA | Direct label_seq 1-102 | Isolated SRCR domain |

**Example:** Q9UEW3 position 452 → label_seq = 452 − 417 = **35**

---

## Appendix A: Original Specs (pre-June 2026)

Legacy `binder` and `peptide` filenames in `specs/` have been converted to VHH scaffold specs and should also be run with `--protocol nanobody-anything`.

The original DSSP-derived hotspot specs remain available:
- `specs/mouse_marco_nanobody_hotspot.yaml` — label_seq 6,8,15,44,50,52,72,83
- `specs/human_marco_nanobody_hotspot.yaml` — label_seq 6,8,15,44,50,52,72,83 (offset-applied)
- `specs/crossreactive_marco_nanobody_hotspot.yaml` — cross-reactive union

---

## Appendix B: Allow Cysteine in CDRs

By default, nanobody-anything avoids Cys. To permit Cys:
```bash
EXTRA_ARGS='--inverse_fold_avoid ""' ./runs/run_nanobody_campaign.sh specs/... runs/...
```

---

## Appendix C: Hotspot Discovery

If you have antibody complex structures and want to discover new interfaces:

```bash
python scripts/find_marco_srcr_hotspots.py \
  --human-structure targets/human_MARCO_input.cif --human-chain A \
  --mouse-structure targets/mouse_marco_srcr.cif --mouse-chain A \
  --human-complexes runs/human_complexes/*.cif \
  --mouse-complexes runs/mouse_complexes/*.cif \
  --human-binder-chains H,L --mouse-binder-chains H,L \
  --out results/marco_hotspots.csv
```