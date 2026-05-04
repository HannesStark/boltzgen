# MARCO/Marco BoltzGen Binder Design Workflow

This folder provides a practical workflow for **de novo binder design** against:
- human MARCO
- mouse Marco
- conserved epitope cross-reactive designs
- optional peptide binder designs
- optional VHH nanobody binder designs

## 1) Setup

```bash
cd /workspace/boltzgen/marco_boltzgen_design
./scripts/setup_env.sh bg-marco
conda activate bg-marco
```

## 2) Prepare target structures

Use your own PDB/mmCIF models (experimental or AlphaFold) and copy them into `targets/`:

```bash
cd scripts
./prepare_targets.sh /abs/path/human.cif A /abs/path/mouse.cif A
```

Then validate YAML parsing and residue indexing (label_seq_id):

```bash
cd ..
boltzgen check specs/human_marco_binder_anywhere.yaml
boltzgen check specs/crossreactive_conserved_surface.yaml
boltzgen check specs/human_marco_nanobody_anywhere.yaml
```

## 3) YAML templates provided

Protein-anything templates:
- `specs/human_marco_binder_anywhere.yaml`
- `specs/mouse_marco_binder_anywhere.yaml`
- `specs/human_marco_binder_hotspot.yaml`
- `specs/mouse_marco_binder_hotspot.yaml`
- `specs/crossreactive_conserved_surface.yaml`

Peptide-anything templates:
- `specs/human_marco_peptide_anywhere.yaml`
- `specs/mouse_marco_peptide_anywhere.yaml`

Nanobody-anything templates (VHH):
- `specs/human_marco_nanobody_anywhere.yaml`
- `specs/mouse_marco_nanobody_anywhere.yaml`
- `specs/human_marco_nanobody_hotspot.yaml`
- `specs/mouse_marco_nanobody_hotspot.yaml`
- `specs/crossreactive_marco_nanobody_hotspot.yaml`

> If you need explicit cyclic constraints, extend using `constraints: - bond:` in the same syntax as BoltzGen examples.

## 4) Run strategy

### Pilot (10–50 designs)
```bash
./runs/run_pilot.sh specs/human_marco_binder_anywhere.yaml protein-anything runs/human_pilot
NUM_DESIGNS=20 BUDGET=5 ./runs/run_pilot.sh specs/crossreactive_conserved_surface.yaml protein-anything runs/cross_pilot
```

### Production (500–2000 designs)
```bash
NUM_DESIGNS=1500 BUDGET=200 ./runs/run_production.sh specs/crossreactive_conserved_surface.yaml protein-anything runs/cross_prod
NUM_DESIGNS=1000 BUDGET=120 ./runs/run_production.sh specs/human_marco_binder_hotspot.yaml protein-anything runs/human_hotspot_prod
```

### Nanobody VHH campaign
```bash
NUM_DESIGNS=200 BUDGET=40 ./runs/run_nanobody_campaign.sh specs/human_marco_nanobody_anywhere.yaml runs/human_vhh_pilot
NUM_DESIGNS=1200 BUDGET=180 ./runs/run_nanobody_campaign.sh specs/crossreactive_marco_nanobody_hotspot.yaml runs/cross_vhh_prod
# If you intentionally want Cys in generated CDRs:
# EXTRA_ARGS='--inverse_fold_avoid ""' ./runs/run_nanobody_campaign.sh ...
```

### SLURM/HPC
```bash
sbatch runs/run_hpc_slurm.sh specs/crossreactive_conserved_surface.yaml protein-anything runs/slurm_cross
```

## 5) Post-processing and ranking

BoltzGen already computes analysis/filtering metrics. Use `rank_designs.py` to apply MARCO-specific developability and cross-reactivity scoring:

```bash
python scripts/rank_designs.py \
  --metrics runs/cross_prod/final_ranked_designs/all_designs_metrics.csv \
  --contacts results/interface_contacts.csv \
  --human-conserved "A:340,A:344,A:350,A:352" \
  --mouse-conserved "A:337,A:341,A:347,A:349" \
  --out results/ranked_candidates.csv
```

Expected outputs tracked per design (when present in metrics/contacts):
- predicted complex structure path (BoltzGen refold CIF)
- binder sequence, binder length
- interface contacts
- buried surface area (if provided by your contacts script)
- confidence metrics (e.g., pLDDT/ipTM/ranking score if present)
- contacted target residues
- species cross-reactivity score
- developability flags (Cys, charge, hydrophobicity, NXS/T motif, length)

## 6) Practical MARCO design logic

1. Start with SRCR domain-focused designs (accessible extracellular surface).
2. First round: unconstrained binder-anywhere on human and mouse separately.
3. Second round: hotspot-constrained templates on exposed residues.
4. Third round: cross-reactive conserved-surface template.
5. For VHH, run the analogous nanobody-anything templates with scaffold inputs.
6. If cross-reactive recovery is low, split into species-selective campaigns.
5. If cross-reactive recovery is low, split into species-selective campaigns.

## 7) Manual fill-in checklist

Before serious runs, edit the spec files and complete:
- [ ] target PDB/mmCIF paths in `targets/`
- [ ] chain IDs (all spec files currently placeholder `A`)
- [ ] hotspot residue numbers (label_seq_id)
- [ ] conserved residue mapping (human↔mouse)
- [ ] desired binder length range (default 50..90 aa for miniproteins)
- [ ] desired binder length range (default 50..90 aa)
- [ ] number of designs / budget
- [ ] GPU/devices and SLURM resources
- [ ] (optional) whether cysteine is allowed

## Notes from repository inspection

- CLI flow: `boltzgen run/check/configure/execute` supports staged workflows.
- Use `--reuse` to resume interrupted runs.
- `protein-anything` includes design-folding; `peptide-anything` has peptide-specific filtering behavior.
- `nanobody-anything` supports VHH CDR design with nanobody scaffolds.
- Binding-site residue indexing must use **mmCIF label_seq_id** indexing.
