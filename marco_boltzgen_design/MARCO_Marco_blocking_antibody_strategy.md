# MARCO / Marco Blocking Antibody Development Strategy

This file operationalizes a function-first plan for discovering blocking antibodies against human MARCO and mouse Marco SRCR domains.

## Scope and intended use
- Focus on **functional blockade**, not affinity alone.
- Prioritize **cross-reactive** conserved epitopes first; run species-specific routes in parallel/fallback.
- Couple in silico design/ranking with assay gating, KO controls, and RO→PD bridging.

## 1) Epitope strategy
- Prioritize SRCR functional surface (basic patch + acidic Ca-associated loop).
- Use three epitope lanes:
  1. cross-species conserved hotspot
  2. human-selective divergent loops
  3. mouse-selective divergent loops

## 2) Antigen panel to generate
Place sequence constructs in `antigens/`:
- `human_marco_srcr_wt.fasta`
- `mouse_marco_srcr_wt.fasta`
- `human_marco_srcr_basic_patch_mutants.fasta`
- `mouse_marco_srcr_basic_patch_mutants.fasta`
- `human_marco_srcr_acidic_loop_mutants.fasta`
- `mouse_marco_srcr_acidic_loop_mutants.fasta`
- `human_marco_srcr_s1_swap.fasta`
- `mouse_marco_srcr_s1_swap.fasta`
- `human_IZN4_srcr_linker_G4S2.fasta`
- `human_IZN4_srcr_linker_G4S3.fasta`
- `mouse_IZN4_srcr_linker_G4S2.fasta`
- `mouse_IZN4_srcr_linker_G4S3.fasta`

Use mammalian expression with Ca/Mg present where practical.

## 3) Discovery funnels
### 3.1 Display campaigns
- R1: specificity foundation + decoy pre-clears.
- R2: mutant negative selection for hotspot dependence.
- R3: ligand competition / masked antigen pressure.
- R4: on-cell ligand-displacement gating.
- R5: monomer/off-rate stringency.

### 3.2 Hybridoma campaigns
- Prime trimer, boost monomer, alternate.
- Primary gate: ligand inhibition on MARCO+ cells.
- Secondary gate: KO/SR-A controls and internalization behavior.

## 4) Functional assays and in vivo controls
Templates are provided under:
- `assays/ligand_competition_template.csv`
- `assays/internalization_template.csv`
- `assays/phagocytosis_template.csv`
- `in_vivo/study_design_template.csv`
- `in_vivo/ro_pd_template.csv`

## 5) Computational developability stack
Use `analysis/developability_rank.py` on VH/VL sequences to flag:
- Cys, N-X-S/T motifs
- charge and hydrophobicity proxies
- length outliers
- simple risk score for triage

## 6) Decision gates (operational)
Advance only if all are met:
1. Functional block on MARCO+ cells with dose response.
2. Specificity (minimal SR-A-only effect, KO-negative).
3. RO→PD support in WT/hMARCO-KI; absent in KO.
4. Acceptable developability profile.

## 7) Immediate next steps
1. Fill antigen FASTAs in `antigens/`.
2. Populate MARCO/Marco target structures in `targets/` and run BoltzGen spec checks.
3. Run nanobody and protein campaigns in pilot mode.
4. Import assay data into templates and run analysis scripts.
