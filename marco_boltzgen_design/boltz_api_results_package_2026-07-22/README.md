# MARCO SRCR Boltz API Results Package

Date: 2026-07-22

This package organizes the MARCO SRCR nanobody design results generated with Boltz API.
It is intended as a compact, repo-friendly record of the focused 1500-design campaign,
the Top50 MARCO/off-target specificity checks, and the final 30-candidate panel.

## Contents

### focused_1500

Focused design campaign around the best-performing settings:

- `broad_original_neighbors_h036`: 500 designs, hydrophobic cap 0.36
- `nterm_basic_patch_h048`: 500 designs, hydrophobic cap 0.48
- `cterm_acidic_patch_h048`: 500 designs, hydrophobic cap 0.48

Key files:

- `tables/all_1500_focused_candidates.csv`: all 1500 focused candidates and design metrics
- `tables/strong_focused_candidates.csv`: 262 stronger focused candidates after filtering/ranking
- `tables/variant_summary.csv`: per-setting summary
- `tables/manifest.json`: local run manifest
- `inputs/*.json`: Boltz API design inputs for the three 500-design jobs
- `top_focused_cifs/*.cif`: top 100 focused-design CIF structures

The full raw focused run directory was about 833 MB and contained thousands of cache/output
files. This repo package keeps the complete candidate tables and representative top CIFs,
but omits bulky run-cache internals.

### specificity_top50

Structure-and-binding validation and counter-screening for the Top50 reranked candidates.
Each candidate was predicted against:

- `MARCO_SRCR_DESIGN_TARGET`
- `MSR1_SCARA1_SRCR`
- `SCARA5_SRCR`
- `CD163_SRCR3`

Key files:

- `tables/all_3700_reranked_candidates.csv`: rerank table across original, diversified, and focused designs
- `tables/top50_for_structure_binding_validation.csv`: Top50 submitted for validation/counter-screening
- `tables/validation_metrics.csv`: all 200 structure-and-binding prediction metrics
- `tables/candidate_selectivity_summary.csv`: per-candidate specificity summary
- `tables/recommended_after_top50_validation.csv`: 13 recommended validated/selective leads
- `tables/prediction_manifest.csv` and `.json`: prediction job manifest
- `tables/prediction_inputs.jsonl`: aggregated JSON inputs for all 200 prediction jobs
- `tables/targets.json`: target/off-target SRCR sequences used in the panel
- `cif_files_by_target/*/*.cif`: all 200 MARCO/off-target predicted complex CIF files

### panel_30

Final 30-candidate handoff panel.

Key files:

- `tables/panel_30_candidates.csv`: full metrics for the selected 30-candidate panel
- `tables/panel_30_synthesis_order_sheet.csv`: compact synthesis/order sheet
- `tables/panel_30_nanobody_sequences.fasta`: FASTA sequences
- `marco_cifs/*.cif`: MARCO complex CIFs for the 30 panel candidates

## Reading the specificity columns

- Higher `marco_complex_iplddt` suggests the MARCO complex structure is locally confident.
- Higher `marco_iptm` suggests the predicted MARCO-binder interface placement is more reliable.
- Lower `marco_interface_pae_mean` suggests less interface uncertainty.
- Positive `iplddt_margin_vs_best_offtarget` and `iptm_margin_vs_best_offtarget` are preferred;
  negative margins indicate an off-target SRCR prediction scored better than MARCO by that metric.

Computational predictions should be used for prioritization, not as final binding proof.
