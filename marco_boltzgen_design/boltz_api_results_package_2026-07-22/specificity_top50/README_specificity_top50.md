# MARCO Top-50 Validation and Counter-Screen 2026-07-21

This package reranks all 3700 MARCO nanobody designs and validates the top 50 using Boltz `predictions:structure-and-binding`.

Validation panel: MARCO SRCR design target plus MSR1/SCARA1 SRCR, SCARA5 SRCR, and CD163 SRCR3.

Total structure-and-binding predictions: 200 = 50 candidates x 4 targets.

Input and idempotency keys are recorded in `prediction_manifest.csv`.

## Completion and Results

Cost estimates returned `$0.0250` per complex. The full 200-complex validation/counter-screen was estimated at about `$5.00`.

All 200 predictions completed and downloaded successfully.

Generated outputs:

- `all_3700_reranked_candidates.csv`: combined rerank of original, diversified, and focused designs.
- `top50_for_structure_binding_validation.csv`: top 50 submitted to Boltz structure-and-binding.
- `validation_metrics.csv`: one row per candidate-target complex.
- `candidate_selectivity_summary.csv`: MARCO-vs-off-target comparison for all 50.
- `recommended_after_top50_validation.csv`: candidates passing the validation/selectivity filters.
- `cif_files_by_target/`: all 200 CIFs grouped by target.
- `recommended_marco_cifs/`: MARCO CIFs for recommended candidates.
- `top20_marco_cifs/`: top 20 MARCO CIFs by validation selectivity score.

