# MARCO Focused Design Batch 2026-07-21

Focused protein-design runs around the best-performing settings from prior MARCO nanobody design and validation.

Planned jobs:

| variant | designs | hydrophobic cap |
|---|---:|---:|
| broad_original_neighbors_h036 | 500 | 0.36 |
| nterm_basic_patch_h048 | 500 | 0.48 |
| cterm_acidic_patch_h048 | 500 | 0.48 |

Total planned designs: 1500. At the recent estimated rate of $0.025/design, expected cost is about $37.50 before final API estimate confirmation.

## Run Completion and Results

All three 500-design focused Boltz API jobs completed successfully. Estimated cost was `$37.50` total.

Generated result tables:

- `all_1500_focused_candidates.csv`: all focused candidates and metrics.
- `strong_focused_candidates.csv`: candidates meeting loose or strict confidence thresholds.
- `variant_summary.csv`: per-setting counts and top candidates.
- `top_focused_cifs/`: top 100 focused-design CIF files by tier-aware ranking.

Per-setting strong counts:

| variant | total | A+ | A | B | strong loose | strong total | best candidate | best ipTM | best PAE |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| broad_original_neighbors_h036 | 500 | 0 | 3 | 21 | 64 | 88 | pres_I45QWRzKrOXi0srlUSJj | 0.909 | 1.162 |
| nterm_basic_patch_h048 | 500 | 1 | 4 | 17 | 66 | 88 | pres_jTG7v5zEhhhpeEHFclpN | 0.936 | 0.927 |
| cterm_acidic_patch_h048 | 500 | 1 | 3 | 25 | 57 | 86 | pres_dg0cFSW9o5McXscKEzHX | 0.946 | 0.814 |

Top focused candidates are listed in `strong_focused_candidates.csv`; the first 100 CIFs have been copied into `top_focused_cifs/` for rapid structural review.
