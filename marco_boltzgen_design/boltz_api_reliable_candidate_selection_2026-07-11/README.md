# MARCO SRCR Reliable Nanobody Candidate Selection

Compared the original 1000-candidate panel with the diversified 1200-candidate panel, for 2200 total unique sequences.

## Selection Rules

- A+: ipTM >= 0.93, interaction PAE <= 1.0 A, structure confidence >= 0.74
- A: ipTM >= 0.90, interaction PAE <= 1.5 A, structure confidence >= 0.70
- B: ipTM >= 0.85, interaction PAE <= 2.5 A, structure confidence >= 0.60

The final 24-candidate shortlist is diversity-aware: selected candidates are ranked by a composite reliability score, while keeping pairwise positional sequence identity below 70% to reduce redundancy.

## Counts

- Total candidates: 2200
- Unique sequences: 2200
- A+ candidates: 6
- A candidates: 21
- B candidates: 113
- Strict A/A+ pool: 27
- Diverse final shortlist: 24

## Recommended Diverse Shortlist

| Rank | Candidate | Source | Epitope set | h cap | ipTM | PAE (A) | Struct conf | Hydrophobic fraction |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `pres_6oROWx3G7YCh4zwEZCKx` | `diversified_1200` | `broad_original_neighbors` | 0.36 | 0.945 | 0.798 | 0.787 | 0.325 |
| 2 | `pres_t9BJw8MufJhEMNUBxgAD` | `diversified_1200` | `nterm_basic_patch` | 0.48 | 0.942 | 0.868 | 0.802 | 0.365 |
| 3 | `pres_6G1B2n1xMy2MZ4HXyb46` | `original_1000` | `original_epitope` | 0.42 | 0.955 | 0.616 | 0.749 | 0.386 |
| 4 | `pres_TGMZaZPoRcVRPwp4ySl4` | `diversified_1200` | `broad_original_neighbors` | 0.48 | 0.942 | 0.776 | 0.775 | 0.393 |
| 5 | `pres_5vdSDmUoQUpmffx3TuZ7` | `diversified_1200` | `cterm_acidic_patch` | 0.48 | 0.939 | 0.917 | 0.758 | 0.379 |
| 6 | `pres_i94bYppgS2dMhtAbEAQD` | `original_1000` | `original_epitope` | 0.42 | 0.940 | 0.876 | 0.743 | 0.429 |
| 7 | `pres_7tckIrdh48jvmrRIDKW2` | `original_1000` | `original_epitope` | 0.42 | 0.941 | 0.711 | 0.727 | 0.336 |
| 8 | `pres_SfWyhjrHayJss4IaEaRK` | `diversified_1200` | `cterm_acidic_patch` | 0.42 | 0.943 | 1.042 | 0.766 | 0.363 |
| 9 | `pres_1LLmyU2Y1EsuD0uiaErt` | `diversified_1200` | `mid_srcr_surface_patch` | 0.42 | 0.946 | 0.850 | 0.704 | 0.315 |
| 10 | `pres_RgGWmEVQeEVgBJLq9i8p` | `diversified_1200` | `mid_srcr_surface_patch` | 0.42 | 0.931 | 1.070 | 0.782 | 0.374 |
| 11 | `pres_WwzPirvtkZE1J06gh7h3` | `diversified_1200` | `mid_srcr_surface_patch` | 0.36 | 0.933 | 0.879 | 0.736 | 0.349 |
| 12 | `pres_gEG1pxmYdznCzAiwtiy4` | `diversified_1200` | `cterm_acidic_patch` | 0.48 | 0.928 | 1.023 | 0.760 | 0.354 |
| 13 | `pres_NGPaSbFqqw6ZhNw5FMqa` | `diversified_1200` | `cterm_acidic_patch` | 0.48 | 0.926 | 0.853 | 0.754 | 0.386 |
| 14 | `pres_atX04pGTXmAjskypxY49` | `diversified_1200` | `cterm_acidic_patch` | 0.42 | 0.928 | 1.144 | 0.753 | 0.345 |
| 15 | `pres_rgETTPRI1vpWt51ARqyH` | `original_1000` | `original_epitope` | 0.42 | 0.924 | 1.189 | 0.752 | 0.328 |
| 16 | `pres_W1yhUy8pQWn4TuThs9oa` | `diversified_1200` | `nterm_basic_patch` | 0.48 | 0.936 | 1.031 | 0.760 | 0.485 |
| 17 | `pres_qmvG8YvpiJb6mKI0w9zc` | `diversified_1200` | `broad_original_neighbors` | 0.36 | 0.927 | 1.159 | 0.762 | 0.395 |
| 18 | `pres_EZoV78yHit3sQ3rBaLdK` | `original_1000` | `original_epitope` | 0.42 | 0.931 | 0.990 | 0.724 | 0.392 |
| 19 | `pres_du8dYk7n3pAwDj5bUdDE` | `diversified_1200` | `cterm_acidic_patch` | 0.48 | 0.920 | 1.211 | 0.748 | 0.388 |
| 20 | `pres_Bzl9sL1K1FTPMmUmSBfb` | `original_1000` | `original_epitope` | 0.42 | 0.923 | 1.215 | 0.728 | 0.439 |
| 21 | `pres_mjNQuh1ApKCHkGoj4IyL` | `original_1000` | `original_epitope` | 0.42 | 0.911 | 1.196 | 0.712 | 0.358 |
| 22 | `pres_USYAeeYrWvEt6YOeh28x` | `diversified_1200` | `cterm_acidic_patch` | 0.42 | 0.907 | 1.007 | 0.701 | 0.382 |
| 23 | `pres_cf889qmcdaFumH1ByqNa` | `original_1000` | `original_epitope` | 0.42 | 0.912 | 1.447 | 0.726 | 0.365 |
| 24 | `pres_PRVOBeiU3zvrDKHXaGnu` | `diversified_1200` | `broad_original_neighbors` | 0.36 | 0.902 | 1.156 | 0.720 | 0.383 |

## Files

- `highly_reliable_diverse_24.csv`: final diversity-aware shortlist with `selected_cif_path` for each copied CIF
- `strict_ranked_a_pool.csv`: all strict A/A+ candidates without diversity filtering
- `selected_cifs/`: CIF copies for the 24 recommended candidates
- `summary.json`: count and distribution summary

## Recommended Next Step

Use these 24 candidates for independent structure-and-binding validation, counter-target screening, and manual interface/developability inspection before experimental ordering.
