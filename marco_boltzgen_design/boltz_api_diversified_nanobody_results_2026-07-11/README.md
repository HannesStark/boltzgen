# MARCO SRCR Diversified Boltz API Nanobody Design Results

This folder contains a diversified Boltz protein-design panel against the
human MARCO SRCR domain. The panel crosses four alternative epitope residue
sets with three maximum hydrophobicity constraints, producing 12 independent
100-candidate runs and 1200 unique sequences total.

## Contents

- `manifest.json`: run manifest with input paths, run directories, idempotency keys, epitope sets, and hydrophobicity caps
- `design_inputs/`: exact Boltz API design payload for each variant
- `run_metadata/`: Boltz API run record for each variant
- `all_1200_candidates.csv`: sequences, metrics, and relative CIF paths for every candidate
- `strong_candidates.csv`: candidates with ipTM at least 0.8 and minimum interaction PAE at most 4 A
- `variant_summary.csv`: per-variant run IDs, counts, and strong-candidate totals
- `structures/<variant>/`: predicted target-binder CIF files for each design variant

Each CIF filename begins with the candidate ID used in the CSV files, allowing
downstream scripts to join structures, sequences, and metrics directly.

## Design Matrix

Target sequence is the human MARCO SRCR domain on chain A:

`SVSVRIVGSSNRGRAEVYYSGTWGTICDDEWQNSDAIVFCRMLGYSKGRALYKVGAGTGQIWLDNVQCRGTESTLWSCTKNSWGHHDCSHEEDAGVECSV`

Binder settings are shared across all variants:

- Binder modality: nanobody, chain B
- Length: 110-140 residues
- Excluded amino acid: cysteine
- Excluded motifs: `NXS`, `NXT`
- Number generated per variant: 100
- Total generated: 1200
- Estimated API cost: USD 30.00 total

Epitope residue sets use zero-based indices:

- `nterm_basic_patch`: `4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22`
- `mid_srcr_surface_patch`: `45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58, 59, 60, 61`
- `cterm_acidic_patch`: `76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 91, 94, 96, 99`
- `broad_original_neighbors`: `5, 6, 8, 10, 12, 13, 15, 18, 19, 20, 48, 49, 50, 52, 53, 96, 97, 99`

Hydrophobicity caps tested: 0.36, 0.42, and 0.48.

## Variant Summary

| Variant | Epitope set | Hydrophobicity cap | Strong candidates | Run ID |
| --- | --- | ---: | ---: | --- |
| `marco_srcr_diversified_nterm_basic_patch_h036_100` | `nterm_basic_patch` | 0.36 | 24 | `prot_des_861DPZFORJErqjwJ3Z72` |
| `marco_srcr_diversified_nterm_basic_patch_h042_100` | `nterm_basic_patch` | 0.42 | 21 | `prot_des_IemcRs3g0mYMTyax5Mzm` |
| `marco_srcr_diversified_nterm_basic_patch_h048_100` | `nterm_basic_patch` | 0.48 | 20 | `prot_des_hJIGXtoiNWqIloMhDH96` |
| `marco_srcr_diversified_mid_srcr_surface_patch_h036_100` | `mid_srcr_surface_patch` | 0.36 | 19 | `prot_des_wNUIam3X4PoUeqRsNkjc` |
| `marco_srcr_diversified_mid_srcr_surface_patch_h042_100` | `mid_srcr_surface_patch` | 0.42 | 22 | `prot_des_DQ5W08YGbYWEYqLw8Rln` |
| `marco_srcr_diversified_mid_srcr_surface_patch_h048_100` | `mid_srcr_surface_patch` | 0.48 | 20 | `prot_des_5wyfuMIxE4ZjEkhUIFcD` |
| `marco_srcr_diversified_cterm_acidic_patch_h036_100` | `cterm_acidic_patch` | 0.36 | 18 | `prot_des_d71Noa1SwYvLbNsWABtn` |
| `marco_srcr_diversified_cterm_acidic_patch_h042_100` | `cterm_acidic_patch` | 0.42 | 16 | `prot_des_V5Ov7ALpqztSzzD8oFbP` |
| `marco_srcr_diversified_cterm_acidic_patch_h048_100` | `cterm_acidic_patch` | 0.48 | 21 | `prot_des_sb1kygtxa77XIKUCJsDg` |
| `marco_srcr_diversified_broad_original_neighbors_h036_100` | `broad_original_neighbors` | 0.36 | 25 | `prot_des_cjHflNF73jUzBWN7wlu5` |
| `marco_srcr_diversified_broad_original_neighbors_h042_100` | `broad_original_neighbors` | 0.42 | 15 | `prot_des_VhTTjvUf4BbTrevsIIBA` |
| `marco_srcr_diversified_broad_original_neighbors_h048_100` | `broad_original_neighbors` | 0.48 | 22 | `prot_des_AWhs2uxehxikffHyP1TV` |

All 1200 sequences were unique. Across the full diversified panel, 243 candidates met the structural-confidence filter of ipTM at least 0.8 and minimum interaction PAE at most 4 A.

## Top Candidates By ipTM

| Candidate | Epitope set | Hydrophobicity cap | ipTM | Min interaction PAE (A) | Structure confidence |
| --- | --- | ---: | ---: | ---: | ---: |
| `pres_4kAUj30P9llaTRNIlYic` | `mid_srcr_surface_patch` | 0.48 | 0.951 | 0.934 | 0.229 |
| `pres_1LLmyU2Y1EsuD0uiaErt` | `mid_srcr_surface_patch` | 0.42 | 0.946 | 0.850 | 0.704 |
| `pres_6oROWx3G7YCh4zwEZCKx` | `broad_original_neighbors` | 0.36 | 0.945 | 0.798 | 0.787 |
| `pres_SfWyhjrHayJss4IaEaRK` | `cterm_acidic_patch` | 0.42 | 0.943 | 1.042 | 0.766 |
| `pres_TGMZaZPoRcVRPwp4ySl4` | `broad_original_neighbors` | 0.48 | 0.942 | 0.776 | 0.775 |
| `pres_t9BJw8MufJhEMNUBxgAD` | `nterm_basic_patch` | 0.48 | 0.942 | 0.868 | 0.802 |
| `pres_CWZqR9eYDeDFdEHfnzIc` | `nterm_basic_patch` | 0.42 | 0.940 | 0.857 | 0.679 |
| `pres_5vdSDmUoQUpmffx3TuZ7` | `cterm_acidic_patch` | 0.48 | 0.939 | 0.917 | 0.758 |
| `pres_W1yhUy8pQWn4TuThs9oa` | `nterm_basic_patch` | 0.48 | 0.936 | 1.031 | 0.760 |
| `pres_WwzPirvtkZE1J06gh7h3` | `mid_srcr_surface_patch` | 0.36 | 0.933 | 0.879 | 0.736 |
| `pres_d7frcrBsh31TEu1zC15Y` | `broad_original_neighbors` | 0.48 | 0.932 | 0.953 | 0.571 |
| `pres_RgGWmEVQeEVgBJLq9i8p` | `mid_srcr_surface_patch` | 0.42 | 0.931 | 1.070 | 0.782 |
| `pres_RuatpHaInwt7MFljZFJ9` | `mid_srcr_surface_patch` | 0.42 | 0.929 | 0.599 | 0.433 |
| `pres_gEG1pxmYdznCzAiwtiy4` | `cterm_acidic_patch` | 0.48 | 0.928 | 1.023 | 0.760 |
| `pres_atX04pGTXmAjskypxY49` | `cterm_acidic_patch` | 0.42 | 0.928 | 1.144 | 0.753 |

## Top Candidates By Structure Confidence

| Candidate | Epitope set | Hydrophobicity cap | ipTM | Min interaction PAE (A) | Structure confidence |
| --- | --- | ---: | ---: | ---: | ---: |
| `pres_t9BJw8MufJhEMNUBxgAD` | `nterm_basic_patch` | 0.48 | 0.942 | 0.868 | 0.802 |
| `pres_6oROWx3G7YCh4zwEZCKx` | `broad_original_neighbors` | 0.36 | 0.945 | 0.798 | 0.787 |
| `pres_RgGWmEVQeEVgBJLq9i8p` | `mid_srcr_surface_patch` | 0.42 | 0.931 | 1.070 | 0.782 |
| `pres_TGMZaZPoRcVRPwp4ySl4` | `broad_original_neighbors` | 0.48 | 0.942 | 0.776 | 0.775 |
| `pres_SfWyhjrHayJss4IaEaRK` | `cterm_acidic_patch` | 0.42 | 0.943 | 1.042 | 0.766 |
| `pres_qmvG8YvpiJb6mKI0w9zc` | `broad_original_neighbors` | 0.36 | 0.927 | 1.159 | 0.762 |
| `pres_W1yhUy8pQWn4TuThs9oa` | `nterm_basic_patch` | 0.48 | 0.936 | 1.031 | 0.760 |
| `pres_gEG1pxmYdznCzAiwtiy4` | `cterm_acidic_patch` | 0.48 | 0.928 | 1.023 | 0.760 |
| `pres_5vdSDmUoQUpmffx3TuZ7` | `cterm_acidic_patch` | 0.48 | 0.939 | 0.917 | 0.758 |
| `pres_NGPaSbFqqw6ZhNw5FMqa` | `cterm_acidic_patch` | 0.48 | 0.926 | 0.853 | 0.754 |
| `pres_atX04pGTXmAjskypxY49` | `cterm_acidic_patch` | 0.42 | 0.928 | 1.144 | 0.753 |
| `pres_du8dYk7n3pAwDj5bUdDE` | `cterm_acidic_patch` | 0.48 | 0.920 | 1.211 | 0.748 |
| `pres_WwzPirvtkZE1J06gh7h3` | `mid_srcr_surface_patch` | 0.36 | 0.933 | 0.879 | 0.736 |
| `pres_EJVRt2Pv8mqZqlSCXY7E` | `nterm_basic_patch` | 0.42 | 0.890 | 1.563 | 0.724 |
| `pres_z8M1f1TpacRzffwhpPHp` | `broad_original_neighbors` | 0.48 | 0.901 | 1.513 | 0.721 |

## Interpretation

ipTM measures confidence in the relative placement of target and binder chains.
Higher values indicate a more confidently predicted complex. Interaction PAE is
the predicted positional error across the interface in angstroms; lower values
are better.

These metrics assess model confidence and geometry, not experimental affinity
or specificity. Treat these designs as computational leads. Follow-up should
include interface inspection, independent structure-and-binding predictions,
developability screening, specificity/counter-target prediction, and experimental
binding validation.
