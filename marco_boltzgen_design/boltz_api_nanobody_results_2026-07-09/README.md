# MARCO SRCR Boltz API Nanobody Design Results

This folder contains ten independent Boltz protein-design runs against the
human MARCO SRCR domain. Each run generated 100 candidates using the same
epitope-focused input, for 1000 unique sequences total.

## Contents

- `design_input.json`: exact Boltz API design payload
- `batch1_run.json`: API record for run `prot_des_T1SPf6J6w0X57HLnuSeD`
- `batch2_run.json`: API record for run `prot_des_HCj2a4VVV6ayP8BVrY0z`
- `batch3_run.json`: API record for run `prot_des_qehG8V7a3X2DT4o8kkY1`
- `batch4_run.json`: API record for run `prot_des_Pl9menCLbQFCVWkWlXwK`
- `batch5_run.json`: API record for run `prot_des_mm6RoTNfK9OVR5QU8Lh7`
- `batch6_run.json`: API record for run `prot_des_9e6ndS7E7lryPNsJvXmX`
- `batch7_run.json`: API record for run `prot_des_IECrsjLYDcQXwyXM5gbT`
- `batch8_run.json`: API record for run `prot_des_Lw9wu4FW4tUTem8Z8j1k`
- `batch9_run.json`: API record for run `prot_des_oZbP6JEFwyOgrbNA2f4d`
- `batch10_run.json`: API record for run `prot_des_UFR4Wccv5T3FzgARBdVZ`
- `all_1000_candidates.csv`: sequences and all returned candidate metrics
- `all_400_candidates.csv`: earlier 400-candidate snapshot retained for traceability
- `structures/batch1`: 100 predicted target-binder CIF structures from batch 1
- `structures/batch2`: 100 predicted target-binder CIF structures from batch 2
- `structures/batch3`: 100 predicted target-binder CIF structures from batch 3
- `structures/batch4`: 100 predicted target-binder CIF structures from batch 4
- `structures/batch5`: 100 predicted target-binder CIF structures from batch 5
- `structures/batch6`: 100 predicted target-binder CIF structures from batch 6
- `structures/batch7`: 100 predicted target-binder CIF structures from batch 7
- `structures/batch8`: 100 predicted target-binder CIF structures from batch 8
- `structures/batch9`: 100 predicted target-binder CIF structures from batch 9
- `structures/batch10`: 100 predicted target-binder CIF structures from batch 10

Each CIF filename begins with the candidate ID used in
`all_1000_candidates.csv`, allowing downstream scripts to join structures,
sequences, and metrics directly. Raw archives, PAE arrays, and duplicate
metadata files remain downloadable from Boltz using the run IDs above.

## Design

- Target: human MARCO SRCR domain, chain A
- Binder modality: nanobody, chain B
- Length: 110-140 residues
- Number generated: 1000
- Excluded amino acid: cysteine
- Excluded motifs: `NXS`, `NXT`
- Maximum hydrophobic fraction: 0.42
- Estimated API cost: USD 25.00 total

The epitope residues use zero-based indices:

`5, 6, 8, 10, 12, 13, 15, 18, 19, 20, 48, 50, 52, 96, 99`

## Results Summary

All 1000 sequences were unique. Using ipTM at least 0.8 and minimum interaction
PAE at most 4 A, batch 1 produced 29, batch 2 produced 18, batch 3 produced
13, batch 4 produced 19, batch 5 produced 16, batch 6 produced 11, batch 7
produced 18, batch 8 produced 20, batch 9 produced 17, and batch 10 produced
15 strong structural candidates.

Selected VHH-like candidates ranked primarily by ipTM, then interaction PAE
and structure confidence:

| Candidate | Batch | ipTM | Min interaction PAE (A) | Structure confidence |
| --- | ---: | ---: | ---: | ---: |
| `pres_6G1B2n1xMy2MZ4HXyb46` | 7 | 0.955 | 0.616 | 0.749 |
| `pres_EGlRqBdcBYobvbQ903X4` | 7 | 0.943 | 0.907 | 0.548 |
| `pres_7tckIrdh48jvmrRIDKW2` | 4 | 0.941 | 0.711 | 0.727 |
| `pres_i94bYppgS2dMhtAbEAQD` | 8 | 0.940 | 0.876 | 0.743 |
| `pres_vG1sxNuCwnvUtHurTTvP` | 3 | 0.932 | 0.943 | 0.615 |
| `pres_EZoV78yHit3sQ3rBaLdK` | 1 | 0.931 | 0.990 | 0.724 |
| `pres_GUg59SorgVYbEmlLD4hi` | 1 | 0.925 | 0.932 | 0.476 |
| `pres_rgETTPRI1vpWt51ARqyH` | 6 | 0.924 | 1.189 | 0.752 |
| `pres_Bzl9sL1K1FTPMmUmSBfb` | 4 | 0.923 | 1.215 | 0.728 |
| `pres_LwyRkhXUVubUUYp8cPBB` | 7 | 0.922 | 0.630 | 0.680 |
| `pres_8sfHwDgO0GXSJ7cKtx1H` | 4 | 0.918 | 1.299 | 0.618 |
| `pres_FCO44cl3nIAQk5GgFFdv` | 5 | 0.917 | 1.153 | 0.552 |
| `pres_7MCABk0RJjhHLkskiGrd` | 2 | 0.916 | 1.452 | 0.043 |
| `pres_o4nxcGF6b9Q16yWi0ska` | 10 | 0.913 | 0.881 | 0.682 |
| `pres_cf889qmcdaFumH1ByqNa` | 6 | 0.912 | 1.447 | 0.726 |

`pres_6G1B2n1xMy2MZ4HXyb46` is the leading structural candidate based on this
combined ranking. Binding confidence remains low across this design set, so
these should be treated as computational leads rather than validated binders.

## Interpretation

ipTM measures confidence in the relative placement of target and binder
chains. Higher values indicate a more confidently predicted complex.
Interaction PAE is the predicted positional error across the interface in
angstroms; lower values are better.

These metrics assess model confidence and geometry, not experimental affinity
or specificity. Candidates should undergo interface inspection, independent
structure-and-binding predictions, developability screening, cross-reactivity
assessment, and experimental binding validation before use.
