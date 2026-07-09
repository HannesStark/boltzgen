# MARCO SRCR Boltz API Nanobody Design Results

This folder contains four independent Boltz protein-design runs against the
human MARCO SRCR domain. Each run generated 100 candidates using the same
epitope-focused input, for 400 unique sequences total.

## Contents

- `design_input.json`: exact Boltz API design payload
- `batch1_run.json`: API record for run `prot_des_T1SPf6J6w0X57HLnuSeD`
- `batch2_run.json`: API record for run `prot_des_HCj2a4VVV6ayP8BVrY0z`
- `batch3_run.json`: API record for run `prot_des_qehG8V7a3X2DT4o8kkY1`
- `batch4_run.json`: API record for run `prot_des_Pl9menCLbQFCVWkWlXwK`
- `all_400_candidates.csv`: sequences and all returned candidate metrics
- `structures/batch1`: 100 predicted target-binder CIF structures from batch 1
- `structures/batch2`: 100 predicted target-binder CIF structures from batch 2
- `structures/batch3`: 100 predicted target-binder CIF structures from batch 3
- `structures/batch4`: 100 predicted target-binder CIF structures from batch 4

Each CIF filename begins with the candidate ID used in
`all_400_candidates.csv`, allowing downstream scripts to join structures,
sequences, and metrics directly. Raw archives, PAE arrays, and duplicate
metadata files remain downloadable from Boltz using the run IDs above.

## Design

- Target: human MARCO SRCR domain, chain A
- Binder modality: nanobody, chain B
- Length: 110-140 residues
- Number generated: 400
- Excluded amino acid: cysteine
- Excluded motifs: `NXS`, `NXT`
- Maximum hydrophobic fraction: 0.42
- Estimated API cost: USD 10.00 total

The epitope residues use zero-based indices:

`5, 6, 8, 10, 12, 13, 15, 18, 19, 20, 48, 50, 52, 96, 99`

## Results Summary

All 400 sequences were unique. Using ipTM at least 0.8 and minimum interaction
PAE at most 4 A, batch 1 produced 29 strong structural candidates, batch 2
produced 18, batch 3 produced 13, and batch 4 produced 19.

Selected VHH-like candidates:

| Candidate | Batch | ipTM | Min interaction PAE (A) | Structure confidence |
| --- | ---: | ---: | ---: | ---: |
| `pres_EZoV78yHit3sQ3rBaLdK` | 1 | 0.931 | 0.990 | 0.724 |
| `pres_WRopBpbTU8i4ch8ndiLs` | 1 | 0.907 | 1.250 | 0.675 |
| `pres_VqiQdwX3YTMTrWKfBVdk` | 1 | 0.896 | 1.581 | 0.593 |
| `pres_ijIn0vkzfn22nbFN05AP` | 2 | 0.893 | 1.229 | 0.723 |
| `pres_hOoj8NGsNH0pksAZaOuZ` | 1 | 0.890 | 1.925 | 0.596 |
| `pres_CjGVTAWLGlPGMKI6HoZi` | 2 | 0.889 | 1.713 | 0.690 |
| `pres_QVJRIqO3wuFQh4hmHHDT` | 2 | 0.894 | 1.850 | 0.633 |
| `pres_7tckIrdh48jvmrRIDKW2` | 4 | 0.941 | 0.711 | 0.727 |
| `pres_vG1sxNuCwnvUtHurTTvP` | 3 | 0.932 | 0.943 | 0.615 |
| `pres_Bzl9sL1K1FTPMmUmSBfb` | 3 | 0.923 | 1.215 | 0.728 |
| `pres_jYILUUMrIZoLcT8j18vn` | 3 | 0.909 | 1.457 | 0.716 |

`pres_7tckIrdh48jvmrRIDKW2` is the leading structural candidate based on its
combined ipTM, interaction PAE, and structure confidence. Binding confidence
remains low, so it should be treated as a computational lead rather than a
validated binder.

## Interpretation

ipTM measures confidence in the relative placement of target and binder
chains. Higher values indicate a more confidently predicted complex.
Interaction PAE is the predicted positional error across the interface in
angstroms; lower values are better.

These metrics assess model confidence and geometry, not experimental affinity
or specificity. Candidates should undergo interface inspection, independent
structure-and-binding predictions, developability screening, cross-reactivity
assessment, and experimental binding validation before use.
