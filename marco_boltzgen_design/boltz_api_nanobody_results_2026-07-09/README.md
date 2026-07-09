# MARCO SRCR Boltz API Nanobody Design Results

This folder contains two independent Boltz protein-design runs against the
human MARCO SRCR domain. Each run generated 100 candidates using the same
epitope-focused input, for 200 unique sequences total.

## Contents

- `design_input.json`: exact Boltz API design payload
- `batch1_run.json`: API record for run `prot_des_T1SPf6J6w0X57HLnuSeD`
- `batch2_run.json`: API record for run `prot_des_HCj2a4VVV6ayP8BVrY0z`
- `all_200_candidates.csv`: sequences and all returned candidate metrics

Raw archives and CIF files are not committed here because they add roughly
113 MB. They remain downloadable from Boltz using the run IDs above.

## Design

- Target: human MARCO SRCR domain, chain A
- Binder modality: nanobody, chain B
- Length: 110-140 residues
- Number generated: 200
- Excluded amino acid: cysteine
- Excluded motifs: `NXS`, `NXT`
- Maximum hydrophobic fraction: 0.42
- Estimated API cost: USD 5.00 total

The epitope residues use zero-based indices:

`5, 6, 8, 10, 12, 13, 15, 18, 19, 20, 48, 50, 52, 96, 99`

## Results Summary

All 200 sequences were unique. Batch 1 produced 29 candidates with ipTM at
least 0.8 and minimum interaction PAE at most 4 A. Batch 2 produced 18 using
the same thresholds.

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

`pres_EZoV78yHit3sQ3rBaLdK` is the leading computational candidate based on
its combined ipTM, interaction PAE, structure confidence, and binding
confidence.

## Interpretation

ipTM measures confidence in the relative placement of target and binder
chains. Higher values indicate a more confidently predicted complex.
Interaction PAE is the predicted positional error across the interface in
angstroms; lower values are better.

These metrics assess model confidence and geometry, not experimental affinity
or specificity. Candidates should undergo interface inspection, independent
structure-and-binding predictions, developability screening, cross-reactivity
assessment, and experimental binding validation before use.
