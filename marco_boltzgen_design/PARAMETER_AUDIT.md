# MARCO spec parameter audit (2026-05-18)

Checked README-defined hotspot parameters against spec YAMLs.

- **mouse_marco_nanobody_setA_so4_pocket.yaml**: PASS (expected `12,14,21,50,56,58,78,89`; found 12,14,21,50,56,58,78,89)
- **human_marco_nanobody_setA_so4_pocket.yaml**: PASS (expected `12,14,21,50,56,58,78,89`; found 12,14,21,50,56,58,78,89)
- **human_marco_nanobody_setB_patent_epitope.yaml**: PASS (expected `33,35,56,70,82,88,90,92,94`; found 33,35,56,70,82,88,90,92,94)
- **crossreactive_marco_nanobody_setC_hybrid.yaml**: PASS (expected `35,50,56,58,70,78,82,88,89,90,92,94|12,14,21,50,56,58,78,89`; found human=35,50,56,58,70,78,82,88,89,90,92,94; mouse=12,14,21,50,56,58,78,89)

## Runtime validation command from README
- `boltzgen check <spec.yaml>` could not be executed in this environment because `boltzgen` is not on PATH.
