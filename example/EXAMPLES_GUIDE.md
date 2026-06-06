# BoltzGen Examples Guide

This guide provides a structured overview of all available examples and practical guidance for creating your own design specifications.

## Quick Start

1. Pick an example matching your use case from the table below
2. Copy the YAML file to your working directory
3. **Validate first:** `boltzgen check your_design.yaml` (always run this before generation!)
4. Generate designs: `boltzgen run your_design.yaml --output results/`

> **Important:** Always run `boltzgen check` before `boltzgen run`. This validates your YAML syntax, verifies residue indexing, and generates a structure file showing the binding site designation. View the output in Molstar to confirm your binding sites are correctly specified.

---

## Example Overview

### By Category

| Example                                                                                          | Category       | Difficulty   | Description                                              |
| ------------------------------------------------------------------------------------------------ | -------------- | ------------ | -------------------------------------------------------- |
| [vanilla_peptide_with_target_binding_site](vanilla_peptide_with_target_binding_site/)               | Peptide        | Beginner     | Basic peptide design against a defined binding site      |
| [vanilla_protein](vanilla_protein/)                                                                 | Protein        | Beginner     | Standard protein binder design                           |
| [inverse_folding](inverse_folding/)                                                                 | Protein        | Beginner     | Sequence optimization for fixed backbone                 |
| [peptide_against_disordered_region_of_protein](peptide_against_disordered_region_of_protein/)       | Peptide        | Intermediate | Target flexible/disordered protein regions               |
| [peptide_against_specific_site_on_ragc](peptide_against_specific_site_on_ragc/)                     | Peptide        | Intermediate | Site-specific peptide targeting RagC                     |
| [binding_disordered_peptides](binding_disordered_peptides/)                                         | Peptide        | Intermediate | Design binders for intrinsically disordered peptides     |
| [binding_disordered_regions_of_proteins](binding_disordered_regions_of_proteins/)                   | Protein        | Intermediate | Target flexible regions within folded proteins           |
| [disulfide_peptide_with_betahairpin_conditioning](disulfide_peptide_with_betahairpin_conditioning/) | Peptide        | Intermediate | Constrained peptide with disulfide bond and beta-hairpin |
| [nanobody](nanobody/)                                                                               | Antibody       | Intermediate | Single-domain antibody (VHH) design                      |
| [nanobody_scaffolds](nanobody_scaffolds/)                                                           | Antibody       | Intermediate | Nanobody scaffold variants and templates                 |
| [fab_scaffolds](fab_scaffolds/)                                                                     | Antibody       | Intermediate | Antibody Fab fragment scaffolds                          |
| [fab_targets](fab_targets/)                                                                         | Antibody       | Intermediate | Fab-based targeting examples                             |
| [protein_binding_small_molecule](protein_binding_small_molecule/)                                   | Small Molecule | Intermediate | Protein designed to bind small molecules                 |
| [small_molecule_from_file_and_smiles](small_molecule_from_file_and_smiles/)                         | Small Molecule | Intermediate | Specifying ligands via CCD codes or SMILES               |
| [streptavidin_partially_flexible_target](streptavidin_partially_flexible_target/)                   | Protein        | Intermediate | Handling partially flexible target regions               |
| [cyclotide](cyclotide/)                                                                             | Cyclic         | Advanced     | Knotted cyclic peptide design                            |
| [cyclic_against_hiv_antibody_site](cyclic_against_hiv_antibody_site/)                               | Cyclic         | Advanced     | Cyclic peptide targeting HIV antibody epitope            |
| [cylcic_against_kras_with_specific_site](cylcic_against_kras_with_specific_site/)                   | Cyclic         | Advanced     | Site-specific cyclic peptide against KRAS                |
| [double_disulfide_peptide_against_specific_site](double_disulfide_peptide_against_specific_site/)   | Peptide        | Advanced     | Highly constrained peptide with two disulfide bonds      |
| [nanobody_against_penguinpox](nanobody_against_penguinpox/)                                         | Antibody       | Advanced     | Nanobody targeting viral protein                         |
| [denovo_zinc_finger_against_dna](denovo_zinc_finger_against_dna/)                                   | Protein        | Advanced     | De novo zinc finger design for DNA binding               |
| [helicon_against_peptide_in_pmhc](helicon_against_peptide_in_pmhc/)                                 | Peptide        | Advanced     | Helical peptide targeting peptide-MHC complex            |
| [hard_targets](hard_targets/)                                                                       | Various        | Expert       | Challenging binding targets for benchmarking             |

### Category Summary

| Category                 | Count | Use Case                                            |
| ------------------------ | ----- | --------------------------------------------------- |
| **Peptide**        | 7     | Short therapeutic peptides (10-50 residues)         |
| **Protein**        | 5     | Larger protein binders (50+ residues)               |
| **Antibody**       | 5     | Nanobodies, Fabs, and antibody fragments            |
| **Cyclic**         | 3     | Constrained cyclic peptides with enhanced stability |
| **Small Molecule** | 2     | Designs involving small molecule ligands            |
| **Various**        | 1     | Benchmark cases (`hard_targets`)                  |

### Difficulty Guide

| Level                  | Description                                                      | Recommended For                |
| ---------------------- | ---------------------------------------------------------------- | ------------------------------ |
| **Beginner**     | Standard designs with clear binding sites                        | New users learning BoltzGen    |
| **Intermediate** | Additional constraints or specialized targets                    | Users familiar with basics     |
| **Advanced**     | Complex constraints, multiple disulfides, or challenging targets | Experienced users              |
| **Expert**       | Benchmark cases with known difficulty                            | Method development and testing |

---

## Validation Guide

### The `boltzgen check` Command

> **Always validate your design file before running generation.** This catches indexing errors and lets you visually confirm binding site placement.

```bash
# Step 1: Check syntax and generate binding site visualization
boltzgen check your_design.yaml

# Step 2: Open the generated .cif file in Molstar
# Verify binding residues (shown in color) match your intent

# Step 3: Only then run generation
boltzgen run your_design.yaml --output results/
```

The `check` command:

- Validates YAML syntax and structure
- Verifies residue indices exist in the target structure
- **Verifies that referenced `.cif` files exist** (relative to YAML location)
- Generates an **output `.cif` file** showing binding site designation (colored residues)
- Reports any unresolved residues or atoms

### Understanding Check Output

- Open the generated `.cif` in Molstar
- Binding residues (`B`) are highlighted - verify these match your intended interface
- Non-binding residues (`N`) should NOT be at the interface
- If residues are "unresolved", they may be disordered in the structure (no coordinates)

---

## File Organization

Keep your YAML and structure files together:

```
my_design/
├── design.yaml          # Your design specification
├── target.cif           # Target structure (referenced in YAML)
└── README.md            # Optional: explain your design
```

When running `boltzgen check` or `boltzgen run`, execute from the directory containing your files, or provide the full path to the YAML.

---

## Common Patterns

| Design Goal           | Key Settings                                            |
| --------------------- | ------------------------------------------------------- |
| Simple peptide binder | `sequence: "15..25"`, define target `binding_types` |
| Cyclic peptide        | Add disulfide `bond` constraint between termini       |
| Nanobody              | Use `nanobody_scaffolds` template, modify CDR loops   |
| Small molecule binder | Add `ligand` entity with `ccd` or `smiles`        |
