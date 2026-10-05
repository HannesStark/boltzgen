# Sequence analysis of multichain designs

Analysis includes every **protein chain with at least one designed, unpadded
protein token**. Fixed scaffold residues in those chains are included in
full-chain sequence and liability metrics. Nonprotein tokens and padding are
excluded from amino-acid sequence metrics; structural design/target masks are
unchanged. Unknown protein residues remain `X` and produce unknown
hydrophobicity (`NaN`).

Single-chain sequence fields retain their existing strings. For multiple chains,
`designed_sequence` and `designed_chain_sequence` use `:` to separate chains in
feature order. Each component of `designed_chain_sequence` is a separate chain,
not a sequence to synthesize as a fusion. The existing
`designed_sequence_<asym_id>` and `full_sequence_<asym_id>` columns remain
available. Chain suffixes are feature asymmetry IDs, not PDB chain labels.

Liabilities are evaluated separately on each complete protein chain using the
configured panel. `liability_score`, violation totals, severity counts and
per-motif counts sum the independent scans. Per-chain fields use the suffix
`_<asym_id>` and retain 1-based positions within that full-chain sequence.
Multichain aggregate motif positions are `-1` and lengths are `0`, because there
is no shared residue coordinate system. Their details and the violation summary
identify the chain explicitly. No motif, terminus, charge or cysteine heuristic
is evaluated on a concatenation of chains. These sequence heuristics do not
infer interchain disulfide bonds.

`design_chain_hydrophobicity` is the residue-weighted mean of the independently
scored full chains. `design_hydrophobicity` similarly averages scores for each
chain's designed residues. Per-chain versions use `_<asym_id>`. Each score keeps
its own terminal, neighbor and length corrections; the aggregate is a summary,
not a prediction of complex retention. Within a chain, the designed-residue
metric retains its previous treatment of concatenated redesigned segments.
Sequence hydrophobicity is also available when refolding is disabled.

Filtering uses the boundary-preserving designed sequence for deduplication.
Sequence diversity aligns corresponding chains in feature order and divides
summed alignment scores by summed maximum chain lengths. Missing chains have
zero matches; delimiters do not count as residues. This is ordered chain
comparison, not a search over chain permutations. Legacy analysis archives
without stored chain boundaries remain readable; rerun analysis to recover
boundaries for those designs.
CSV loading and merging preserve `NA` as an amino-acid sequence; genuinely empty
sequence cells remain missing, and numeric metrics retain normal missing values.

PDF liability heatmaps scan complete chains independently, matching analysis.
Sequence logos and composition plots are grouped by chain. The existing choice
between a scaffold's full sequence and its designed residues is made separately
for each chain. Nanobody and antibody protocols default to the antibody panel
in both analysis and reporting; explicit step configuration overrides still
apply. Protein/peptide defaults are unchanged.

Optional numbered CDR logos require `abnumber` and its dependencies. If these
are unavailable, the PDF still includes ordinary sequence logos, composition
plots and liability heatmaps. Numbering receives complete per-chain scaffolds.
Composition plots label chains with no recognized amino acids instead of trying
to draw a pie chart with zero total counts.
