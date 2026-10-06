# ESMFold2 interaction scoring

The `protein-anything`, `peptide-anything`, `antibody-anything`, and
`nanobody-anything` protocols
score designed-polymer–target-polymer interactions with the full ESMFold2 2021
checkpoint. The `protein-small_molecule` protocol keeps the existing Boltz2
affinity pipeline and ranking. Protocol selection is explicit: a ligand-only
target in an `*-anything` protocol raises an error directing you to
`protein-small_molecule`.

A polymer chain with residues selected for design belongs to the binder;
fixed polymer chains are target partners. A covalent link does not change
those roles, even when the linked chains are refolded together.

ESMFold2 replaces the interface iPTM and minimum-interface-PAE ranking terms,
including the ranking tie-breaker. Binder pTM, physical metrics, diversity
selection, and Boltz2 complex/binder-only structural checks remain in use.
Consequently, polymer design still needs the Boltz2 structure checkpoint.

## Redesigning complexes without fixed partners

`protein-redesign` uses ESMFold2 even when every chain is redesigned. With two
or more polymer chains, score each chain against all other polymer chains using
minimum directional ipSAE, then take the weakest chain-versus-rest score. The
best of the five samples maximizes that weakest score. Ligands remain folding
context and never become an ipSAE partner. If any polymer is RNA/DNA, these
pooled chain-versus-rest interfaces use the 2 Å d0 floor. Scores from these
mixed-polymer campaigns should not be compared directly with protein-only scores
that use a different normalization.

With one polymer chain, there is no polymer interface: select the sample with
highest native ESMFold2 pTM instead. This remains the native whole-model pTM,
including any modeled cofactor context; it is not labeled as ipSAE.
`esmfold2_score` is the ranking column for this protocol, and
`esmfold2_score_metric` records whether it is `esmfold2_ipsae_min` or
`esmfold2_ptm`. Rank monomer and multichain redesigns in separate campaigns;
the filter rejects mixing these different score types. Redesign scoring does
not accept `scoring_target_chains`; ordinary binder protocols retain that option.

The redesign ranking uses this ESM score and the existing design RMSD term.
Boltz iPTM/pTM columns remain diagnostic. The old Boltz-normalized
`absolute_score` and `structure_confidence` composites are not computed for
polymer protocols because their calibration does not describe ESMFold2 scores.

## Installation and use

Installation and normal use keep the same commands:

```bash
pip install boltzgen
boltzgen run design.yaml --output results --protocol protein-anything
```

BoltzGen includes uv and automatically prepares a cached ESMFold2 runtime before
starting a fresh design run. It obtains Python 3.12 if needed and installs the pinned
dependency set shipped in the wheel. Later runs try the complete local cache
first, without network access. A fresh run checks that this runtime can execute
a CUDA operation before starting design. The main
BoltzGen environment retains its Python >=3.11 support and existing dependencies;
ESMFold2's Torch/CUDA packages are isolated from it. The small-molecule protocol
does not prepare or download the ESMFold2 runtime.

With `--reuse`, runtime setup is deferred until a design actually needs a new
score. A run with complete matching scores can reuse them offline without an
ESMFold2 runtime cache.

First use requires network access and approximately 33 GB of additional disk
space (about 27 GB of ESM weights and 6 GB for the runtime), plus temporary
installation space. This is in addition to BoltzGen's existing downloads.
ESMFold2's upstream
runtime requires a CUDA 13 compatible NVIDIA driver; automatic environment
setup cannot upgrade the host driver. Each worker loads its model once and
scores its shard of designs. `--devices` controls the number of workers over
the visible GPUs. Weights use Hugging Face's ordinary cache (`HF_HOME`), and
the isolated runtime uses uv's cache (`UV_CACHE_DIR`). `--cache` continues to
control Boltz downloads. These environment variables are optional cache-location
controls, not installation steps. For containers, persist both caches if you
want to avoid downloading them again when the container is removed.

Before downloading, BoltzGen prints the resolved checkpoint cache directory
(normally `~/.cache/huggingface/hub`). `HF_HOME` changes the Hugging Face cache
root, while `HF_HUB_CACHE` directly overrides the checkpoint cache directory.
Startup messages distinguish runtime installation, chemical component data,
each checkpoint download, loading weights into memory, and transfer to the GPU.
Download progress bars come from Hugging Face; they may be disabled in some
environments. Loading cached weights can also take several minutes without a
download progress bar. Each completed download prints its local snapshot path,
and a final message announces when the model is ready for scoring.

For an administrator-managed or pre-provisioned offline installation,
`--esmfold2_python /path/to/python` or `BOLTZGEN_ESMFOLD2_PYTHON` remains an optional
override. That interpreter needs the dependencies in
`src/boltzgen/resources/runtime/esmfold2.txt`; it does not need BoltzGen installed.
The worker loads only BoltzGen's own code, with Python environment isolation
enabled, so the parent environment cannot shadow its dependencies.

Pinned artifacts:

| Artifact | Revision |
| --- | --- |
| `esm` | `3.4.1.post1` |
| `biohub/ESMFold2` (full 2021) and CCD | `8fc3ff471022fdce52c77030685eb775de0c00a3` |
| `biohub/ESMC-6B` | `af1602ba7406f521b11bf8f81d52af378cde09e4` |

Defaults are 20 loops, 200 diffusion steps, five samples, LM dropout 0.3,
LM masking 0, and seed 0. No homologous MSA is searched for or loaded. ESMC
encodes complete source chains; the folding model receives only the requested
structural crop and the corresponding ESMC states. No target coordinate
template is supplied to ESMFold2.

## Inference acceleration

See [H200 measurements and reproduction commands](esmfold2-performance.md).

ESMFold2 scoring uses `--esmfold2_acceleration auto` by default. Installation
and runtime dependencies are unchanged. The worker adapts three techniques from
[Anthropic's public optimization kit](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/esmfold2):
reuse the atom-attention mask within an input, transfer the diffusion schedule
to the CPU once, and replay CUDA graphs for the recycling trunk and deterministic
diffusion forward. The native precision and kernel backend remain unchanged.
This is a port of these techniques to the pinned native ESM implementation,
not an installation of the kit's older Transformers stack or its `fast` kernels.

For additional speed, `--esmfold2_acceleration fused` combines the adapter with
ESMFold2's bundled fused Triton/BF16 backend and unchunked pair operations, the
backend used by the public kit's `exact` mode. It needs no additional packages.
This mode can change predicted structures and scores relative to the original
backend, and unchunked operations can use more memory on large inputs. It is an
explicit choice; `auto` keeps the original numerical backend. Each worker fixes
its backend before seeding requests, and results record which backend ran.
Workers score requests sequentially. Programmatic callers must use separate
model instances for concurrent requests.
The pinned fused pair-bias kernel uses 32-bit offsets: crops that would overflow
its indexing are rejected before GPU inference, with instructions to use `auto`
or `off` (at the default five samples and pair width, this is 1,296 tokens or more).

Graph preparation is included in each candidate's scoring time. Graphs and masks
are released between candidates, including candidates with matching shapes.
ESMC still encodes full source sequences before cropping. The checkpoint, loops,
diffusion steps, five samples, random seed, and LM dropout are unchanged. The
stochastic dropout path and batched confidence trunk execute eagerly.

Graphs are attempted for crops up to 256 tokens in `auto` and 512 in `fused`.
Larger inputs retain mask caching and schedule-transfer optimization without
graphs. These conservative limits avoid capture overhead outweighing replay
savings in the measured native workload. Retained masks are
limited to 256 MiB. Unsupported graph captures fall back to eager execution
with a warning; incompatible upstream source disables the adapter. The result
JSON's `execution` field records the effective path and capture/replay counts.
The requested mode and adapter revision are part of the score fingerprint, so
changing the mode invalidates old cached scores. Upgrading from a version without
acceleration metadata also recomputes ESMFold2 scores once, including in `off`
mode; existing design and Boltz2 folding artifacts remain reusable.

Use the original execution path for comparison or troubleshooting:

```bash
boltzgen run design.yaml --output results --protocol protein-anything \
  --esmfold2_acceleration off
```

The small-molecule protocol still uses Boltz2 affinity and never loads this adapter.

To reproduce worker benchmarks from a source checkout using the same isolated
dependencies as scoring:

```bash
uv run --no-project --python 3.12 \
  --with-requirements src/boltzgen/resources/runtime/esmfold2.txt \
  python scripts/benchmark_esmfold2.py --output benchmark-results \
  --modes off auto --repeats 3
```

The benchmark saves all five PAE/coordinate predictions, RNG states, input
requests, effective execution settings, timings, peak GPU memory, and GPU memory
remaining after each request. It counts
full input preparation, ESMC, graph setup, folding, scoring, and result writing;
it reports weight loading separately and never reuses saved scores. The first
call per shape and mode is marked separately from repeated calls. Add
`--deterministic` to compare both arms using deterministic PyTorch algorithms
without disabling LM dropout. Add `--profile` to collect operator totals on a
separate untimed call. Native GPU execution can vary between repeated runs even
at a fixed seed, so use the deterministic comparison for numerical checks.
Benchmark `fused` separately with `--modes fused` and a new output directory, so
the model's numerical backend remains fixed for the lifetime of its process.

## Score definition

For binder protocols, the selected sample has the highest `esmfold2_ipsae_min`,
independent of iPTM.
The score uses PAE strictly below 10 Å, per-residue d0 normalization, and the
**minimum** of design→target and target→design ipSAE. This minimum convention is
deliberate; the original ipSAE implementation also reports a symmetric maximum.
There is no coordinate-distance cutoff. Empty low-PAE interfaces score zero.

RNA/DNA use the reference's 2 Å d0 floor; protein-only interfaces use 1 Å.
Modified polymer residues count once, using their CA/C1′ token, rather than once
per atom. Polymer caps without that representative atom are rejected explicitly.
Ligands and cofactors stay in the ESMFold2 complex but do not contribute
to ipSAE. Explicit covalent connections are preserved by chain, residue, and
atom identity; an unavailable endpoint is an error.
Explicit `leaving_atoms` are carried through the source metadata and removed
from the ESMFold2 structural features, including for side-chain cyclization.
Removing an atom needed for a model residue frame is rejected explicitly.

For multi-chain targets of one polymer type, all target residues form one
partner and all designed-polymer residues form the other. Mixed protein and
nucleic-acid targets require an explicit subset of target chains, so the score
does not silently mix the two normalization conventions:

```bash
boltzgen run design.yaml --output results --protocol protein-anything \
  --config esmfold2_scoring 'scoring_target_chains=[A,B]'
```

Chain IDs here are those in the generated design CIF. Unselected target chains
remain in the folding context. Directional chain-pair scores are also saved.

## Spatial crops and existing output folders

New designs preserve `source_context` as a JSON string in their metadata NPZ.
It contains the complete source residue identities, retained source positions,
and source-chain names. Cropping and `reset_res_index` do not discard that map.
Sequence design and inverse folding substitute the final generated residues
into the preserved full sequence before ESMC encoding.

For each chain included in the modeled complex, spatial selection retains the
full source sequence for ESMC. A replacement is recognized using existing YAML
fields: a contiguous interval explicitly removed by `exclude` must be entirely
designable after applying `design` and `not_design`, and a `design_insertions`
anchor must fall anywhere inside that interval, including either endpoint.
The original residues of that interval are removed from ESMC's sequence and
the sampled insertion is retained with its final generated amino acids.
Multiple insertions inside the same interval remove the old interval only once.
Insertion positions refer to the original input numbering, before other insertions.
Symmetric chains share sampled insertion lengths at the same original position,
even when their insertion entries appear in different orders.
Excluding a whole chain is equivalent to excluding its full residue interval.

Excluded intervals without a matching insertion, or with any non-designable
residue, remain full-sequence context. Include/proximity cropping alone never
declares a replacement. This convention supports replacements and ordinary
spatial crops in the same chain, without additional YAML settings. A spatial
crop that is fully designable and also has an insertion inside it is interpreted
as a replacement by this convention.

Generated substitutions and sampled insertions update the sequence and its
crop-index mapping. Replacement removal shifts indices in the edited sequence;
gaps for other spatial crops remain. `reset_res_index` does not erase that map.
Metadata records pre-removal sequences and selected positions in
`replacement_sources` (after sampling insertions, before generated substitutions).
Only ESMFold2's structural features and the selected ESMC representations are
cropped. Chains omitted entirely from the modeled complex are not encoded.
This happens automatically and requires no additional scaffold YAML settings.

An explicit `fuse` operation assembles the selected segments into one physical
chain. ESMC encodes that assembled sequence, including any inserted linkers,
and its source positions refer to the assembled construct. Metadata labels this
as `context_mode: fused_construct`. Cropping without `fuse` keeps the ordinary
full-source rule above. No additional YAML setting is needed.

To score a different full sequence, provide it as the source input or through
the full-sequence mapping or context sidecar described below. Selection of
residues alone never changes the full-sequence context into a shorter molecule.

Input files must declare their complete polymer sequences. An ATOM-only PDB
does not establish a full sequence. Supply the missing information on the file
entity, for example:

```yaml
entities:
  - file:
      path: cropped_target.pdb
      full_sequences:
        - chain:
            id: A
            sequence: MAGCAG
            # One 1-based full-sequence position per residue in the input chain,
            # before any additional include/exclude selection.
            source_res_indices: [2, 4, 6]
  - protein:
      id: B
      sequence: 60
```

The declared sequence must match those input residues. The same full-source
sequence requirement applies to input files used by `fuse`.
Use complete construct inputs or provide an explicit context sidecar.

Old generated folders lack this information. Regenerate from the original
YAML/files, or provide `source_context_dir/<design_id>.json` and rerun:

```bash
boltzgen run original_design.yaml --output results --protocol protein-anything \
  --steps esmfold2_scoring analysis filtering \
  --config esmfold2_scoring source_context_dir=/path/to/source_context
```

Each sidecar uses the same format as the metadata's `source_context`:

```json
{
  "version": 1,
  "chains": [
    {
      "source_chain": "A",
      "source": "original_target.cif",
      "mol_type": 0,
      "residue_names": ["MET", "ALA", "GLY", "CYS", "ALA", "GLY"],
      "indices": [1, 3, 5],
      "complete": true
    }
  ]
}
```

Include **every chain in generated-CIF order**, including the designed chains
and nonpolymers. `indices` are zero-based, increasing positions in the complete
`residue_names` list, with one entry per retained residue. `mol_type` is 0 for
protein, 1 for DNA, 2 for RNA, and 3 for nonpolymer. The example above illustrates
one chain only. Supply the actual experimental source sequences and mappings;
do not substitute a concatenated crop or a canonical UniProt sequence.

## Outputs, resume, and validation

`esmfold2_scores/<design_id>.input.json` records the exact input and inference
settings. The corresponding `.json` records the selected sample, all five
scores, directional chain-pair scores, sequence hashes, pinned revisions, and
input audit. `.npz` stores all sample PAEs and the selected structure/confidence;
`.cif` stores the selected structure with full-source residue numbering.
The existing top/diverse structure folders still contain the Boltz2 refolds
used for geometry checks; the structures underlying ipSAE are in `esmfold2_scores`.

For binder protocols, the three `esmfold2_*ipsae*` columns enter the
aggregate/final CSVs. Redesign scores and their metric identity enter these
CSVs under the columns described above; per-chain-versus-rest scores for every
sample are retained in the result JSON. Existing
Boltz2 confidence columns remain diagnostic and are not PPI ranking terms.
The summary report presents ESMFold2 ipSAE as the interaction score.

`--reuse` only reuses ESMFold2 results whose complete input/settings hash matches.
Analysis is refreshed for polymer protocols so an old Boltz-only analysis
cannot hide the new score. Missing, non-finite, or stale ESMFold2 scores raise
errors; there is no implicit fallback to Boltz2 iPTM.

CPU regression checks:

```bash
uv run --extra test pytest tests/test_esmfold2_scoring.py \
  tests/test_esmfold2_source_context.py tests/test_esmfold2_source_guard.py
# In the ESM environment, with the pinned CCD available:
ESMCFOLD_CCD_PATH=/path/to/ccd.pkl PYTHONPATH=src \
  .venv-esmfold2/bin/python -m pytest tests/test_esmfold2_inputs.py \
  tests/test_esmfold2_acceleration.py
```

The latter tests require pytest and `gemmi>=0.6.5` in the ESM test environment
(Gemmi is only needed there to check exported files against BoltzGen's reader).
They exercise actual
ESMFold2 feature construction without downloading model weights. A model-boundary
test proves full-chain LM inputs, cropped folding inputs, and ipSAE sample
selection. Acceleration checks also exercise source guards and native sampler
equivalence; their CUDA graph cases run when a GPU is available. Full-checkpoint
GPU inference qualification is separate.

References: [ESMFold2 model card](https://huggingface.co/biohub/ESMFold2),
[pinned public ESM implementation](https://github.com/Biohub/esm/tree/43b4548b86762edfa747b07d5f440aad3c33acee),
[ipSAE reference](https://github.com/DunbrackLab/IPSAE/blob/6174cf9e71cb1bd660cc805856a18c4871a6dec3/ipsae.py).
