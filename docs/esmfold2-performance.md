# ESMFold2 performance measurements

Measured on NVIDIA H200 on 2026-10-05, with ESM 3.4.1.post1 and PyTorch 2.11.0+cu130.
These timings cover BoltzGen's ESMFold2 scoring worker. They do not measure the
complete design pipeline or installation. [Raw measurements](benchmarks/esmfold2-h200-20261005.json)
include every reported timing and the execution mode.

## Results

Median of two measured calls after one first call per fixture and mode. Every
call includes full-source ESMC, structural cropping, fresh graph capture where
applicable, all five structure samples, scoring and normal result writing. Saved
scores are never reused. Weight loading is excluded and reported separately.
In this run, model loading took 92.3 seconds for the native worker and 92.5
seconds for the fused worker. Downloads and a cold runtime install add further
startup time; these are not accelerated by this change.

| Fixture | Full / cropped tokens | Original (`off`) | Automatic | Optional fused |
| --- | ---: | ---: | ---: | ---: |
| Ubiquitin pair | 152 / 152 | 6.43 s | 5.01 s (1.28×) | 3.34 s (1.93×) |
| Discontinuous GFP complex | 552 / 428 | 29.43 s | 28.56 s (1.03×) | 12.93 s (2.28×) |
| GCN4 zipper pair | 66 / 66 | 5.38 s | 2.33 s (2.31×) | 2.74 s (1.97×) |

`auto` keeps the original numerical backend. `fused` changes it to ESMFold2's
bundled fused Triton/BF16 backend, which can change structures and scores.
Fused is not always the fastest choice: automatic was faster on the zipper in
this run. Two repeats provide a small timing sample; the raw data includes an
outlier in the fused zipper measurements. These results are not hardware-wide
or biological accuracy claims.

Peak allocated GPU memory, in GiB:

| Fixture | Original | Automatic | Fused |
| --- | ---: | ---: | ---: |
| Ubiquitin pair | 14.18 | 14.22 | 14.00 |
| Discontinuous GFP complex | 22.97 | 22.97 | 20.84 |
| GCN4 zipper pair | 13.20 | 13.20 | 13.26 |

First measured calls (existing kernel caches were not cleared; these are not
cold-install or cold-compilation measurements):

| Fixture | Original | Automatic | Fused |
| --- | ---: | ---: | ---: |
| Ubiquitin pair | 7.05 s | 5.02 s | 5.33 s |
| Discontinuous GFP complex | 29.59 s | 28.66 s | 14.64 s |
| GCN4 zipper pair | 5.56 s | 2.36 s | 2.47 s |

The protocol is unchanged: the pinned full2021 checkpoint and ESMC-6B, 20 loops,
200 requested diffusion steps, five samples, seed 0, LM dropout 0.3 and a query-only
MSA. The native sampler's sigma cutoff produces 134 denoising forwards from that
requested schedule in both arms. Full source sequences are encoded before
cropping; the GFP fixture has two discontinuous target crops.

## Implementation and limits

This ports mask caching, schedule-transfer hoisting and CUDA graph replay from
[Anthropic's public kit](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/esmfold2)
to the pinned native ESM implementation. It does not install the kit's older
Transformers environment or its additional `fast` kernels. Normal installation
and runtime dependencies are unchanged.

Automatic mode captures graphs through 256 cropped tokens; fused mode through
512. In the 428-token native fixture, graphs made scoring slightly slower;
mask caching and schedule hoisting alone were faster, so automatic uses those
optimizations at that size. Larger inputs use the same bounded caching path.
Each request releases its graphs and masks. A persistent capture stream avoids
accumulating PyTorch cuBLAS workspaces over a design campaign.

An untimed profiler run on the small pair reduced `cudaLaunchKernel` calls from
248,921 to 69,211, including graph setup. Profiling is excluded from the timing
table. H200 is the only GPU qualified here. Large fused inputs are checked against
the pinned kernel's int32 indexing limit; see [usage and limits](esmfold2.md#inference-acceleration).

## Correctness and lifetime checks

- 63 worker requests, comprising 315 structure samples, passed finite-output,
  full-sequence/crop-index, query-only MSA and highest-score selection checks.
  Coverage included the three timing fixtures, monomer pTM, RNA, DNA, and a return
  to an earlier input after changes in shape and molecular type.
- Deterministic comparisons on the zipper and GFP fixtures matched every
  coordinate, PAE, pLDDT, pTM and final CUDA RNG state bit for bit: automatic
  against the native backend, and the fused adapter against the unwrapped fused
  backend. This does not make fused identical to the original numerical backend.
- Ordinary GPU execution varies even at a fixed seed. For example, repeated
  native zipper ipSAE was 0.08376–0.08401 and fused was 0.08345–0.08391; both
  selected sample 1. These are numerical checks, not an accuracy benchmark.
- After initial workspace allocation, resident memory stayed at 12.862 GiB for
  the native/automatic worker and 12.918 GiB for fused across changes in input.
  The component test also checks eight successive graph lifetimes on one stream.
- CPU suite: 517 passed, 36 skipped. Native ESM/GPU component and input tests:
  28 passed, two Gemmi-dependent checks skipped in that isolated environment.
  The built wheel contains the runtime, adapter and license notices, and its
  worker imports successfully with Python isolation enabled.

RNA/DNA runs establish functional coverage. Their timing data is not used above:
a separate verification process loaded a checkpoint during the later coverage
runs. The reported three-fixture measurements finished before that load.

## Reproduction

From the checkout, use the same isolated runtime as scoring:

```bash
uv run --no-project --python 3.12 \
  --with-requirements src/boltzgen/resources/runtime/esmfold2.txt \
  python scripts/benchmark_esmfold2.py --output benchmark-native \
  --modes off auto --cases ubiquitin gfp_crop zipper --repeats 2

uv run --no-project --python 3.12 \
  --with-requirements src/boltzgen/resources/runtime/esmfold2.txt \
  python scripts/benchmark_esmfold2.py --output benchmark-fused \
  --modes fused --cases ubiquitin gfp_crop zipper --repeats 2
```

Use a new output directory each time. Add `--deterministic` to both comparison
arms for numerical checks, or `--profile` for a separate untimed operator profile.
`--cases monomer rna dna` exercises the additional scoring paths. Timing and
correctness fixtures use public or synthetic sequences and are not binding
predictions intended for biological interpretation.
