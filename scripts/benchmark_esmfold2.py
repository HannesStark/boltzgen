"""Benchmark BoltzGen's actual ESMFold2 scoring worker on benign fixtures.

Run in the pinned ESM environment; see docs/esmfold2.md. No Boltz generation
checkpoint is needed. These are computational fixtures, not binding predictions
intended for biological interpretation. Timings include preparation and artifact
writing, including fresh graph capture for every candidate; weight loading is
reported separately. No result cache is reused.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Run directly from a source checkout without installing Boltz's dependencies
# into the separate ESM runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHM"
    "KQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYITADK"
    "QKNGIKANFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)
# Canonical segment from https://www.rcsb.org/fasta/entry/2ZTA/display,
# excluding its leading X. This supplies a short interface fixture.
ZIPPER = "RMKQLEDKVEELLSKNYHLENEVARLKKLVGER"
AA = dict(
    zip(
        "ARNDCQEGHILKMFPSTWYV",
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    )
)


def protein(chain_id: str, sequence: str, indices: list[int] | None = None) -> dict:
    return dict(
        id=chain_id,
        mol_type=0,
        residue_names=[AA[c] for c in sequence],
        indices=list(range(len(sequence))) if indices is None else indices,
    )


def make_fixture(name: str, mode: str) -> dict:
    from boltzgen.task.esmfold2.contract import (
        ACCELERATION_REVISION,
        ESM_VERSION,
        MODEL_REVISION,
        ESMC_REVISION,
    )

    if name == "ubiquitin":
        chains = [protein("A", UBIQUITIN), protein("B", UBIQUITIN)]
    elif name == "gfp_crop":
        crop = list(range(8, 96)) + list(range(128, 216))
        chains = [
            protein("A", GFP, crop),
            protein("C", GFP, crop),
            protein("B", UBIQUITIN),
        ]
    elif name == "zipper":
        chains = [protein("A", ZIPPER), protein("B", ZIPPER)]
    elif name == "monomer":
        chains = [protein("B", UBIQUITIN)]
    elif name in ("rna", "dna"):
        letters = ("ACGU" if name == "rna" else "ACGT") * 8
        chains = [
            dict(
                id="A",
                mol_type=2 if name == "rna" else 1,
                residue_names=[c if name == "rna" else "D" + c for c in letters],
                indices=list(range(0, 32, 2)),
            ),
            protein("B", UBIQUITIN),
        ]
    else:
        raise ValueError(name)
    return dict(
        schema_version=1,
        design_id=name,
        design_sha256=hashlib.sha256(json.dumps(chains).encode()).hexdigest(),
        model_revision=MODEL_REVISION,
        esmc_revision=ESMC_REVISION,
        esm_version=ESM_VERSION,
        chains=chains,
        bonds=[],
        design_chains=["B"],
        target_chains=[c["id"] for c in chains if c["id"] != "B"],
        nucleic_acid=name in ("rna", "dna"),
        scoring_mode="redesign" if name == "monomer" else "binder",
        options=dict(
            seed=0,
            num_loops=20,
            sampling_steps=200,
            diffusion_samples=5,
            lm_dropout=0.3,
            lm_mask_pct=0.0,
            pae_cutoff=10.0,
            acceleration=mode,
            acceleration_revision=ACCELERATION_REVISION,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--modes", nargs="+", choices=["off", "auto", "fused"], default=["off", "auto"]
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["ubiquitin", "gfp_crop", "zipper", "monomer", "rna", "dna"],
        default=["ubiquitin", "gfp_crop", "zipper"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic PyTorch algorithms in BOTH arms; sampling/dropout stay unchanged.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect operator totals on a separate untimed call.",
    )
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must be unique")
    if "fused" in args.modes and set(args.modes) != {"fused"}:
        parser.error("Benchmark fused in a separate process with --modes fused")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty, to keep benchmark evidence immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from boltzgen.task.esmfold2 import worker
    from boltzgen.task.esmfold2.contract import (
        ESM_VERSION,
        MODEL_REVISION,
        ESMC_REVISION,
    )

    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    worker.configure_ccd()
    from esm.models.esmfold2 import EsmFold2Model, ESMFold2InputBuilder

    start = time.perf_counter()
    model_path = snapshot_download(
        "biohub/ESMFold2",
        revision=MODEL_REVISION,
        allow_patterns=["*.json", "*.safetensors"],
    )
    esmc_path = snapshot_download(
        "biohub/ESMC-6B",
        revision=ESMC_REVISION,
        allow_patterns=["*.json", "*.safetensors"],
    )
    model = EsmFold2Model.from_pretrained(model_path, load_esmc=False)
    model.load_esmc(esmc_path)
    model.to(args.device).eval().requires_grad_(False)
    worker.configure_backend(model, "fused" if args.modes == ["fused"] else "off")
    torch.cuda.synchronize(args.device)
    builder = ESMFold2InputBuilder()
    environment = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(args.device),
        esm=ESM_VERSION,
        model_revision=MODEL_REVISION,
        esmc_revision=ESMC_REVISION,
        deterministic=args.deterministic,
        model_load_seconds=time.perf_counter() - start,
    )
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2))
    print("MODEL_READY", json.dumps(environment), flush=True)
    captured = {}

    def retain_prediction(module, inputs, prediction):
        captured.clear()
        captured.update(
            {k: prediction[k] for k in ("pae", "plddt", "ptm", "sample_atom_coords")}
        )

    model.register_forward_hook(retain_prediction)
    timings = []
    for case_index, name in enumerate(args.cases):
        for mode in args.modes:
            request = make_fixture(name, mode)
            for repeat in range(args.repeats + 1):
                dest = args.output / f"{case_index}_{name}" / mode / str(repeat)
                dest.mkdir(parents=True)
                (dest / f"{name}.input.json").write_text(json.dumps(request, indent=2))
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                start = time.perf_counter()
                worker.run_request(model, builder, request, dest, args.device)
                torch.cuda.synchronize(args.device)
                row = dict(
                    case=name,
                    case_index=case_index,
                    mode=mode,
                    repeat=repeat,
                    warmup=repeat == 0,
                    seconds=time.perf_counter() - start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)
                    / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(args.device)
                    / 2**30,
                )
                result = json.loads((dest / f"{name}.json").read_text())
                row["execution"] = result["execution"]
                np.savez_compressed(
                    dest / "all_predictions.npz",
                    **{k: v.float().cpu().numpy() for k, v in captured.items()},
                    rng=torch.cuda.get_rng_state(args.device).cpu().numpy(),
                )
                captured.clear()
                torch.cuda.synchronize(args.device)
                row["resident_after_gib"] = (
                    torch.cuda.memory_allocated(args.device) / 2**30
                )
                timings.append(row)
                (args.output / "timings.json").write_text(json.dumps(timings, indent=2))
                print("BENCH_RESULT", json.dumps(row), flush=True)
            if args.profile and case_index == 0:
                profile_dest = args.output / f"{case_index}_{name}" / mode / "profile"
                profile_dest.mkdir()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    worker.run_request(
                        model, builder, request, profile_dest, args.device
                    )
                captured.clear()
                rows = [
                    dict(
                        operator=e.key,
                        calls=e.count,
                        cpu_self_us=e.self_cpu_time_total,
                        device_self_us=e.self_device_time_total,
                    )
                    for e in prof.key_averages()
                ]
                (args.output / f"profile_{mode}.json").write_text(
                    json.dumps(rows, indent=2)
                )
    print("BENCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
