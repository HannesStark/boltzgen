"""Build auditable ESMFold2 requests and execute them in a separate environment."""

import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from rdkit import Chem

from boltzgen.data.source_context import update_designed
from boltzgen.task.esmfold2.contract import (
    ACCELERATION_REVISION,
    ESM_VERSION,
    ESMC_REVISION,
    MODEL_REVISION,
    SCHEMA_VERSION,
    SCORE_DIR,
    file_sha256,
    fingerprint,
    load_result,
    validate_scoring_mode,
    validate_acceleration,
)
from boltzgen.task.task import Task
from boltzgen.task.esmfold2.runtime import resolve_python, worker_command

logger = logging.getLogger(__name__)


def _publish_validated_result(
    staging_dir: Path,
    outdir: Path,
    request_path: Path,
    staged_request_path: Path,
    request: dict,
) -> None:
    """Publish one result, invalidating any old completion marker first."""
    design_id = request["design_id"]
    staged_result_path = staging_dir / f"{design_id}.json"
    result_path = outdir / f"{design_id}.json"
    # A crash while replacing artifacts must not leave an old result marker
    # beside new CIF/NPZ files; readers treat that JSON as the completion proof.
    result_path.unlink(missing_ok=True)
    for suffix in ("cif", "npz"):
        os.replace(
            staging_dir / f"{design_id}.{suffix}",
            outdir / f"{design_id}.{suffix}",
        )
    os.replace(staged_request_path, request_path)
    os.replace(staged_result_path, result_path)


def validate_context(context: dict | None) -> None:
    """Require trustworthy full source sequences, including for legacy rescoring."""
    if context is None or context.get("version") != 1:
        raise ValueError(
            "Missing full-source context. Regenerate with the original YAML/files, or provide source_context_dir/<design_id>.json; cropped sequences cannot substitute for full ESMC context."
        )
    for chain in context["chains"]:
        if chain["mol_type"] != 3 and not chain["complete"]:
            raise ValueError(
                f"Full sequence is unavailable for {chain['source_chain']} ({chain['source']}). Supply file.full_sequences with source_res_indices, or an explicit source_context_dir sidecar. {chain.get('reason', '')}"
            )
        indices = chain["indices"]
        if (
            not indices
            or indices != sorted(set(indices))
            or indices[0] < 0
            or indices[-1] >= len(chain["residue_names"])
        ):
            raise ValueError("Invalid full-source residue mapping")


def make_request(
    feat: dict,
    options: dict,
    target_chains: list[str] | None = None,
    context: dict | None = None,
    scoring_mode: str = "binder",
) -> dict:
    """Translate actual sampled/generated identities; never reparse a stochastic YAML."""
    validate_scoring_mode(scoring_mode, target_chains)
    structure = feat["str_gen"]
    tokenized = feat["tokenized"]
    if context is None:
        context = json.loads(feat["source_context"])
    validate_context(context)
    context = update_designed(context, structure)
    chains = []
    residues = {}
    for entry, chain in zip(context["chains"], structure.chains, strict=True):
        selected = tokenized.tokens["asym_id"] == chain["asym_id"]
        # Covalent closure expands chain_design_mask for refolding; it must
        # not turn a fixed target into a designed scoring partner.
        role = (
            "design"
            if np.asarray(feat["design_mask"])[selected].any()
            else "target"
        )
        chain_id = str(chain["name"])
        entry = dict(entry, id=chain_id, role=role)
        extra_mols = feat.get("extra_mols") or {}
        if entry["mol_type"] == 3 and any(
            name in extra_mols for name in entry["residue_names"]
        ):
            if len(entry["residue_names"]) != 1:
                raise ValueError(
                    "Custom multi-residue ligands require explicit CCD identities"
                )
            name = entry["residue_names"][0]
            molecule = extra_mols[name]
            molecule = Chem.RemoveHs(molecule)
            entry["smiles"] = Chem.MolToSmiles(molecule)
            # SMILES traversal can reorder atoms. Preserve their actual Boltz
            # names in the serialized order for covalent endpoints/deletions.
            order = json.loads(molecule.GetProp("_smilesAtomOutputOrder"))
            entry["smiles_atom_names"] = [
                molecule.GetAtomWithIdx(i).GetProp("name") for i in order
            ]
        for offset, source_index in enumerate(entry["indices"]):
            residues[int(chain["res_idx"]) + offset] = (chain_id, source_index)
        chains.append(entry)
    candidate = [c for c in chains if c["role"] == "design" and c["mol_type"] != 3]
    target = [c for c in chains if c["role"] == "target" and c["mol_type"] != 3]
    if target_chains is not None:
        if not set(target_chains) <= {c["id"] for c in target}:
            raise ValueError(
                "scoring_target_chains must name nonempty, unique target polymer chains"
            )
        target = [c for c in target if c["id"] in target_chains]
    if not candidate:
        raise ValueError("ESMFold2 scoring requires a designed polymer chain")
    if scoring_mode == "binder" and not target:
        if any(c["role"] == "target" and c["mol_type"] == 3 for c in chains):
            raise ValueError(
                "A ligand-only target has no polymer ipSAE; use --protocol protein-small_molecule"
            )
        raise ValueError("Interaction scoring requires a separate target polymer chain")
    if scoring_mode == "binder" and len({c["mol_type"] == 0 for c in target}) > 1:
        raise ValueError(
            "A mixed protein/nucleic-acid target requires explicit scoring_target_chains of one polymer type"
        )
    bonds = []
    omitted_atoms = {
        (c["id"], position, name)
        for c in chains
        for position, name in c.get("omitted_atoms", [])
    }
    for bond in structure.bonds:
        if bond["res_1"] == bond["res_2"]:
            continue
        chain1, pos1 = residues[int(bond["res_1"])]
        chain2, pos2 = residues[int(bond["res_2"])]
        if (
            chain1,
            pos1,
            str(structure.atoms[bond["atom_1"]]["name"]),
        ) in omitted_atoms or (
            chain2,
            pos2,
            str(structure.atoms[bond["atom_2"]]["name"]),
        ) in omitted_atoms:
            raise ValueError("A covalent bond endpoint cannot also be an omitted atom")
        bonds.append(
            [
                chain1,
                pos1,
                str(structure.atoms[bond["atom_1"]]["name"]),
                chain2,
                pos2,
                str(structure.atoms[bond["atom_2"]]["name"]),
            ]
        )
    request = {
        "schema_version": SCHEMA_VERSION,
        "model_revision": MODEL_REVISION,
        "esmc_revision": ESMC_REVISION,
        "esm_version": ESM_VERSION,
        "design_id": feat["id"],
        "design_sha256": file_sha256(Path(feat["path"])),
        "chains": chains,
        "bonds": bonds,
        "design_chains": [c["id"] for c in candidate],
        "target_chains": [c["id"] for c in target],
        "nucleic_acid": any(c["mol_type"] in (1, 2) for c in candidate + target),
        "options": options,
    }
    if scoring_mode == "redesign":
        request["scoring_mode"] = scoring_mode
    return request


class ESMFold2Score(Task):
    """Run the pinned model once per process, retaining five samples per design."""

    def __init__(
        self,
        data,
        design_dir: str,
        python: str | None = None,
        reuse: bool = False,
        devices: int = 1,
        source_context_dir: str | None = None,
        scoring_target_chains: list[str] | None = None,
        scoring_mode: str = "binder",
        num_loops: int = 20,
        sampling_steps: int = 200,
        diffusion_samples: int = 5,
        lm_dropout: float = 0.3,
        seed: int = 0,
        acceleration: str = "auto",
    ):
        self.data = data
        self.design_dir = Path(design_dir)
        self.python = python
        self.reuse = reuse
        self.devices = int(devices)
        self.source_context_dir = (
            Path(source_context_dir) if source_context_dir else None
        )
        validate_scoring_mode(scoring_mode, scoring_target_chains)
        self.scoring_mode = scoring_mode
        self.target_chains = (
            list(scoring_target_chains) if scoring_target_chains is not None else None
        )
        if (
            min(num_loops, sampling_steps, diffusion_samples, self.devices) < 1
            or not 0 <= lm_dropout <= 1
        ):
            raise ValueError("Invalid ESMFold2 inference settings")
        self.options = dict(
            num_loops=num_loops,
            sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            lm_dropout=lm_dropout,
            lm_mask_pct=0.0,
            seed=seed,
            pae_cutoff=10.0,
            acceleration=acceleration,
            acceleration_revision=ACCELERATION_REVISION,
        )
        validate_acceleration(acceleration)

    def run(self, config=None) -> None:
        outdir = self.design_dir / SCORE_DIR
        outdir.mkdir(parents=True, exist_ok=True)
        dataset = self.data.predict_set
        requests: list[tuple[Path, dict]] = []
        for path, metadata, native in zip(
            dataset.generated_paths,
            dataset.metadata_paths,
            dataset.native_paths,
            strict=True,
        ):
            # Avoid __getitem__'s random replacement on fetch failure: scoring must
            # never silently substitute another candidate.
            feat = dataset.getitem_from_paths(metadata, path, native)
            context = None
            if self.source_context_dir is not None:
                context = json.loads(
                    (self.source_context_dir / f"{path.stem}.json").read_text()
                )
            request = make_request(
                feat, self.options, self.target_chains, context, self.scoring_mode
            )
            input_hash = fingerprint(request)
            output = outdir / f"{path.stem}.json"
            if self.reuse and output.exists():
                try:
                    load_result(output, input_hash)
                except ValueError:
                    logger.info("Recomputing stale ESMFold2 score for %s", path.stem)
                else:
                    continue
            request_path = outdir / f"{path.stem}.input.json"
            requests.append((request_path, request))
        if not requests:
            return
        # Resolve first so an offline cache miss cannot replace valid prior
        # artifacts. Workers write into an isolated staging directory; publish a
        # design's request only after its complete result validates.
        python = resolve_python(self.python)
        workers = []
        with tempfile.TemporaryDirectory(prefix=".esmfold2-stage-", dir=outdir) as temp:
            staging_dir = Path(temp)
            pending_requests: list[tuple[Path, Path, dict]] = []
            for request_path, request in requests:
                pending_path = staging_dir / request_path.name
                pending_path.write_text(
                    json.dumps(request, indent=2, allow_nan=False) + "\n"
                )
                pending_requests.append((request_path, pending_path, request))

            try:
                for index in range(min(self.devices, len(pending_requests))):
                    manifest = staging_dir / f"worker_{index}.json"
                    worker_requests = [
                        str(pending_path)
                        for _, pending_path, _ in pending_requests[index :: self.devices]
                    ]
                    manifest.write_text(json.dumps(worker_requests))
                    workers.append(
                        subprocess.Popen(
                            worker_command(python, manifest, f"cuda:{index}"),
                        )
                    )
                codes = [worker.wait() for worker in workers]
            finally:
                for worker in workers:
                    if worker.poll() is None:
                        worker.terminate()
                        worker.wait()

            validation_errors = []
            for request_path, pending_path, request in pending_requests:
                result_path = staging_dir / f"{request['design_id']}.json"
                try:
                    load_result(result_path, fingerprint(request))
                except (OSError, ValueError) as exc:
                    validation_errors.append(exc)
                    continue

                _publish_validated_result(
                    staging_dir,
                    outdir,
                    request_path,
                    pending_path,
                    request,
                )

            if any(codes):
                raise RuntimeError(
                    f"ESMFold2 scoring failed (worker exit codes {codes}); see preceding error. No Boltz2 score fallback is used."
                )
            if validation_errors:
                raise validation_errors[0]
