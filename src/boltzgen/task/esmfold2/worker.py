"""ESMFold2 subprocess entrypoint; does not import the Boltz runtime."""

import argparse
from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import types

# Executing this file with `python -I` isolates dependencies even when BoltzGen
# is installed as a wheel. Expose only our package, never the parent environment's
# site-packages directory (which could shadow this runtime's torch/esm/numpy).
if not __package__:
    package = types.ModuleType("boltzgen")
    package.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["boltzgen"] = package

import numpy as np
import torch

from boltzgen.task.esmfold2.contract import (
    ESM_VERSION,
    ESMC_REPO,
    ESMC_REVISION,
    MODEL_REPO,
    MODEL_REVISION,
    SCHEMA_VERSION,
    SCORE_KEY,
    REDESIGN_SCORE_KEY,
    PTM_KEY,
    fingerprint,
)
from boltzgen.task.esmfold2.crop import crop_features, polymer_representatives
from boltzgen.task.esmfold2.ipsae import score_chain_vs_rest, score_interface


def configure_ccd() -> None:
    """Resolve the pinned CCD before importing ESM, which snapshots the path."""
    from huggingface_hub import hf_hub_download

    os.environ["ESMCFOLD_CCD_PATH"] = hf_hub_download(
        MODEL_REPO, "ccd.pkl", revision=MODEL_REVISION
    )


def restore_smiles_atom_names(features: dict, infos: list, chains: list[dict]) -> None:
    """Restore source atom identities after ESM's SMILES-specific naming."""
    from rdkit import Chem
    from esm.models.esmfold2.layers import CHAR_VOCAB_SIZE

    by_id = {chain["id"]: chain for chain in chains}
    for info in infos:
        chain = by_id[info.chain_id]
        if "smiles" not in chain:
            continue
        molecule = Chem.MolFromSmiles(chain["smiles"])
        names = chain["smiles_atom_names"]
        if molecule is None or molecule.GetNumAtoms() != len(names):
            raise ValueError("Invalid source atom mapping for SMILES ligand")
        molecule = Chem.AddHs(molecule)
        ranks = Chem.CanonicalRankAtoms(molecule)
        aliases = {
            f"{molecule.GetAtomWithIdx(i).GetSymbol().upper()}{ranks[i] + 1}": name.upper()
            for i, name in enumerate(names)
        }
        info.ligand_bonds = [(aliases[a], aliases[b]) for a, b in info.ligand_bonds]
        for token in info.tokens:
            for i in range(token.atom_start, token.atom_start + token.atom_count):
                encoded = features["ref_atom_name_chars"][0, i]
                current = "".join(chr(n + 32) for n in encoded.tolist() if n)
                name = aliases[current]
                if not 1 <= len(name) <= encoded.numel() or any(
                    not 0 <= ord(c) - 32 < CHAR_VOCAB_SIZE for c in name
                ):
                    raise ValueError(f"Unsupported source atom name {name}")
                encoded.zero_()
                encoded[: len(name)] = encoded.new_tensor([ord(c) - 32 for c in name])


def prepare_request(request: dict, builder) -> tuple[dict, list]:
    """Build full-chain features and explicit chemistry without coordinate templates."""
    from esm.models.esmfold2 import (
        CovalentBond,
        DNAInput,
        LigandInput,
        Modification,
        ProteinInput,
        RNAInput,
        StructurePredictionInput,
    )
    from esm.models.esmfold2.conformers import load_ccd
    from Bio.Data.PDBData import protein_letters_3to1_extended

    canonical = dict(
        zip(
            "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
            "ARNDCQEGHILKMFPSTWYV",
        )
    )
    ccd = load_ccd()
    inputs = []
    for chain in request["chains"]:
        kind = chain["mol_type"]
        names = chain["residue_names"]
        if kind == 3:
            inputs.append(
                LigandInput(id=chain["id"], smiles=chain["smiles"])
                if "smiles" in chain
                else LigandInput(id=chain["id"], ccd=names)
            )
            continue
        letters, modifications = [], []
        for i, name in enumerate(names):
            if kind == 0:
                letter = canonical.get(name)
                if letter is None:
                    letter = protein_letters_3to1_extended.get(name, "X")
            else:
                letter = name.removeprefix("D") if kind == 1 else name
                if letter not in ("ACGT" if kind == 1 else "ACGU"):
                    letter = "N"
            standard = (
                name in canonical
                if kind == 0
                else name
                in ({"DA", "DC", "DG", "DT"} if kind == 1 else {"A", "C", "G", "U"})
            )
            if not standard:
                if name not in ccd:
                    raise ValueError(
                        f"Residue {name} is absent from the pinned ESMFold2 CCD"
                    )
                modifications.append(Modification(position=i, ccd=name))
            letters.append(letter)
        cls = (ProteinInput, DNAInput, RNAInput)[kind]
        kwargs = dict(
            id=chain["id"],
            sequence="".join(letters),
            modifications=modifications or None,
        )
        if kind != 1:
            kwargs["msa"] = None
        inputs.append(cls(**kwargs))
    # Preliminary endpoints turn on native leaving-atom treatment for bonded
    # ligands. Resolve the real endpoints by name in those prepared residues.
    preliminary = [
        CovalentBond(a, b, 0, d, e, 0) for a, b, _, d, e, _ in request["bonds"]
    ]
    spi = StructurePredictionInput(sequences=inputs, covalent_bonds=preliminary or None)
    features, infos = builder.prepare_input(
        spi, seed=request["options"]["seed"], device="cpu"
    )
    restore_smiles_atom_names(features, infos, request["chains"])
    if preliminary:
        atom_names = features["ref_atom_name_chars"][0].tolist()
        lookup = {}
        for info in infos:
            by_residue = {}
            for token in info.tokens:
                for i in range(token.atom_start, token.atom_start + token.atom_count):
                    if features["atom_attention_mask"][0, i]:
                        by_residue.setdefault(token.residue_index, []).append(
                            "".join(chr(n + 32) for n in atom_names[i] if n)
                        )
            for position, names in by_residue.items():
                for offset, name in enumerate(names):
                    lookup[(info.chain_id, position, name)] = offset
        bonds = []
        smiles_chains = {
            chain["id"] for chain in request["chains"] if "smiles" in chain
        }
        for a, b, c, d, e, f in request["bonds"]:
            # Boltz SMILES names include Cl1/Br1; native ESM encodes uppercase
            # names in a 64-character vocabulary. Normalize endpoint aliases too.
            c = c.upper() if a in smiles_chains else c
            f = f.upper() if d in smiles_chains else f
            try:
                bonds.append(CovalentBond(a, b, lookup[a, b, c], d, e, lookup[d, e, f]))
            except KeyError as exc:
                raise ValueError(
                    f"ESMFold2 cannot preserve covalent endpoint {exc.args[0]}"
                ) from exc
        features, infos = builder.prepare_input(
            replace(spi, covalent_bonds=bonds),
            seed=request["options"]["seed"],
            device="cpu",
        )
        restore_smiles_atom_names(features, infos, request["chains"])
    from esm.models.esmfold2.paired_msa import protein_letter_to_res_type

    letter_types = protein_letter_to_res_type()
    input_by_id = {item.id: item for item in inputs}
    expected_query = [
        letter_types.get(input_by_id[info.chain_id].sequence[token.residue_index], 22)
        if info.mol_type == 0
        else token.res_type
        for info in infos
        for token in info.tokens
    ]
    if (
        features["msa"].shape != (1, 1, len(expected_query))
        or features["msa"][0, 0].tolist() != expected_query
    ):
        raise ValueError(
            "ESMFold2 features contain an unexpected MSA; only the native query row is allowed"
        )
    return features, infos


def write_structure(
    path: Path, coords: torch.Tensor, plddt: torch.Tensor, features: dict, infos: list
) -> None:
    """Write the selected sample using full-source residue numbers and chain IDs."""
    from biotite.structure import AtomArray
    from biotite.structure.io.pdbx import CIFFile, set_structure
    from esm.models.esmfold2.constants import ELEMENT_NUMBER_TO_SYMBOL

    mask = features["atom_attention_mask"][0].bool().cpu().numpy()
    names = features["ref_atom_name_chars"][0].cpu().tolist()
    array = AtomArray(int(mask.sum()))
    array.add_annotation("chain_id", f"U{max(len(info.chain_id) for info in infos)}")
    array.coord = coords.detach().float().cpu().numpy()[mask]
    array.set_annotation("occupancy", np.ones(len(array)))
    array.add_annotation("b_factor", float)
    atom_to_output = {int(old): new for new, old in enumerate(np.flatnonzero(mask))}
    for info in infos:
        for token in info.tokens:
            for old in range(token.atom_start, token.atom_start + token.atom_count):
                if old not in atom_to_output:
                    continue
                index = atom_to_output[old]
                array.chain_id[index] = info.chain_id
                array.res_id[index] = token.residue_index + 1
                array.res_name[index] = token.residue_name
                array.atom_name[index] = "".join(chr(n + 32) for n in names[old] if n)
                array.element[index] = ELEMENT_NUMBER_TO_SYMBOL[
                    int(features["ref_element"][0, old])
                ]
                array.hetero[index] = info.mol_type == 3
                array.b_factor[index] = float(plddt[token.token_index]) * 100
    cif = CIFFile()
    set_structure(cif, array)
    cif.write(path)


@contextmanager
def lm_dropout_context(model, probability: float):
    """Apply the requested dropout, including zero, to the pinned standard model."""
    config = model.config.lm_encoder
    saved = config.lm_dropout, config.per_loop_lm_dropout
    config.lm_dropout, config.per_loop_lm_dropout = probability, True
    try:
        yield
    finally:
        config.lm_dropout, config.per_loop_lm_dropout = saved


def run_request(model, builder, request: dict, output: Path, device: str) -> None:
    """Run full-context ESMC, crop inputs, and select the highest protocol score."""
    full, full_infos = prepare_request(request, builder)
    cropped, infos, selected, audit = crop_features(full, full_infos, request["chains"])
    representatives = polymer_representatives(cropped, infos)
    design = [i for chain in request["design_chains"] for i in representatives[chain]]
    target = [i for chain in request["target_chains"] for i in representatives[chain]]
    options = request["options"]
    torch.manual_seed(options["seed"])
    torch.cuda.manual_seed_all(options["seed"])
    full = {k: v.to(device) for k, v in full.items()}
    cropped = {k: v.to(device) for k, v in cropped.items()}
    # The native model controls mixed precision around its LM and trunk. An
    # outer autocast would also change the diffusion/confidence heads' precision.
    with torch.inference_mode():
        full_lm = model._compute_lm_hidden_states(
            full["input_ids"],
            full["asym_id"],
            full["residue_index"],
            full["mol_type"],
            full["token_attention_mask"],
            lm_mask_pct=0.0,
        )
        if full_lm.shape[:2] != full["input_ids"].shape:
            raise ValueError("ESMC did not encode the complete original source chains")
        crop_lm = full_lm.index_select(1, selected.to(device))
        audit["full_lm_shape"] = list(full_lm.shape)
        audit["crop_lm_shape"] = list(crop_lm.shape)
        del full_lm, full
        with lm_dropout_context(model, options["lm_dropout"]):
            prediction = model(
                **cropped,
                lm_hidden_states=crop_lm,
                num_loops=options["num_loops"],
                num_sampling_steps=options["sampling_steps"],
                num_diffusion_samples=options["diffusion_samples"],
                lm_mask_pct=0.0,
            )
    pae = prediction["pae"].float().cpu().numpy()
    if len(pae) != options["diffusion_samples"]:
        raise ValueError("ESMFold2 returned an unexpected number of confidence samples")
    kind = {c["id"]: c["mol_type"] for c in request["chains"]}
    mode = request.get("scoring_mode", "binder")
    chain_rest_samples = []
    if mode == "redesign":
        ranking_key = REDESIGN_SCORE_KEY
        if len(representatives) == 1:
            score_metric = PTM_KEY
            ptm = prediction["ptm"].float().cpu().numpy().reshape(-1)
            if (
                len(ptm) != len(pae)
                or not np.isfinite(ptm).all()
                or ((ptm < 0) | (ptm > 1)).any()
            ):
                raise ValueError("ESMFold2 returned invalid pTM samples")
            samples = [
                {PTM_KEY: float(value), ranking_key: float(value)} for value in ptm
            ]
        else:
            score_metric = SCORE_KEY
            chain_rest_samples = [
                score_chain_vs_rest(p, representatives, kind) for p in pae
            ]
            samples = []
            for scores in chain_rest_samples:
                weakest = min(metrics[SCORE_KEY] for metrics in scores.values())
                samples.append({SCORE_KEY: weakest, ranking_key: weakest})
    elif mode == "binder":
        ranking_key = score_metric = SCORE_KEY
        samples = [
            score_interface(p, design, target, nucleic_acid=request["nucleic_acid"])
            for p in pae
        ]
    else:
        raise ValueError(f"Unknown ESMFold2 scoring mode: {mode}")
    best = max(range(len(samples)), key=lambda i: samples[i][ranking_key])
    pairs = (
        [
            (first, second)
            for first in representatives
            for second in representatives
            if first != second
        ]
        if mode == "redesign"
        else [
            (first, second)
            for first in request["design_chains"]
            for second in request["target_chains"]
        ]
    )
    pair_scores = {
        f"{first}:{second}": score_interface(
            pae[best],
            representatives[first],
            representatives[second],
            nucleic_acid=kind[first] in (1, 2) or kind[second] in (1, 2),
        )
        for first, second in pairs
    }

    def artifact(suffix: str) -> Path:
        return output / f"{request['design_id']}.{suffix}"

    # A forced rerun may replace previously complete artifacts. Invalidate its
    # completion marker before the first write so an interrupted write cannot
    # reuse an old JSON alongside partially replaced coordinates or PAE.
    artifact("json").unlink(missing_ok=True)
    write_structure(
        artifact("cif"),
        prediction["sample_atom_coords"][best],
        prediction["plddt"][best],
        cropped,
        infos,
    )
    np.savez_compressed(
        artifact("npz"),
        pae=pae,
        selected_sample=best,
        coords=prediction["sample_atom_coords"][best].float().cpu().numpy(),
        plddt=prediction["plddt"][best].float().cpu().numpy(),
        residue_index=cropped["residue_index"].cpu().numpy(),
        asym_id=cropped["asym_id"].cpu().numpy(),
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "input_hash": fingerprint(request),
        "design_sha256": request["design_sha256"],
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "esmc_revision": ESMC_REVISION,
        "esm_version": ESM_VERSION,
        "options": options,
        "selected_sample": best,
        "sample_selection": "maximum_" + score_metric,
        "scoring_mode": mode,
        "score_metric": score_metric,
        "chain_vs_rest_samples": chain_rest_samples,
        "metrics": samples[best],
        "samples": samples,
        "chain_pair_scores": pair_scores,
        "input_audit": audit,
        "full_sequence_hashes": {
            c["id"]: fingerprint({"residue_names": c["residue_names"]})
            for c in request["chains"]
        },
    }
    # The result JSON is the completion marker; never reuse a partial prediction.
    temporary = artifact("json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(artifact("json"))
    print(
        f"ESMFold2 {request['design_id']}: sample {best}, "
        f"{score_metric}={samples[best][ranking_key]:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if version("esm") != ESM_VERSION:
        raise RuntimeError(
            f"Install esm=={ESM_VERSION} in the --esmfold2_python environment"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ESMFold2 scoring requires a CUDA GPU")
    from huggingface_hub import constants, snapshot_download

    print(
        f"ESM checkpoint cache: {Path(constants.HF_HUB_CACHE).resolve()}\n"
        "First use downloads about 27 GB of ESMFold2 and ESMC weights; "
        "cached files are reused on later runs. Downloads may take several minutes "
        "or longer depending on the connection.\n"
        "Set HF_HOME or HF_HUB_CACHE before starting to change this location; "
        "--cache controls Boltz downloads only.",
        flush=True,
    )
    print("Checking/downloading ESMFold2 chemical component data...", flush=True)
    configure_ccd()
    print(
        f"Chemical component data ready: {os.environ['ESMCFOLD_CCD_PATH']}", flush=True
    )
    from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model

    print("Checking/downloading ESMFold2 2021 checkpoint...", flush=True)
    model_path = snapshot_download(
        MODEL_REPO, revision=MODEL_REVISION, allow_patterns=["*.json", "*.safetensors"]
    )
    print(f"ESMFold2 checkpoint ready: {model_path}", flush=True)
    print("Checking/downloading ESMC-6B checkpoint...", flush=True)
    esmc_path = snapshot_download(
        ESMC_REPO, revision=ESMC_REVISION, allow_patterns=["*.json", "*.safetensors"]
    )
    print(f"ESMC-6B checkpoint ready: {esmc_path}", flush=True)
    print(
        "Loading ESMFold2 weights into memory; this can take several minutes...",
        flush=True,
    )
    model = EsmFold2Model.from_pretrained(model_path, load_esmc=False)
    print("Loading ESMC-6B weights into memory...", flush=True)
    model.load_esmc(esmc_path)
    print(f"Moving ESMFold2 and ESMC to {args.device}...", flush=True)
    model = model.to(args.device).eval().requires_grad_(False)
    builder = ESMFold2InputBuilder()
    print(f"ESMFold2 ready for scoring on {args.device}.", flush=True)
    for path in json.loads(args.manifest.read_text()):
        request = json.loads(Path(path).read_text())
        if (
            request["model_revision"] != MODEL_REVISION
            or request["esmc_revision"] != ESMC_REVISION
            or request["esm_version"] != ESM_VERSION
        ):
            raise ValueError("Request does not match the pinned ESMFold2 runtime")
        print(f"Scoring {request['design_id']} with ESMFold2", flush=True)
        run_request(model, builder, request, Path(path).parent, args.device)


if __name__ == "__main__":
    main()
