import io
import warnings
from pathlib import Path
import random
from typing import List
import subprocess
import re
import biotite
import hydride
from sklearn.cluster import DBSCAN
from Bio import PDB
from biotite import structure
from Bio.Seq import Seq


from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.max_open_warning"] = 100

from Bio import Align

from boltzgen.data.rmsd_computation import get_true_coordinates
from boltzgen.model.loss.diffusion import weighted_rigid_align
from boltzgen.task.predict.data_from_generated import collate

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from boltzgen.data import const
from boltzgen.data.data import Structure, biotite_array_from_feat
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.model.loss.validation import factored_lddt_loss, compute_subset_rmsd

from biotite.structure.sasa import sasa
from biotite.structure.info import vdw_radius_single, vdw_radius_protor
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdbio

TARGET_ID_RE = re.compile(
    r"^(?:(?:sample\d+_|batch\d+_|rank\d+_)+)?([^_]+)(?:_[^_]+)*?(?:_(?:gen))*$"
)


def _load_stack(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        cif_file = pdbx.CIFFile.read(str(path))
        stack = pdbx.get_structure(cif_file, use_author_fields=False)
    elif suffix in {".pdb", ".ent"}:
        pdb_file = pdbio.PDBFile.read(str(path))
        stack = pdbio.get_structure(pdb_file, model=None)
    else:
        raise ValueError(f"Unsupported structure file extension: {suffix}")

    return stack


def compute_rmsd(atom_coords: torch.Tensor, pred_atom_coords: torch.Tensor):
    rmsd, _ = compute_subset_rmsd(
        atom_coords,
        pred_atom_coords,
        atom_mask=torch.ones_like(atom_coords[..., 0]),
        align_weights=torch.ones_like(atom_coords[..., 0]),
        subset_mask=torch.ones_like(atom_coords[..., 0]),
        multiplicity=1,
    )
    return rmsd


def make_histogram(
    df,
    column_name: str,
):
    data = df[column_name].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(data, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(data.mean(), color="red", linestyle="dashed", linewidth=1)

    ax.set_title(
        f"{column_name.replace('_', ' ').capitalize()} Distribution", fontsize=12
    )
    ax.set_xlabel(column_name.replace("_", " "), fontsize=10)
    ax.set_ylabel("Count", fontsize=10)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=8)

    plt.tight_layout()
    return fig


def get_best_folding_sample(folded):
    confidence = 0.8 * folded["design_to_target_iptm"] + 0.2 * folded["design_ptm"]
    best_idx = np.argmax(confidence)

    # TODO: remove the "if k in folded"
    best_sample = {
        k: folded[k][best_idx] for k in const.eval_keys_confidence if k in folded
    }
    best_sample["coords"] = folded["coords"][best_idx]
    return best_sample


def get_fold_metrics(
    feat,
    folded,
    compute_lddts=True,
    prefix="",
):
    batch = collate([feat])
    diffusion_samples = batch["coords"].shape[0]
    best_sample = get_best_folding_sample(folded)

    # Compute RMSDs
    rmsd_out = get_true_coordinates(
        batch=batch,
        out={"sample_atom_coords": torch.from_numpy(best_sample["coords"])},
        diffusion_samples=1,
        symmetry_correction=False,
        protein_lig_rmsd=True,
    )
    true_coords_resolved_mask = rmsd_out["true_coords_resolved_mask"]

    # Add to metrics dictionary
    metrics = {}
    metrics["rmsd"] = rmsd_out.get("rmsd").item()
    metrics["rmsd_design"] = rmsd_out.get("rmsd_design").item()
    metrics["rmsd_target"] = rmsd_out.get("rmsd_target").item()
    metrics["rmsd_design_target"] = rmsd_out.get("rmsd_design_target").item()
    metrics["target_aligned_rmsd_design"] = rmsd_out.get(
        "target_aligned_rmsd_design"
    ).item()
    metrics["rmsd<2.5"] = bool(metrics["rmsd"] <= 2.5)
    metrics["target_aligned<2.5"] = bool(metrics["target_aligned_rmsd_design"] <= 2.5)
    metrics["designability_rmsd_2"] = bool(metrics["rmsd_design"] <= 2.0)
    metrics["designability_rmsd_4"] = bool(metrics["rmsd_design"] <= 4.0)

    # Comput LDDTs
    if compute_lddts:
        all_lddt_dict, _ = factored_lddt_loss(
            feats=batch,
            atom_mask=true_coords_resolved_mask,
            true_atom_coords=batch["coords"],
            pred_atom_coords=torch.from_numpy(best_sample["coords"]),
            multiplicity=diffusion_samples,
            exclude_ions=False,
        )
        metrics.update({f"lddt_{k}": v.max().item() for k, v in all_lddt_dict.items()})
        metrics["designability_lddt_60"] = bool(metrics["lddt_intra_design"] >= 0.6)
        metrics["designability_lddt_65"] = bool(metrics["lddt_intra_design"] >= 0.65)
        metrics["designability_lddt_70"] = bool(metrics["lddt_intra_design"] >= 0.7)
        metrics["designability_lddt_75"] = bool(metrics["lddt_intra_design"] >= 0.75)
        metrics["designability_lddt_80"] = bool(metrics["lddt_intra_design"] >= 0.8)
        metrics["designability_lddt_85"] = bool(metrics["lddt_intra_design"] >= 0.85)
        metrics["designability_lddt_90"] = bool(metrics["lddt_intra_design"] >= 0.9)

    # metrics without prefix (backbone only is the same as all atom)
    # TODO: remove the "if k in best_sample"
    confs = {k: best_sample[k] for k in const.eval_keys_confidence if k in best_sample}
    confs["min_interaction_pae<1.5"] = bool(confs["min_interaction_pae"] <= 1.5)
    confs["min_interaction_pae<2"] = bool(confs["min_interaction_pae"] <= 2.0)
    confs["min_interaction_pae<2.5"] = bool(confs["min_interaction_pae"] <= 2.5)
    confs["min_interaction_pae<3"] = bool(confs["min_interaction_pae"] <= 3)
    confs["min_interaction_pae<4"] = bool(confs["min_interaction_pae"] <= 4)
    confs["min_interaction_pae<5"] = bool(confs["min_interaction_pae"] <= 5)
    confs["design_ptm>80"] = bool(confs["design_ptm"] >= 0.8)
    confs["design_ptm>75"] = bool(confs["design_ptm"] >= 0.75)
    confs["design_iptm>80"] = bool(confs["design_iptm"] >= 0.8)
    confs["design_iptm>70"] = bool(confs["design_iptm"] >= 0.7)
    confs["design_iptm>60"] = bool(confs["design_iptm"] >= 0.6)
    confs["design_iptm>50"] = bool(confs["design_iptm"] >= 0.5)

    prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    prefixed_metrics.update(confs)
    return prefixed_metrics


def count_noncovalents(feat):
    metrics = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        biotite_array = biotite_array_from_feat(feat)
        biotite_array, _ = hydride.add_hydrogen(biotite_array)
        hbond = biotite.structure.hbond(biotite_array)
    donor_idxs, acceptor_idxs = hbond[:, 0], hbond[:, 2]
    donor_design_hbonds = int(
        (
            biotite_array.is_design[donor_idxs]
            & ~biotite_array.is_chain_design[acceptor_idxs]
        ).sum()
    )
    acceptor_design_hbonds = int(
        (
            ~biotite_array.is_chain_design[donor_idxs]
            & biotite_array.is_design[acceptor_idxs]
        ).sum()
    )
    metrics["plip_hbonds"] = donor_design_hbonds + acceptor_design_hbonds

    # saltbridges
    pos_atoms = biotite_array[biotite_array.charge > 0]
    neg_atoms = biotite_array[biotite_array.charge < 0]
    if len(neg_atoms) > 0 and len(pos_atoms) > 0:
        pos_neg_distances = torch.cdist(
            torch.as_tensor(pos_atoms.coord), torch.as_tensor(neg_atoms.coord)
        )
        pos_idxs, neg_idxs = torch.where(
            (pos_neg_distances > 0.5) & (pos_neg_distances < 5.5)
        )
        # only keep the ones between design and non design
        pos_design_sb = int(
            (pos_atoms.is_design[pos_idxs] & ~neg_atoms.is_chain_design[neg_idxs]).sum()
        )
        neg_design_sb = int(
            (~pos_atoms.is_chain_design[pos_idxs] & neg_atoms.is_design[neg_idxs]).sum()
        )
        metrics["plip_saltbridge"] = pos_design_sb + neg_design_sb
    else:
        metrics["plip_saltbridge"] = 0
    return metrics


def tm_score(coords1, coords2):
    num_atoms1 = coords1.shape[0]
    num_atoms2 = coords2.shape[0]

    atom_array1 = structure.AtomArray(num_atoms1)
    atom_array1.coord = coords1.numpy()
    atom_array1.element = np.array(["C"] * num_atoms1)
    atom_array1.atom_name = np.array(["CA"] * num_atoms1)
    atom_array1.res_name = np.array(["ALA"] * num_atoms1)
    atom_array1.chain_id = np.array(["A"] * num_atoms1)
    atom_array1.res_id = np.arange(1, num_atoms1 + 1)

    atom_array2 = structure.AtomArray(num_atoms2)
    atom_array2.coord = coords2.numpy()
    atom_array2.element = np.array(["C"] * num_atoms2)
    atom_array2.atom_name = np.array(["CA"] * num_atoms2)
    atom_array2.res_name = np.array(["ALA"] * num_atoms2)
    atom_array2.chain_id = np.array(["A"] * num_atoms2)
    atom_array2.res_id = np.arange(1, num_atoms2 + 1)

    try:
        # This fails with a value error if the structures are too dissimilar. In that event, we return 0 as the TM-Score
        aligned, transform, fixed_indices, mobile_indices = (
            structure.superimpose_structural_homologs(
                atom_array1, atom_array2, max_iterations=25
            )
        )
        tm_align_fixed = structure.tm_score(
            atom_array1,
            aligned,
            fixed_indices,
            mobile_indices,
        )
    except:
        tm_align_fixed = 0

    tm_score_rmsd_aligned = 0
    if num_atoms1 == num_atoms2:
        coords1 = weighted_rigid_align(
            coords1.float()[None],
            coords2.float()[None],
            weights=torch.ones(len(coords1)).float()[None],
            mask=torch.ones(len(coords2))[None],
        ).squeeze()

        atom_array1 = structure.AtomArray(num_atoms1)
        atom_array1.coord = coords1.numpy()
        atom_array1.element = np.array(["C"] * num_atoms1)
        atom_array1.atom_name = np.array(["CA"] * num_atoms1)
        atom_array1.res_name = np.array(["ALA"] * num_atoms1)
        atom_array1.chain_id = np.array(["A"] * num_atoms1)
        atom_array1.res_id = np.arange(1, num_atoms1 + 1)

        atom_array2 = structure.AtomArray(num_atoms2)
        atom_array2.coord = coords2.numpy()
        atom_array2.element = np.array(["C"] * num_atoms2)
        atom_array2.atom_name = np.array(["CA"] * num_atoms2)
        atom_array2.res_name = np.array(["ALA"] * num_atoms2)
        atom_array2.chain_id = np.array(["A"] * num_atoms2)
        atom_array2.res_id = np.arange(1, num_atoms2 + 1)
        try:
            _, _, fixed_indices, mobile_indices = (
                structure.superimpose_structural_homologs(
                    atom_array1, atom_array2, max_iterations=25
                )
            )
            tm_score_rmsd_aligned = structure.tm_score(
                atom_array1,
                atom_array2,
                fixed_indices,
                mobile_indices,
            )
        except:
            pass

    return tm_score_rmsd_aligned, tm_align_fixed


def vendi_from_sim(mat):
    mat = mat + mat.T
    np.fill_diagonal(mat, 1.0)
    eigvals, _ = np.linalg.eigh(mat / len(mat))
    eigvals = np.clip(eigvals, 0.0, None)
    return np.exp(np.nansum(-(eigvals * np.log(eigvals))))


def vendi_scores(
    all_ca_coords: List[np.ndarray],
    all_metrics: list = None,
    fold_metrics: bool = False,
    diversity_subset: int = None,
    compute_lddts: bool = True,
    compute_iptms: bool = True,
    compute_min_int_paes: bool = True,
    backbone_fold_metrics: bool = False,
    allatom_fold_metrics: bool = True,
) -> float:
    if fold_metrics or diversity_subset is not None:
        assert all_metrics is not None
    if all_metrics is not None:
        assert len(all_ca_coords) == len(all_metrics)
    if diversity_subset is not None and diversity_subset < len(all_ca_coords):
        indices = random.sample(range(len(all_ca_coords)), diversity_subset)
        all_metrics = [all_metrics[i] for i in indices]
        all_ca_coords = [all_ca_coords[i] for i in indices]
    N = len(all_ca_coords)
    tm = np.zeros((N, N), dtype=np.float32)
    tm_fixed = np.zeros((N, N), dtype=np.float32)

    for i in tqdm(range(N), desc="Computing structure diversity."):
        for j in range(i + 1, N):
            tm_score_rmsd_aligned, tm_fixeds = tm_score(
                all_ca_coords[i], all_ca_coords[j]
            )
            tm[i, j] = tm_score_rmsd_aligned
            tm_fixed[i, j] = tm_fixeds

    scores = {
        "vendi_tm_fixed": vendi_from_sim(tm_fixed),
        "vendi_tm_align": vendi_from_sim(tm),
    }
    prefixes = []
    if allatom_fold_metrics:
        prefixes.append("")
    if backbone_fold_metrics:
        prefixes.append("bb_")
    for prefix in prefixes:
        mask_2 = np.array([m[f"{prefix}designability_rmsd_2"] for m in all_metrics])
        mask_4 = np.array([m[f"{prefix}designability_rmsd_4"] for m in all_metrics])
        mask_25 = np.array([m[f"{prefix}rmsd<2.5"] for m in all_metrics])
        mask_target_25 = np.array(
            [m[f"{prefix}target_aligned<2.5"] for m in all_metrics]
        )
        scores.update(
            {
                f"vendi_tm_{prefix}rmsd<2.5": vendi_from_sim(
                    tm_fixed[mask_25][:, mask_25]
                )
                if np.sum(mask_25) > 0
                else 0.0,
                f"vendi_tm_{prefix}rmsd_2": vendi_from_sim(tm_fixed[mask_2][:, mask_2])
                if np.sum(mask_2) > 0
                else 0.0,
                f"vendi_tm_{prefix}rmsd_4": vendi_from_sim(tm_fixed[mask_4][:, mask_4])
                if np.sum(mask_4) > 0
                else 0.0,
                f"vendi_tm_{prefix}target_aligned_rmsd<2.5": vendi_from_sim(
                    tm_fixed[mask_target_25][:, mask_target_25]
                )
                if np.sum(mask_target_25) > 0
                else 0.0,
            }
        )

        if compute_lddts:
            mask_60 = np.array(
                [m[f"{prefix}designability_lddt_60"] for m in all_metrics]
            )
            mask_65 = np.array(
                [m[f"{prefix}designability_lddt_65"] for m in all_metrics]
            )
            mask_70 = np.array(
                [m[f"{prefix}designability_lddt_70"] for m in all_metrics]
            )
            mask_75 = np.array(
                [m[f"{prefix}designability_lddt_75"] for m in all_metrics]
            )
            mask_80 = np.array(
                [m[f"{prefix}designability_lddt_80"] for m in all_metrics]
            )
            mask_85 = np.array(
                [m[f"{prefix}designability_lddt_85"] for m in all_metrics]
            )
            mask_90 = np.array(
                [m[f"{prefix}designability_lddt_90"] for m in all_metrics]
            )
            scores.update(
                {
                    f"vendi_tm_{prefix}lddt_60": vendi_from_sim(
                        tm_fixed[mask_60][:, mask_60]
                    )
                    if np.sum(mask_60) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}lddt_65": vendi_from_sim(
                        tm_fixed[mask_65][:, mask_65]
                    )
                    if np.sum(mask_65) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}lddt_70": vendi_from_sim(
                        tm_fixed[mask_70][:, mask_70]
                    )
                    if np.sum(mask_70) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}lddt_75": vendi_from_sim(
                        tm_fixed[mask_75][:, mask_75]
                    )
                    if np.sum(mask_75) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}lddt_80": vendi_from_sim(
                        tm_fixed[mask_80][:, mask_80]
                    )
                    if np.sum(mask_80) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}lddt_85": vendi_from_sim(
                        tm_fixed[mask_85][:, mask_85]
                    )
                    if np.sum(mask_85) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}lddt_90": vendi_from_sim(
                        tm_fixed[mask_90][:, mask_90]
                    )
                    if np.sum(mask_90) > 0
                    else 0.0,
                }
            )
        if compute_iptms:
            mask_80 = np.array([m[f"design_iptm>80"] for m in all_metrics])
            mask_70 = np.array([m[f"design_iptm>70"] for m in all_metrics])
            mask_60 = np.array([m[f"design_iptm>60"] for m in all_metrics])
            mask_50 = np.array([m[f"design_iptm>50"] for m in all_metrics])
            scores.update(
                {
                    f"vendi_tm_{prefix}design_iptm_80": vendi_from_sim(
                        tm_fixed[mask_80][:, mask_80]
                    )
                    if np.sum(mask_80) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}design_iptm_70": vendi_from_sim(
                        tm_fixed[mask_70][:, mask_70]
                    )
                    if np.sum(mask_70) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}design_iptm_60": vendi_from_sim(
                        tm_fixed[mask_60][:, mask_60]
                    )
                    if np.sum(mask_60) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}design_iptm_50": vendi_from_sim(
                        tm_fixed[mask_50][:, mask_50]
                    )
                    if np.sum(mask_50) > 0
                    else 0.0,
                }
            )
        if compute_min_int_paes:
            mask_15 = np.array([m[f"min_interaction_pae<1.5"] for m in all_metrics])
            mask_2 = np.array([m[f"min_interaction_pae<2"] for m in all_metrics])
            mask_25 = np.array([m[f"min_interaction_pae<2.5"] for m in all_metrics])
            mask_3 = np.array([m[f"min_interaction_pae<3"] for m in all_metrics])
            mask_4 = np.array([m[f"min_interaction_pae<4"] for m in all_metrics])
            mask_5 = np.array([m[f"min_interaction_pae<5"] for m in all_metrics])
            scores.update(
                {
                    f"vendi_tm_{prefix}min_interaction_pae_1.5": vendi_from_sim(
                        tm_fixed[mask_15][:, mask_15]
                    )
                    if np.sum(mask_15) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}min_interaction_pae_2": vendi_from_sim(
                        tm_fixed[mask_2][:, mask_2]
                    )
                    if np.sum(mask_2) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}min_interaction_pae_2.5": vendi_from_sim(
                        tm_fixed[mask_25][:, mask_25]
                    )
                    if np.sum(mask_25) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}min_interaction_pae_3": vendi_from_sim(
                        tm_fixed[mask_3][:, mask_3]
                    )
                    if np.sum(mask_3) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}min_interaction_pae_4": vendi_from_sim(
                        tm_fixed[mask_4][:, mask_4]
                    )
                    if np.sum(mask_4) > 0
                    else 0.0,
                    f"vendi_tm_{prefix}min_interaction_pae_5": vendi_from_sim(
                        tm_fixed[mask_5][:, mask_5]
                    )
                    if np.sum(mask_5) > 0
                    else 0.0,
                }
            )

    return scores


def vendi_sequences(all_seqs: List[np.ndarray], diversity_subset: int = None) -> float:
    if diversity_subset is not None and diversity_subset < len(all_seqs):
        all_seqs = random.sample(all_seqs, diversity_subset)

    N = len(all_seqs)
    sims = np.zeros((N, N), dtype=np.float32)
    aligner = Align.PairwiseAligner()
    for i in tqdm(range(N), desc="Computing sequence diversity."):
        for j in range(i + 1, N):
            seq1 = Seq(all_seqs[i])
            seq2 = Seq(all_seqs[j])
            alignments = aligner.align(seq1, seq2)

            similarity = alignments[0].score / max(len(seq1), len(seq2))
            sims[i, j] = similarity

    return {
        "vendi_seq_sim": vendi_from_sim(sims),
    }


def compute_novelty_foldseek(
    indir: Path,
    outdir: Path,
    reference_db: Path,
    files: List[str],
    foldseek_binary: str = "/data/rbg/users/hstark/foldseek/bin/foldseek",
) -> pd.DataFrame:
    if len(files) == 0:
        return np.nan

    aln_tsv = outdir / "aln.tsv"
    tmp_dir = outdir / "tmp"

    cmd = [
        foldseek_binary,
        "easy-search",
        str(indir),
        str(reference_db),
        str(aln_tsv),
        str(tmp_dir),
        "--format-output",
        "query,target,alntmscore,qtmscore,ttmscore",
        "--alignment-type",
        "1",
        "--exhaustive-search",
        "1",
    ]

    subprocess.run(cmd, check=True)

    df = pd.read_csv(
        aln_tsv,
        sep="\t",
        names=["query", "target", "alntmscore", "qtmscore", "ttmscore"],
    )
    df["tmscore"] = (df["qtmscore"] + df["ttmscore"]) / 2
    df = df.groupby("query").max().reset_index()
    queries = [Path(f).stem for f in files]
    df = df.set_index("query").reindex(queries, fill_value=0.0).reset_index()
    df_novelty = df[["query", "tmscore"]].rename(columns={"tmscore": "novelty"})
    return df_novelty


def _radius(res_name: str, atom_name: str, element: str) -> float:
    """
    ProtOr radius with element fallback.
    """
    try:
        r = vdw_radius_protor(res_name, atom_name)
        if r is not None:
            return r
    except KeyError:
        pass
    r = vdw_radius_single(element)
    return r if r is not None else 1.8


def compute_sasa(structure_path):
    HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "TRP"}
    atoms = _load_stack(structure_path)[0]
    res_names = [
        bytes(r).decode() if isinstance(r, bytes) else str(r) for r in atoms.res_name
    ]
    atom_names = [
        bytes(a).decode() if isinstance(a, bytes) else str(a) for a in atoms.atom_name
    ]
    elements = [
        bytes(e).decode() if isinstance(e, bytes) else str(e) for e in atoms.element
    ]

    radii = np.array(
        [
            _radius(rn.strip(), an.strip(), el)
            for rn, an, el in zip(res_names, atom_names, elements)
        ],
        dtype=float,
    )
    atom_sasa = sasa(atoms, probe_radius=1.4, point_number=960, vdw_radii=radii)
    mask = (
        np.array([rn.strip() in HYDROPHOBIC_RESIDUES for rn in res_names])
        & np.char.startswith(atom_names, "C")
        & (atom_sasa > 0)
    )

    return atoms.coord[mask], atom_sasa[mask]


def largest_hydrophobic_patch_area(cif_path, distance_cutoff=6.0):
    result = compute_sasa(cif_path)
    if result is None or result[0].size == 0:
        return np.nan

    coords, sasa_vals = result
    if len(coords) == 0:
        return 0.0
    clustering = DBSCAN(eps=distance_cutoff, min_samples=1).fit(coords)
    labels = clustering.labels_

    max_patch_area = 0.0
    for label in np.unique(labels):
        area = sasa_vals[labels == label].sum()
        max_patch_area = max(max_patch_area, area)

    return max_patch_area


def get_delta_sasa(
    path,
    atom_target_mask,                
    atom_design_mask,           
):
    stack = _load_stack(path)
    atoms = stack[0]

    res = [
        r.decode().strip() if isinstance(r, bytes) else str(r).strip()
        for r in atoms.res_name
    ]
    atm = [
        a.decode().strip() if isinstance(a, bytes) else str(a).strip()
        for a in atoms.atom_name
    ]
    elem = [e.decode() if isinstance(e, bytes) else str(e) for e in atoms.element]

    radii = np.array(
        [_radius(rn, an, el) for rn, an, el in zip(res, atm, elem)], dtype=float
    )

    bound_mask = atom_design_mask | atom_target_mask
    # biotite's sasa() raises ValueError on an empty atom selection, which
    # happens whenever the target and/or design chain(s) have no resolved
    # atoms (e.g. an unresolved target in the input structure).
    if not bound_mask.any() or not atom_target_mask.any():
        return float("nan"), float("nan"), float("nan")

    atoms_bound = atoms[bound_mask]
    radii_bound = radii[bound_mask]

    area_bound = sasa(
        atoms_bound,
        probe_radius=1.4,
        point_number=960,
        vdw_radii=radii_bound,
    )
    
    target_in_bound = atom_target_mask[bound_mask]
    target_bound    = area_bound[target_in_bound].sum()
    
    

    target_atoms = atoms[atom_target_mask]
    target_res = [r for r, m in zip(res, atom_target_mask) if m]
    target_atm = [a for a, m in zip(atm, atom_target_mask) if m]
    target_elem = [e for e, m in zip(elem, atom_target_mask) if m]

    radii_lig = np.array(
        [_radius(rn, an, el) for rn, an, el in zip(target_res, target_atm, target_elem)],
        dtype=float,
    )
    target_area = sasa(
        target_atoms,
        probe_radius=1.4,
        point_number=960,
        vdw_radii=radii_lig,
    )
    delta = target_area.sum() - target_bound
    return delta, target_area.sum(), target_bound


def compute_ss_metrics(dssp_pred, ss_conditioning_metricsed):
    ss_metrics = {}
    conditioned_mask = ss_conditioning_metricsed != 0
    if conditioned_mask.sum() == 0:
        return {
            "precision_loop": float("nan"),
            "recall_loop": float("nan"),
            "accuracy_loop": float("nan"),
            "precision_helix": float("nan"),
            "recall_helix": float("nan"),
            "accuracy_helix": float("nan"),
            "precision_sheet": float("nan"),
            "recall_sheet": float("nan"),
            "accuracy_sheet": float("nan"),
            "accuracy_overall": float("nan"),
        }
    types = {1: "loop", 2: "helix", 3: "sheet"}
    TP_total, total_conditioned = 0, conditioned_mask.sum().item()
    for i, name in types.items():
        TP = ((dssp_pred == i) & (ss_conditioning_metricsed == i)).sum().item()
        FP = (
            ((dssp_pred == i) & (ss_conditioning_metricsed != i) & conditioned_mask)
            .sum()
            .item()
        )
        FN = ((dssp_pred != i) & (ss_conditioning_metricsed == i)).sum().item()
        precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
        recall = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
        accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else float("nan")
        ss_metrics[f"precision_{name}"] = precision
        ss_metrics[f"recall_{name}"] = recall
        ss_metrics[f"accuracy_{name}"] = accuracy
        TP_total += TP

    accuracy_overall = (
        TP_total / total_conditioned if total_conditioned > 0 else float("nan")
    )
    ss_metrics["accuracy_overall"] = accuracy_overall
    return ss_metrics


def filter_resolved_atoms(structure: Structure) -> Structure:
    resolved_atom_indices = np.where(structure.atoms["is_present"])[0]
    return Structure.extract_atoms(structure, resolved_atom_indices)


def save_design_only_structure_to_cif(atom_design_mask, structure, output_path: Path):
    design_atom_indices = torch.where(atom_design_mask)[0].cpu().numpy()
    design_only_str = Structure.extract_atoms(
        structure, design_atom_indices, res_reindex=True
    )
    cif_text = to_mmcif(design_only_str)
    output_path.write_text(cif_text)
    return cif_text


def save_design_only_structure_to_pdb(atom_design_mask, structure, output_path: Path):
    cif_path = output_path.with_suffix(".cif")
    cif_text = save_design_only_structure_to_cif(atom_design_mask, structure, cif_path)

    cif_io = io.StringIO(cif_text)
    mmcif_parser = PDB.MMCIFParser()
    pdb_writer = PDB.PDBIO()
    parsed_structure = mmcif_parser.get_structure("des_only", cif_io)
    pdb_writer.set_structure(parsed_structure)
    pdb_writer.save(str(output_path))


########################################################################################################
# Hydrophobycity computation functions. From Jeremie Alexander.
########################################################################################################


def calc_base_h(seq: str) -> float:
    s = seq.upper()
    n = len(s)
    if n == 0:
        return 0.0
    H = 0.0

    # position-specific coefficients
    for i, aa in enumerate(s):
        if aa not in const.hydrophobicity_info:
            raise ValueError(f"Unknown residue '{aa}' in '{seq}'")
        if i == 0:
            key = "Rc1"
        elif i == 1:
            key = "Rc2"
        elif i == n - 1:
            key = "Rn"
        elif i == n - 2:
            key = "Rn1"
        else:
            key = "Rc"
        H += const.hydrophobicity_info[aa][key]

    # nearest-neighbor penalties around H/R/K
    for i, aa in enumerate(s):
        if aa in ("H", "R", "K"):
            for j in (i - 1, i + 1):
                if 0 <= j < n and s[j] in const.nn_penalty:
                    H -= const.nn_penalty[s[j]]

    # proline run penalties
    i = 0
    while i < n:
        if s[i] == "P":
            j = i
            while j < n and s[j] == "P":
                j += 1
            run = j - i
            if run >= 4:
                H -= 5.0
            elif run == 3:
                H -= 3.5
            elif run == 2:
                H -= 1.2
            i = j
        else:
            i += 1
    return H


def apply_length_weight(H: float, n: int) -> float:
    if n < 8:
        KL = 1.0 - 0.055 * (8 - n)
    elif n > 20:
        KL = 1.0 / (1.0 + 0.027 * (n - 20))
    else:
        KL = 1.0
    return H * KL


def overall_penalty(H: float) -> float:
    if H <= 20:
        return H
    if H <= 30:
        return H - 0.27 * (H - 18.0)
    if H <= 40:
        return H - 0.33 * (H - 18.0)
    if H <= 50:
        return H - 0.38 * (H - 18.0)
    return H - 0.447 * (H - 18.0)


def calc_hydrophobicity(seq: str) -> float:
    s = (seq or "").strip().upper()
    if not s or "X" in s:
        return float("nan")
    base = calc_base_h(s)
    base = apply_length_weight(base, len(s))
    return round(overall_penalty(base), 4)


########################################################################################################
# Liability computation functions
########################################################################################################


def get_motif_set(modality="antibody", peptide_type="linear"):
    """
    Returns a dict of motif_name -> compiled regex for the given modality.
    modality: 'antibody' or 'peptide'
    peptide_type: 'linear' or 'cyclic' (only for peptide modality)


    """
    if modality == "antibody":
        return {
            "DeAmdH": re.compile(r"N[GS]"),  # High-risk deamidation
            "DeAmdM": re.compile(r"N[AHNT]"),  # Medium-risk deamidation
            "DeAmdL": re.compile(r"[STK]N"),  # Low-risk deamidation
            "Ngly": re.compile(r"N[^P][ST]"),  # N-glycosylation sites
            "Isom": re.compile(r"D[DG HST]".replace(" ", "")),  # Isomerization
            "Isomer": re.compile(r"DG|DS|DD"),  # Isomerization variants
            "FragH": re.compile(r"DP"),  # High fragmentation risk
            "FragM": re.compile(r"TS"),  # Medium fragmentation risk
            "TrpOx": re.compile(r"W"),  # Tryptophan oxidation
            "MetOx": re.compile(r"M"),  # Methionine oxidation
            "Hydro": re.compile(r"NP"),  # Hydrolysis prone
            "IntBind": re.compile(r"GPR|RGD|RYD|LDV|DGE|KGD|NGR"),  # Integrin binding
            "Polyreactive": re.compile(
                r"GGG|GG|RR|VG|VVV|WWW|YY|WxW"
            ),  # Polyreactivity
            "AggPatch": re.compile(r"FHW"),  # Aggregation patches
            "ViscPatch": re.compile(r"HYF|HWH"),  # Viscosity patches
            "DeAmdH": re.compile(r"N[GS]"),
            "DeAmdM": re.compile(r"N[AHNT]"),
            "DeAmdL": re.compile(r"[STK]N"),
            "Ngly": re.compile(r"N[^P][ST]"),
            "Isom": re.compile(r"D[DG HST]".replace(" ", "")),
            "Isomer": re.compile(r"DG|DS|DD"),
            "FragH": re.compile(r"DP"),
            "FragM": re.compile(r"TS"),
            "TrpOx": re.compile(r"W"),
            "MetOx": re.compile(r"M"),
            "Hydro": re.compile(r"NP"),
            "IntBind": re.compile(r"GPR|RGD|RYD|LDV|DGE|KGD|NGR"),
            "Polyreactive": re.compile(r"GGG|GG|RR|VG|VVV|WWW|YY|WxW"),
            "AggPatch": re.compile(r"FHW"),
            "ViscPatch": re.compile(r"HYF|HWH"),
        }
    elif modality == "peptide":
        motifs = {
            "AspBridge": re.compile(r"N[GSQA]"),  # Deamidation hotspots
            "AspCleave": re.compile(r"D[PGS]"),  # Acidic cleavage sites
            "NTCycl": re.compile(r"^[QN]"),  # N-terminal cyclization
            "ProtTryp": re.compile(r"[KR](?=.)"),  # Trypsin cleavage sites
            "DPP4": re.compile(r"^[PX]?[AP]"),  # DPP4 cleavage sites
            "MetOx": re.compile(r"M"),  # Methionine oxidation
            "TrpOx": re.compile(r"W"),  # Tryptophan oxidation
            "HydroPatch": re.compile(r"[FILVWY]{3,}"),  # Hydrophobic patches
        }
        if peptide_type == "cyclic":
            # remove N-term liabilities for cyclic peptides
            motifs.pop("NTCycl", None)
            motifs.pop("DPP4", None)
        elif peptide_type == "linear":
            # For linear peptides, we handle cysteine pairing separately
            # so we don't include CysOx in the motif set
            pass
        return motifs
    else:
        raise ValueError(f"Unknown modality: {modality}")


def severity_score(name):
    return const.liability_severity.get(name, const.default_severity)


def compute_liability_scores(sequences, modality="antibody", peptide_type="linear"):
    """
    Compute liability scores for given sequences.
    modality: 'antibody' or 'peptide'; peptide_type: 'linear' or 'cyclic'.
    For cyclic peptides, terminal CysOx flags are skipped.

    Returns:
        dict: sequence -> {'score': int, 'violations': list of dicts}
    """
    motifs = get_motif_set(modality, peptide_type)
    results = {}
    for seq in sequences:
        violations = []
        total_score = 0
        length = len(seq)
        # motif scanning
        for name, pat in motifs.items():
            for m in pat.finditer(seq):
                pos = m.start() + 1
                # skip terminal cysteines for cyclic peptides
                if (
                    modality == "peptide"
                    and peptide_type == "cyclic"
                    and name == "CysOx"
                    and pos in (1, length)
                ):
                    continue
                sev = severity_score(name)
                violations.append(
                    {"motif": name, "pos": pos, "len": len(m.group()), "severity": sev}
                )
                total_score += sev
        # antibody-specific extras
        if modality == "antibody":
            # unpaired cysteines
            cpos = [i for i, aa in enumerate(seq) if aa == "C"]
            paired = set()
            for i in range(len(cpos) - 1):
                if abs(cpos[i + 1] - cpos[i]) in (1, 2):
                    paired.update({cpos[i], cpos[i + 1]})
            for i in cpos:
                if i not in paired:
                    sev = severity_score("UnpairedCys")
                    violations.append(
                        {
                            "motif": "UnpairedCys",
                            "pos": i + 1,
                            "len": 1,
                            "severity": sev,
                        }
                    )
                    total_score += sev
            # net charge
            charge = seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")
            if charge > 1:
                sev = const.default_severity
                violations.append(
                    {"motif": "HighNetCharge", "pos": None, "len": 0, "severity": sev}
                )
                total_score += sev
        # peptide-specific extras
        elif modality == "peptide":
            # For linear peptides, only flag unpaired cysteines (odd number of cysteines)
            # For cyclic peptides, terminal cysteines are expected, so only flag internal unpaired cysteines
            cpos = [i for i, aa in enumerate(seq) if aa == "C"]
            if peptide_type == "linear":
                # For linear peptides, if there's an odd number of cysteines, flag all cysteines as potential liabilities
                # since we don't know which one is unpaired
                if len(cpos) % 2 == 1:
                    sev = severity_score("UnpairedCys")
                    for cys_pos in cpos:
                        violations.append(
                            {
                                "motif": "UnpairedCys",
                                "pos": cys_pos + 1,
                                "len": 1,
                                "severity": sev,
                            }
                        )
                        total_score += sev
            elif peptide_type == "cyclic":
                # For cyclic peptides, terminal cysteines are expected for cyclization
                # Only flag internal unpaired cysteines
                internal_cpos = [
                    i for i in cpos if i != 0 and i != len(seq) - 1
                ]  # exclude terminal positions
                if len(internal_cpos) % 2 == 1:
                    # Flag all internal cysteines as potential liabilities
                    sev = severity_score("UnpairedCys")
                    for cys_pos in internal_cpos:
                        violations.append(
                            {
                                "motif": "UnpairedCys",
                                "pos": cys_pos + 1,
                                "len": 1,
                                "severity": sev,
                            }
                        )
                        total_score += sev

                # Additional liability checks for cyclic peptides
                # 1. Check for low hydrophilic content (< 40%)
                hydrophilic_residues = (
                    seq.count("D")
                    + seq.count("E")
                    + seq.count("K")
                    + seq.count("R")
                    + seq.count("H")
                    + seq.count("N")
                    + seq.count("Q")
                    + seq.count("S")
                    + seq.count("T")
                )
                hydrophilic_percentage = (hydrophilic_residues / len(seq)) * 100
                if hydrophilic_percentage < 40:
                    sev = severity_score("LowHydrophilic")
                    violations.append(
                        {
                            "motif": "LowHydrophilic",
                            "pos": None,
                            "len": 0,
                            "severity": sev,
                            "details": f"{hydrophilic_percentage:.1f}% hydrophilic",
                        }
                    )
                    total_score += sev

                # 2. Check for consecutive identical residues
                max_consec_identical = 1
                current_consec = 1
                for i in range(1, len(seq)):
                    if seq[i] == seq[i - 1]:
                        current_consec += 1
                        max_consec_identical = max(max_consec_identical, current_consec)
                    else:
                        current_consec = 1

                if max_consec_identical > 1:
                    sev = severity_score("ConsecIdentical")
                    violations.append(
                        {
                            "motif": "ConsecIdentical",
                            "pos": None,
                            "len": 0,
                            "severity": sev,
                            "details": f"{max_consec_identical} consecutive identical",
                        }
                    )
                    total_score += sev

                # 3. Check for more than 4 consecutive hydrophobic residues
                max_consec_hydrophobic = 0
                current_consec = 0
                for aa in seq:
                    if aa in "FILVWY":
                        current_consec += 1
                        max_consec_hydrophobic = max(
                            max_consec_hydrophobic, current_consec
                        )
                    else:
                        current_consec = 0

                if max_consec_hydrophobic > 4:
                    sev = severity_score("LongHydrophobic")
                    violations.append(
                        {
                            "motif": "LongHydrophobic",
                            "pos": None,
                            "len": 0,
                            "severity": sev,
                            "details": f"{max_consec_hydrophobic} consecutive hydrophobic",
                        }
                    )
                    total_score += sev
        results[seq] = {"score": total_score, "violations": violations}
    return results


def compute_liability_metrics(sequence, liability_modality, liability_peptide_type):
    metrics = {}
    # check if sequence is valid
    if not sequence or len(sequence) == 0:
        raise ValueError(f"Sequence is empty: '{sequence}'")

    liability_results = compute_liability_scores(
        [sequence],
        modality=liability_modality,
        peptide_type=liability_peptide_type,
    )
    liability_data = liability_results[sequence]

    # Store liability metrics
    metrics["liability_score"] = liability_data["score"]
    metrics["liability_num_violations"] = len(liability_data["violations"])

    # Count violations by severity
    high_severity_violations = [
        v for v in liability_data["violations"] if v["severity"] >= 10
    ]
    medium_severity_violations = [
        v for v in liability_data["violations"] if 5 <= v["severity"] < 10
    ]
    low_severity_violations = [
        v for v in liability_data["violations"] if v["severity"] < 5
    ]

    metrics["liability_high_severity_violations"] = len(high_severity_violations)
    metrics["liability_medium_severity_violations"] = len(medium_severity_violations)
    metrics["liability_low_severity_violations"] = len(low_severity_violations)

    # Count violations by type
    violation_counts = {}
    for v in liability_data["violations"]:
        motif = v["motif"]
        violation_counts[motif] = violation_counts.get(motif, 0) + 1

    # Store individual violation type counts as metrics
    for motif, count in violation_counts.items():
        metrics[f"liability_{motif}_count"] = count

    # Add detailed violation information
    # Group violations by type for intelligent reporting
    violations_by_type = {}
    for v in liability_data["violations"]:
        motif = v["motif"]
        if motif not in violations_by_type:
            violations_by_type[motif] = []
        violations_by_type[motif].append(v)

    # Initialize default values for all motifs to ensure consistent dataframe columns
    # Use the full motif set for the configured modality/peptide_type so columns are consistent
    all_motifs = set(
        get_motif_set(
            modality=liability_modality,
            peptide_type=liability_peptide_type,
        ).keys()
    )
    for motif in all_motifs:
        # Initialize all possible fields with default values
        metrics[f"liability_{motif}_count"] = 0
        metrics[
            f"liability_{motif}_position"
        ] = -1  # use -1 for no position (keeps int dtype)
        metrics[f"liability_{motif}_length"] = 0
        metrics[f"liability_{motif}_severity"] = 0
        metrics[f"liability_{motif}_details"] = ""
        metrics[f"liability_{motif}_positions"] = ""
        metrics[f"liability_{motif}_num_positions"] = 0
        metrics[f"liability_{motif}_global_details"] = ""
        metrics[f"liability_{motif}_avg_severity"] = 0.0

    # Store detailed violation information
    for motif, motif_violations in violations_by_type.items():
        if len(motif_violations) == 1:
            # Single violation - store all details
            v = motif_violations[0]
            # Ensure position is an integer; use -1 for non-positional violations
            metrics[f"liability_{motif}_position"] = (
                int(v["pos"]) if v["pos"] is not None else -1
            )
            metrics[f"liability_{motif}_length"] = v["len"]
            metrics[f"liability_{motif}_severity"] = v["severity"]
            if "details" in v:
                metrics[f"liability_{motif}_details"] = v["details"]
            else:
                metrics[f"liability_{motif}_details"] = ""
        else:
            # Multiple violations - store summary information
            positions = [v["pos"] for v in motif_violations if v["pos"] is not None]
            global_violations = [v for v in motif_violations if v["pos"] is None]

            if positions:
                # Store position range for positional violations
                metrics[f"liability_{motif}_positions"] = (
                    f"{min(positions)}-{max(positions)}"
                )
                metrics[f"liability_{motif}_num_positions"] = len(positions)
            else:
                metrics[f"liability_{motif}_positions"] = ""
                metrics[f"liability_{motif}_num_positions"] = 0

            if global_violations:
                # Store details for global violations
                details = [
                    v.get("details", "") for v in global_violations if v.get("details")
                ]
                if details:
                    metrics[f"liability_{motif}_global_details"] = "; ".join(details)
                else:
                    metrics[f"liability_{motif}_global_details"] = ""
            else:
                metrics[f"liability_{motif}_global_details"] = ""

            # Store average severity
            avg_severity = sum(v["severity"] for v in motif_violations) / len(
                motif_violations
            )
            metrics[f"liability_{motif}_avg_severity"] = round(avg_severity, 1)

    # Add a comprehensive violation summary for easy interpretation
    violation_summary = []
    for motif, motif_violations in violations_by_type.items():
        count = len(motif_violations)
        if count == 1:
            v = motif_violations[0]
            if v["pos"] is not None:
                violation_summary.append(f"{motif}(pos{v['pos']},sev{v['severity']})")
            else:
                details = v.get("details", "")
                violation_summary.append(
                    f"{motif}({details},sev{v['severity']})"
                    if details
                    else f"{motif}(sev{v['severity']})"
                )
        else:
            positions = [v["pos"] for v in motif_violations if v["pos"] is not None]
            if positions:
                violation_summary.append(
                    f"{motif}x{count}(pos{min(positions)}-{max(positions)},sev{motif_violations[0]['severity']})"
                )
            else:
                violation_summary.append(
                    f"{motif}x{count}(sev{motif_violations[0]['severity']})"
                )

    metrics["liability_violations_summary"] = (
        "; ".join(violation_summary) if violation_summary else ""
    )
    return metrics
