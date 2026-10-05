"""Exercise chain-aware sequence analysis through CSV, filtering and PDF output."""
# ruff: noqa: INP001, PLR2004, CPY001, S301
# Pickles in these tests are generated locally by the analysis under test.

import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from Bio import Align
from matplotlib.figure import Figure
from test_atom_confidence_export import _real_confidence_features
from test_filter_rule_integrity import _load_filter
from torch.nn.functional import one_hot

from boltzgen.cli import boltzgen as cli
from boltzgen.data import const
from boltzgen.task.analyze.analyze import Analyze
from boltzgen.task.analyze.analyze_utils import (
    calc_hydrophobicity,
    compute_liability_metrics,
    compute_liability_scores,
)
from boltzgen.task.filter import filter as filter_module
from boltzgen.task.filter.filter import Filter


def _features(
    path: Path,
    chains: tuple[str, ...],
    *,
    ligand: bool = False,
    padded: bool = False,
) -> dict:
    """Supply normal unbatched generated features, with only two designs per chain."""
    letters = "".join(chains)
    types = [const.token_ids[const.prot_letter_to_token[aa]] for aa in letters]
    asym = [index * 2 for index, seq in enumerate(chains) for _ in seq]
    designed = [i >= len(seq) - 2 for seq in chains for i in range(len(seq))]
    mol_type = [const.chain_type_ids["PROTEIN"]] * len(letters)
    if ligand:
        types.append(const.token_ids["UNK"])
        asym.append(9)
        designed.append(True)
        mol_type.append(const.chain_type_ids["NONPOLYMER"])
    real = len(types)
    if padded:
        types += [const.token_ids["GLY"]] * 2
        asym += [0, 0]
        designed += [True, True]
        mol_type += [const.chain_type_ids["PROTEIN"]] * 2
    n = len(types)
    present = torch.arange(n) < real
    protein = torch.tensor(mol_type) == const.chain_type_ids["PROTEIN"]
    backbone = (present & protein).repeat_interleave(4)
    return {
        "id": path.stem,
        "path": path,
        "exception": False,
        "res_type": one_hot(torch.tensor(types), len(const.tokens)).float(),
        "asym_id": torch.tensor(asym),
        "mol_type": torch.tensor(mol_type),
        "design_mask": torch.tensor(designed),
        "chain_design_mask": present.clone(),
        "token_pad_mask": present,
        "token_resolved_mask": present,
        "atom_resolved_mask": present.repeat_interleave(4),
        "atom_pad_mask": present.repeat_interleave(4),
        "atom_to_token": torch.eye(n).repeat_interleave(4, dim=0),
        "coords": torch.arange(n * 12, dtype=torch.float).reshape(1, n * 4, 3),
        "backbone_mask": backbone,
        "binding_type": torch.zeros(n),
    }


def _analyze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chains: tuple[str, ...],
    *,
    modality: str = "antibody",
    peptide_type: str = "linear",
    **kwargs: bool,
) -> tuple[Analyze, dict]:
    features = _features(tmp_path / "target_model_0.cif", chains, **kwargs)
    data = SimpleNamespace(
        cfg=SimpleNamespace(target_id_regex=r"(target)"),
        predict_set=SimpleNamespace(get_sample=lambda **_: features),
        return_native=False,
    )
    # Thread setup is process-wide and is unrelated to the analysis contract.
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    task = Analyze(
        "test",
        data,
        design_dir=str(tmp_path),
        allatom_fold_metrics=False,
        liability_analysis=True,
        liability_modality=modality,
        liability_peptide_type=peptide_type,
        compute_lddts=False,
    )
    assert task.compute_metrics(sample_id=features["id"]) == features["id"]
    with np.load(task.metrics_dir / f"metrics_{features['id']}.npz") as archive:
        metrics = {key: value.item() for key, value in archive.items()}
    return task, metrics


def test_analysis_includes_all_chains_without_cross_chain_motifs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, metrics = _analyze(tmp_path, monkeypatch, ("AAN", "GWM"), padded=True)
    assert metrics["designed_sequence"] == "AN:WM"
    assert metrics["designed_chain_sequence"] == "AAN:GWM"
    assert metrics["full_sequence_0"] == "AAN"
    assert metrics["full_sequence_2"] == "GWM"
    expected = [compute_liability_scores([seq])[seq] for seq in ("AAN", "GWM")]
    assert metrics["liability_score"] == sum(value["score"] for value in expected)
    assert metrics["liability_MetOx_count"] == 1
    assert metrics["liability_DeAmdH_count"] == 0  # N|G is not a peptide bond.
    assert (
        compute_liability_metrics("GWM", "antibody", "linear").keys() <= metrics.keys()
    )
    assert metrics["liability_MetOx_position"] == -1
    assert metrics["liability_MetOx_position_2"] == 3
    assert "chain 2" in metrics["liability_violations_summary"]
    assert metrics["design_chain_hydrophobicity"] == pytest.approx(
        (calc_hydrophobicity("AAN") + calc_hydrophobicity("GWM")) / 2,
    )
    assert metrics["design_hydrophobicity"] == pytest.approx(
        (calc_hydrophobicity("AN") + calc_hydrophobicity("WM")) / 2,
    )
    task.aggregate_metrics()
    frame = pd.read_csv(tmp_path / "aggregate_metrics_test.csv")
    assert frame.loc[0, "designed_chain_sequence"] == "AAN:GWM"
    assert "chain 2" in frame.loc[0, "liability_details"]
    sequences = pd.read_pickle(tmp_path / "ca_coords_sequences.pkl.gz")
    assert sequences.loc[0, "sequence"] == "AN:WM"


def test_single_chain_sequence_metrics_remain_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, metrics = _analyze(tmp_path, monkeypatch, ("GWM",))
    assert metrics["designed_sequence"] == "WM"
    assert metrics["designed_chain_sequence"] == "GWM"
    assert metrics["liability_MetOx_count"] == 1
    assert metrics["design_chain_hydrophobicity"] == calc_hydrophobicity("GWM")
    assert metrics["design_hydrophobicity"] == calc_hydrophobicity("WM")
    for key, expected in compute_liability_metrics("GWM", "antibody", "linear").items():
        assert metrics[key] == expected
    assert metrics["liability_MetOx_count"] == 1


def test_designed_nonprotein_tokens_are_not_amino_acids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, metrics = _analyze(tmp_path, monkeypatch, ("GWM",), ligand=True)
    assert metrics["designed_sequence"] == "WM"
    assert metrics["designed_chain_sequence"] == "GWM"
    task.aggregate_metrics()
    sequences = pd.read_pickle(tmp_path / "ca_coords_sequences.pkl.gz")
    assert sequences.loc[0, "sequence"] == "WM"


@pytest.mark.parametrize(
    "protocol", ["nanobody-anything", "antibody-anything", "peptide-anything"]
)
@pytest.mark.parametrize("override", [False, True])
def test_protocol_analysis_and_reporting_agree(
    protocol: str, override: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(
        cli,
        "get_artifact_path",
        lambda _args, artifact: Path("/weights") / artifact.rsplit(":", 1)[-1],
    )
    args = cli.build_parser().parse_args(
        [
            "run",
            "input.yaml",
            "--output",
            str(tmp_path),
            "--protocol",
            protocol,
            "--devices",
            "1",
        ]
        + (
            [
                "--config",
                "analysis",
                "liability_modality=peptide",
                "--config",
                "filtering",
                "modality=peptide",
            ]
            if override
            else []
        )
    )
    steps = {
        step.name: step.get_config()
        for step in cli.BinderDesignPipeline(args, Path("/mols")).steps
    }
    expected = "peptide" if override or protocol == "peptide-anything" else "antibody"
    assert steps["analysis"].liability_modality == expected
    assert steps["filtering"].modality == expected


def test_pdf_liabilities_use_full_chains_instead_of_joined_cdrs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = Filter(
        str(tmp_path), use_affinity=True, num_liability_plots=1, modality="antibody"
    )
    task.df = pd.DataFrame(
        [
            {
                "id": "paired",
                "designed_sequence": "AN:WM",
                "designed_chain_sequence": "AAN:GWM",
                "full_sequence_0": "AAN",
                "full_sequence_2": "GWM",
                "designed_sequence_0": "AN",
                "designed_sequence_2": "WM",
            }
        ]
    )
    task.df_div = task.df.copy()
    task.filters = [{"feature": "id", "lower_is_better": True, "threshold": "z"}]
    observed = []
    original_plot = filter_module.plot_seq_liabilities

    def record(
        seq: str, title: str, violations: list[dict], *, total_score: int
    ) -> Figure:
        observed.append((seq, title, violations, total_score))
        return original_plot(seq, title, violations, total_score=total_score)

    monkeypatch.setattr(filter_module, "plot_seq_liabilities", record)
    task.make_visualization(
        [], [], [], [], [["score", 1]], "test", [["id", "design ID"]]
    )
    assert [value[0] for value in observed] == ["AAN", "GWM"]
    assert "chain 2" in observed[1][1]
    assert not any(
        v["motif"] == "DeAmdH" for _, _, violations, _ in observed for v in violations
    )
    assert (task.outdir / "results_overview.pdf").stat().st_size > 1000


@pytest.mark.parametrize(
    ("chains", "modality", "peptide_type"),
    [
        (("NAK", "NAK"), "peptide", "linear"),
        (("CC", "C"), "antibody", "linear"),
        (("AN", "GS"), "antibody", "linear"),
        (("VV", "VV"), "peptide", "cyclic"),
    ],
)
def test_per_chain_terminal_cysteine_and_global_liabilities(
    chains: tuple[str, ...],
    modality: str,
    peptide_type: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, actual = _analyze(
        tmp_path, monkeypatch, chains, modality=modality, peptide_type=peptide_type
    )
    reference = [
        compute_liability_scores([seq], modality, peptide_type)[seq] for seq in chains
    ]
    assert actual["liability_score"] == sum(item["score"] for item in reference)
    assert actual["liability_num_violations"] == sum(
        len(item["violations"]) for item in reference
    )
    for motif in {v["motif"] for item in reference for v in item["violations"]}:
        assert actual[f"liability_{motif}_count"] == sum(
            v["motif"] == motif for item in reference for v in item["violations"]
        )


def test_chain_boundaries_survive_dedup_ranking_and_diversity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(
        {
            "designed_sequence": ["A:GG", "AG:G", "A:GG"],
            "designed_chain_sequence": ["A:GG", "AG:G", "A:GG"],
            "num_design": [3, 3, 3],
            "quality": [1.0, 0.5, 0.0],
        }
    )
    task = _load_filter(tmp_path, frame, [])
    assert len(task.df) == 2
    task.filter_df()
    task.sort_df()
    assert set(task.df["designed_sequence"]) == {"A:GG", "AG:G"}
    pd.DataFrame(
        {"id": task.df["id"], "sequence": task.df["designed_sequence"]}
    ).to_pickle(
        tmp_path / "ca_coords_sequences.pkl.gz",
    )
    measured = []
    real_selection = task.select_lazy_greedy

    def observe(
        k: int, quality: np.ndarray, sim_fn: Callable[[int, int], float]
    ) -> list[int]:
        measured.append(sim_fn(0, 1))
        return real_selection(k, quality, sim_fn)

    monkeypatch.setattr(task, "select_lazy_greedy", observe)
    task.optimize_diversity()
    aligner = Align.PairwiseAligner()
    expected = (aligner.score("A", "AG") + aligner.score("GG", "G")) / 4
    assert measured == [pytest.approx(expected)]
    assert measured[0] != aligner.score("A:GG", "AG:G") / 4
    assert set(task.df_div["designed_sequence"]) == {"A:GG", "AG:G"}


@pytest.mark.parametrize("modality", ["peptide", "antibody"])
def test_sequence_logos_keep_chains_separate_and_choose_each_scaffold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
) -> None:
    task = Filter(
        str(tmp_path),
        use_affinity=True,
        plot_seq_logos=True,
        top_budget=2,
        budget=2,
        modality=modality,
    )
    task.df = pd.DataFrame(
        [
            {
                "id": name,
                "designed_sequence": "AN:WM",
                "designed_chain_sequence": "AAAAN:GWM",
                "full_sequence_0": "AAAAN",
                "full_sequence_2": "GWM",
                "designed_sequence_0": "AN",
                "designed_sequence_2": "WM",
            }
            for name in ("one", "two")
        ]
    )
    task.df_div = task.df.copy()
    task.filters = [{"feature": "id", "lower_is_better": True, "threshold": "z"}]
    original_logo = filter_module.create_alignment_logo
    observed = []

    def record(sequences: list[str], title: str) -> Figure | None:
        observed.append((sequences, title))
        return original_logo(sequences, title)

    # Exercise the real optional dependency boundary without an antibody model.
    monkeypatch.setitem(sys.modules, "abnumber", None)
    original_cdr = filter_module.cdr_logo
    cdr_sequences = []

    def record_cdr(sequences: list[str], title: str) -> Figure | None:
        cdr_sequences.append(sequences)
        return original_cdr(sequences, title)

    monkeypatch.setattr(filter_module, "cdr_logo", record_cdr)
    monkeypatch.setattr(filter_module, "create_alignment_logo", record)
    task.make_visualization(
        [], [], [], [], [["score", 1]], "test", [["id", "design ID"]]
    )
    assert len(observed) == 6
    assert all(
        sequences == ["AN", "AN"] for sequences, title in observed if "chain 0" in title
    )
    assert all(
        sequences == ["GWM", "GWM"]
        for sequences, title in observed
        if "chain 2" in title
    )
    assert all(":" not in seq for sequences, _ in observed for seq in sequences)
    if modality == "antibody":
        assert cdr_sequences == [["AAAAN", "AAAAN"]] * 3 + [["GWM", "GWM"]] * 3
    else:
        assert not cdr_sequences


def test_real_refolding_analysis_uses_all_sequence_chains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = _real_confidence_features(None)
    features.update(
        {
            "id": "target",
            "path": tmp_path / "target.cif",
            "asym_id": torch.tensor([0, 2]),
            "design_mask": torch.ones(2, dtype=torch.bool),
            "chain_design_mask": torch.ones(2, dtype=torch.bool),
            "res_type": one_hot(
                torch.tensor([const.token_ids["TRP"], const.token_ids["MET"]]),
                len(const.tokens),
            ).float(),
        }
    )
    data = SimpleNamespace(
        cfg=SimpleNamespace(target_id_regex=r"(target)"),
        return_native=False,
        predict_set=SimpleNamespace(get_sample=lambda **_: features),
    )
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    task = Analyze("test", data, design_dir=str(tmp_path), compute_lddts=False)
    folded_dir = tmp_path / const.folding_dirname
    folded_dir.mkdir()
    values = {key: np.array([0.8]) for key in const.eval_keys_confidence}
    np.savez(
        folded_dir / "target.npz",
        **values,
        coords=features["coords"],
        res_type=features["res_type"],
    )
    assert task.compute_metrics(sample_id="target") == "target"
    with np.load(task.metrics_dir / "metrics_target.npz") as archive:
        assert archive["rmsd"] == pytest.approx(0, abs=1e-5)
        assert archive["design_hydrophobicity"] == pytest.approx(
            (calc_hydrophobicity("W") + calc_hydrophobicity("M")) / 2
        )
        assert (
            archive["design_chain_hydrophobicity"] == archive["design_hydrophobicity"]
        )


def test_nonprotein_only_design_has_no_protein_liability_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, metrics = _analyze(tmp_path, monkeypatch, (), ligand=True)
    assert metrics["designed_sequence"] == ""
    assert metrics["designed_chain_sequence"] == ""
    assert "liability_score" not in metrics
    assert np.isnan(metrics["design_hydrophobicity"])
    task.aggregate_metrics()
    sequences = pd.read_pickle(tmp_path / "ca_coords_sequences.pkl.gz")
    assert sequences.loc[0, "sequence"] == ""


def test_diversity_size_buckets_count_residues_not_chain_delimiters(
    tmp_path: Path,
) -> None:
    task = Filter(
        str(tmp_path),
        use_affinity=True,
        budget=2,
        alpha=0,
        size_buckets=[{"min": 3, "max": 4, "num_designs": 1}],
    )
    task.df = pd.DataFrame(
        {"id": ["first", "same_size", "longer"], "quality_score": [1.0, 0.9, 0.8]},
    )
    pd.DataFrame(
        {"id": task.df["id"], "sequence": ["A:GG", "AG:G", "AAAA:GG"]},
    ).to_pickle(tmp_path / "ca_coords_sequences.pkl.gz")
    task.optimize_diversity()
    assert task.df_div["id"].tolist() == ["first", "longer"]
