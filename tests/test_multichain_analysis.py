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
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_pdf import RendererPdf
from matplotlib.figure import Figure
from matplotlib.legend import Legend
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
    sequence_identity,
)
from boltzgen.task.filter import filter as filter_module
from boltzgen.task.filter.filter import Filter
from boltzgen.task.filter.seqplot_utils import plot_seq_liabilities


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


@pytest.mark.parametrize("chains", [("NANANA",), ("NANANA", "ANA")])
def test_repeated_motif_severity_survives_analysis_and_csv(
    chains: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, metrics = _analyze(tmp_path, monkeypatch, chains)
    assert metrics["liability_DeAmdM_severity"] == 5
    if len(chains) > 1:
        assert metrics["liability_DeAmdM_severity_0"] == 5
        assert metrics["liability_DeAmdM_severity_2"] == 5
    task.aggregate_metrics()
    row = pd.read_csv(tmp_path / "aggregate_metrics_test.csv").iloc[0]
    assert row["liability_DeAmdM_severity"] == 5
    assert "sev0" not in row["liability_details"]


@pytest.mark.parametrize("chains", [("CGWM", "GWM"), ("AGWM", "GWM")])
def test_scanned_chain_has_zero_for_absent_motif(
    chains: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, metrics = _analyze(tmp_path, monkeypatch, chains)
    expected = int(chains[0].startswith("C"))
    assert metrics["liability_UnpairedCys_count_0"] == expected
    assert metrics["liability_UnpairedCys_count_2"] == 0
    assert metrics["liability_UnpairedCys_position_2"] == -1
    assert metrics["liability_UnpairedCys_length_2"] == 0
    assert metrics["liability_UnpairedCys_severity_2"] == 0
    assert metrics["liability_UnpairedCys_num_positions_2"] == 0
    assert metrics["liability_UnpairedCys_avg_severity_2"] == 0.0
    assert metrics["liability_UnpairedCys_details_2"] == ""
    assert metrics["liability_UnpairedCys_positions_2"] == ""
    assert metrics["liability_UnpairedCys_global_details_2"] == ""
    task.aggregate_metrics()
    row = pd.read_csv(tmp_path / "aggregate_metrics_test.csv").iloc[0]
    assert row["liability_UnpairedCys_count_2"] == 0
    assert row["liability_UnpairedCys_count"] == expected


def test_per_chain_zero_count_filter_distinguishes_absent_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = []
    for index, chains in enumerate((("C", "A"), ("A", "C"), ("A", "G"), ("A",))):
        _, metrics = _analyze(tmp_path / str(index), monkeypatch, chains)
        metrics["id"] = str(index)
        rows.append(metrics)
    rule = {
        "feature": "liability_UnpairedCys_count_2",
        "lower_is_better": True,
        "threshold": 0,
    }
    task = _load_filter(tmp_path, pd.DataFrame(rows), [rule])
    task.filter_df()
    passed = dict(
        zip(
            task.df["id"].astype(str),
            task.df["pass_liability_UnpairedCys_count_2_filter"],
        )
    )
    assert passed == {"0": True, "1": False, "2": True, "3": False}


@pytest.mark.parametrize(
    ("modality", "peptide_type", "extras"),
    [
        ("antibody", "linear", ("UnpairedCys", "HighNetCharge")),
        ("peptide", "linear", ("UnpairedCys",)),
        (
            "peptide",
            "cyclic",
            ("UnpairedCys", "LowHydrophilic", "ConsecIdentical", "LongHydrophobic"),
        ),
    ],
)
def test_supplemental_scan_counts_are_zero_when_no_violation_exists(
    modality: str, peptide_type: str, extras: tuple[str, ...]
) -> None:
    raw = compute_liability_scores(["GE"], modality, peptide_type)["GE"]
    assert raw["violations"] == []
    metrics = compute_liability_metrics("GE", modality, peptide_type)
    for motif in extras:
        assert metrics[f"liability_{motif}_count"] == 0
        assert metrics[f"liability_{motif}_position"] == -1
        assert metrics[f"liability_{motif}_severity"] == 0


def test_hydrophobicity_weights_unequal_full_and_designed_chain_lengths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, metrics = _analyze(tmp_path, monkeypatch, ("AAAAAAAAAA", "W"))
    assert metrics["design_chain_hydrophobicity"] == pytest.approx(
        (10 * calc_hydrophobicity("AAAAAAAAAA") + calc_hydrophobicity("W")) / 11
    )
    assert metrics["design_hydrophobicity"] == pytest.approx(
        (2 * calc_hydrophobicity("AA") + calc_hydrophobicity("W")) / 3
    )


@pytest.mark.parametrize(("left", "right"), [("AA", "AA:GG"), ("AA:GG", "AA")])
def test_missing_chain_contributes_length_without_matches(
    left: str, right: str
) -> None:
    assert sequence_identity(left, right) == 0.5


def test_legacy_archive_without_sequence_retains_flat_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task, _ = _analyze(tmp_path, monkeypatch, ("AAN", "GWM"))
    path = task.metrics_dir / "data_target_model_0.npz"
    with np.load(path, allow_pickle=True) as archive:
        legacy = {key: value for key, value in archive.items() if key != "sequence"}
    np.savez_compressed(path, **legacy)
    task.aggregate_metrics()
    sequences = pd.read_pickle(tmp_path / "ca_coords_sequences.pkl.gz")
    assert sequences.loc[0, "sequence"] == "ANWM"


def test_native_recovery_excludes_nonprotein_and_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feat = _features(
        tmp_path / "target_model_0.cif", ("AAA",), ligand=True, padded=True
    )
    feat["design_mask"] = torch.ones_like(feat["design_mask"])
    native = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in feat.items()
    }
    native["res_type"][1] = one_hot(
        torch.tensor(const.token_ids["GLY"]), len(const.tokens)
    ).float()
    feat.update({f"native_{key}": value for key, value in native.items()})
    data = SimpleNamespace(
        cfg=SimpleNamespace(target_id_regex=r"(target)"),
        predict_set=SimpleNamespace(get_sample=lambda **_: feat),
        return_native=True,
    )
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    task = Analyze(
        "test",
        data,
        design_dir=str(tmp_path),
        allatom_fold_metrics=False,
        compute_lddts=False,
        native=True,
        sequence_recovery=True,
    )
    assert task.compute_metrics(sample_id=feat["id"]) == feat["id"]
    with np.load(task.metrics_dir / "metrics_target_model_0.npz") as archive:
        assert archive["seq_recovery"].item() == pytest.approx(2 / 3)


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
    if override:
        # A second pipeline in the same process must start from protocol defaults.
        args.config = None
        next_steps = {
            step.name: step.get_config()
            for step in cli.BinderDesignPipeline(args, Path("/mols")).steps
        }
        default = "peptide" if protocol == "peptide-anything" else "antibody"
        assert next_steps["analysis"].liability_modality == default
        assert next_steps["filtering"].modality == default


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


@pytest.mark.parametrize("length", [12, 40, 41, 120, 240])
def test_liability_pdf_keeps_residue_glyphs_legible(
    length: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = Filter(str(tmp_path), use_affinity=True, num_liability_plots=1)
    task.df = pd.DataFrame(
        [
            {
                "id": "layout",
                "designed_sequence": "W" * 11,
                "designed_chain_sequence": "W" * length,
            }
        ]
    )
    task.df_div = task.df.copy()
    task.filters = [{"feature": "id", "lower_is_better": True, "threshold": "z"}]
    original_save = filter_module.PdfPages.savefig
    measured = []

    def record(pdf: filter_module.PdfPages, figure: Figure) -> None:
        if any(ax.get_title().startswith("Qualityrank") for ax in figure.axes):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            count, overlaps, fonts = 0, 0, []
            for ax in figure.axes:
                letters = [text for text in ax.texts if text.get_text() == "W"]
                boxes = [text.get_window_extent(renderer) for text in letters]
                count += len(letters)
                fonts.extend(text.get_fontsize() for text in letters)
                overlaps += sum(
                    left.x1 > right.x0 + 0.5 for left, right in zip(boxes, boxes[1:])
                )
            measured.append((count, overlaps, min(fonts)))
        original_save(pdf, figure)

    monkeypatch.setattr(filter_module.PdfPages, "savefig", record)
    task.make_visualization([], [], [], [], [["score", 1]], "test", [["id", "ID"]])
    assert measured == [(length, 0, 12)]


def test_wrapped_liability_plot_preserves_full_sequence_coordinates() -> None:
    # One marker crosses the visual row boundary; scoring must not restart there.
    figure = plot_seq_liabilities(
        "A" * 85,
        "layout",
        [{"motif": "marker", "pos": 38, "len": 6, "severity": 5}],
        total_score=5,
    )
    letters = [
        text for ax in figure.axes for text in ax.texts if text.get_text() == "A"
    ]
    assert len(letters) == 85
    colors = [text.get_bbox_patch().get_facecolor() for text in letters]
    assert all(color == colors[0] for color in colors[:37] + colors[43:])
    assert all(color == colors[37] for color in colors[37:43])
    assert colors[37] != colors[0]
    assert any("Score: 5" in ax.get_title() for ax in figure.axes)
    labels = [text.get_text() for ax in figure.axes for text in ax.texts]
    assert "Residues 1-40" in labels
    assert "Residues 41-80" in labels
    assert "Residues 81-85" in labels
    filter_module.plt.close(figure)


@pytest.mark.parametrize("length", [120, 240, 1000])
def test_dense_liability_legend_fits_saved_pdf(
    length: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise the largest supported legend through the real scanner and report.
    sequence = "NGNASNNSTDDDPTSWMNPRGDGGFHWHYFCKKKKKK"
    sequence = sequence.ljust(length, "A")
    task = Filter(
        str(tmp_path), use_affinity=True, num_liability_plots=1, modality="antibody"
    )
    task.df = pd.DataFrame(
        [
            {
                "id": "legend",
                "designed_sequence": "AA",
                "designed_chain_sequence": sequence,
            }
        ]
    )
    task.df_div = task.df.copy()
    task.filters = [{"feature": "id", "lower_is_better": True, "threshold": "z"}]
    original_draw = Legend.draw
    measured = []

    def record(legend: Legend, renderer: RendererBase) -> None:
        original_draw(legend, renderer)
        if not isinstance(getattr(renderer, "_vector_renderer", renderer), RendererPdf):
            return
        figure = legend.figure
        frame = legend.get_window_extent(renderer)
        label = figure.axes[-2].xaxis.label.get_window_extent(renderer)
        overlaps = 0
        for ax in figure.axes[:-2]:
            ranges = [
                text for text in ax.texts if text.get_text().startswith("Residues ")
            ]
            letters = [text for text in ax.texts if len(text.get_text()) == 1]
            overlaps += sum(
                title.get_window_extent(renderer).overlaps(
                    letter.get_window_extent(renderer)
                )
                for title in ranges
                for letter in letters
            )
        measured.append(
            (
                len(legend.get_texts()),
                frame.y0 - figure.bbox.y0,
                label.y0 - frame.y1,
                overlaps,
            )
        )

    monkeypatch.setattr(Legend, "draw", record)
    task.make_visualization([], [], [], [], [["score", 1]], "test", [["id", "ID"]])
    assert measured
    for entries, bottom_margin, label_gap, overlaps in measured:
        assert entries == 17
        assert bottom_margin >= 0
        assert label_gap >= 0
        assert overlaps == 0


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


def test_logo_cohorts_preserve_sparse_rows_and_duplicate_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = Filter(
        str(tmp_path),
        use_affinity=True,
        plot_seq_logos=True,
        top_budget=2,
        modality="antibody",
    )
    task.df = pd.DataFrame(
        [
            {
                "id": "first",
                "designed_sequence": "AN:WM",
                "designed_chain_sequence": "AAAAN:GWM",
                "full_sequence_0": "AAAAN",
                "full_sequence_2": "GWM",
                "designed_sequence_0": "AN",
                "designed_sequence_2": "WM",
            },
            {
                "id": "second",
                "designed_sequence": "AD:YF",
                "designed_chain_sequence": "AAAAD:GYF",
                "full_sequence_0": "AAAAD",
                "full_sequence_2": "GYF",
                "designed_sequence_0": "AD",
                "designed_sequence_2": "YF",
            },
            {"id": "legacy", "designed_sequence": "G", "designed_chain_sequence": "AG"},
        ],
        index=[7, 7, 7],
    )
    task.df_div = task.df.iloc[[2, 1, 1]].copy()
    task.filters = [{"feature": "id", "lower_is_better": True, "threshold": "z"}]
    logos, scaffolds = [], []

    def logo(sequences: list[str], title: str) -> None:
        logos.append((sequences, title.split(maxsplit=1)[0]))

    def cdr(sequences: list[str], _title: str) -> None:
        scaffolds.append(sequences)

    monkeypatch.setattr(filter_module, "create_alignment_logo", logo)
    monkeypatch.setattr(filter_module, "aa_composition_pie", lambda *_: None)
    monkeypatch.setattr(filter_module, "cdr_logo", cdr)
    task.make_visualization([], [], [], [], [["score", 1]], "test", [["id", "ID"]])
    assert logos == [
        (["AN", "AD"], "All"),
        (["AN", "AD"], "Top"),
        (["AD", "AD"], "Diverse"),
        (["GWM", "GYF"], "All"),
        (["GWM", "GYF"], "Top"),
        (["GYF", "GYF"], "Diverse"),
        (["G"], "All"),
        (["G"], "Diverse"),
    ]
    assert scaffolds == [
        ["AAAAN", "AAAAD"],
        ["AAAAN", "AAAAD"],
        ["AAAAD", "AAAAD"],
        ["GWM", "GYF"],
        ["GWM", "GYF"],
        ["GYF", "GYF"],
        ["AG"],
        ["AG"],
    ]


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


@pytest.mark.parametrize("chains", [("NA",), ("NA", "GWM"), ("AANA", "GWM")])
@pytest.mark.parametrize("merge", [False, True])
def test_valid_na_sequences_survive_csv_merge_and_reporting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chains: tuple[str, ...],
    merge: bool,
) -> None:
    source = tmp_path / "run"
    designs = source / "intermediate_designs"
    designs.mkdir(parents=True)
    analysis, metrics = _analyze(designs, monkeypatch, chains, modality="peptide")
    metrics.update(
        bb_rmsd=0.0,
        bb_rmsd_design=0.0,
        min_interaction_pae=1.0,
        esmfold2_input_hash=float("nan"),
    )
    np.savez(analysis.metrics_dir / f"metrics_{metrics['id']}.npz", **metrics)
    analysis.aggregate_metrics()
    (designs / "aggregate_metrics_test.csv").rename(
        designs / "aggregate_metrics_analyze.csv"
    )
    (designs / metrics["file_name"]).write_text("coordinate copy fixture")
    if merge:
        output = tmp_path / "merged"
        cli.merge_command(SimpleNamespace(sources=[source], output=output))
        designs = output / designs.name

    task = Filter(
        str(designs),
        use_affinity=True,
        plot_seq_logos=True,
        num_liability_plots=1,
        modality="peptide",
    )
    task.load_dataframe()
    row = task.df.iloc[0]
    assert row["designed_sequence"] == ":".join(seq[-2:] for seq in chains)
    assert row["designed_chain_sequence"] == ":".join(chains)
    assert not row["has_x"]
    assert pd.isna(row["esmfold2_input_hash"])
    assert row["bb_rmsd"] == 0.0
    if len(chains) > 1:
        for index, seq in enumerate(chains):
            assert row[f"full_sequence_{index * 2}"] == seq
            assert row[f"designed_sequence_{index * 2}"] == seq[-2:]
    task.df_div = task.df.copy()
    task.filters = [{"feature": "id", "lower_is_better": True, "threshold": "z"}]
    task.make_visualization([], [], [], [], [["score", 1]], "test", [["id", "ID"]])
    assert (task.outdir / "results_overview.pdf").stat().st_size > 1000


def test_pdf_composition_handles_unknown_full_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, metrics = _analyze(tmp_path, monkeypatch, ("AAA", "XXX"))
    _, other = _analyze(tmp_path, monkeypatch, ("AGA", "XXX"))
    other["id"] = "target_model_1"
    task = _load_filter(tmp_path, pd.DataFrame([metrics, other]), [])
    task.plot_seq_logos = True
    task.df_div = task.df.copy()
    task.filters = [{"feature": "id", "lower_is_better": True, "threshold": "z"}]
    task.make_visualization([], [], [], [], [["score", 1]], "test", [["id", "ID"]])
    assert (task.outdir / "results_overview.pdf").stat().st_size > 1000


def test_filter_still_rejects_missing_protein_sequence(tmp_path: Path) -> None:
    task = _load_filter(
        tmp_path,
        pd.DataFrame({"designed_sequence": [""], "designed_chain_sequence": [""]}),
        [],
    )
    assert task.df.iloc[0]["has_x"]
