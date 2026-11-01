"""CSV text identities survive real merge, filtering and Foldseek consumers."""

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_filter_rule_integrity import _load_filter
from test_merge_resume_integrity import _merge, _pair

from boltzgen.task.analyze.analyze import Analyze
from boltzgen.task.analyze.analyze_utils import compute_novelty_foldseek


IDS = ["001", "NA", "null", "nan", "quoted,id", "unicode_\u03b1"]


def raw_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_merge_retains_text_cells_and_numeric_missingness(tmp_path, monkeypatch):
    source = tmp_path / "run"
    designs = source / "intermediate_designs"
    sequences = ["NA", "AANA", "GNA", "NA:G", "G:G", "AGNA"]
    rows = []
    for index, (stem, sequence) in enumerate(zip(IDS, sequences, strict=True)):
        _pair(designs, stem, stem)
        rows.append(
            {
                "id": stem,
                "file_name": f"{stem}.cif",
                "target_id": IDS[-index - 1],
                "designed_sequence": sequence,
                "designed_chain_sequence": "NA",
                "designed_sequence_0": "NA",
                "full_sequence_0": "AANA",
                "full_sequence_chainB": "NA",
                "optional_note": "quoted, Unicode \u03b1\r\nnext line",
                "esmfold2_input_hash": "",
                "bb_rmsd": "" if index % 2 else "1.25",
            }
        )
    write_rows(designs / "aggregate_metrics_analyze.csv", rows)
    pd.DataFrame({"id": IDS, "sequence": sequences}).to_pickle(
        designs / "ca_coords_sequences.pkl.gz"
    )
    output = tmp_path / "merged"
    _merge(monkeypatch, [source], output)
    merged = output / designs.name
    result = raw_rows(merged / "aggregate_metrics_analyze.csv")
    assert [row["id"] for row in result] == [f"run_{stem}" for stem in IDS]
    for original, row in zip(rows, result, strict=True):
        for key in original.keys() - {"id", "file_name"}:
            assert row[key] == original[key]
        assert (merged / row["file_name"]).read_text() == original["id"]
    numerical = pd.read_csv(merged / "aggregate_metrics_analyze.csv")
    assert pd.api.types.is_numeric_dtype(numerical["bb_rmsd"])
    assert numerical["bb_rmsd"].isna().tolist() == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert numerical["esmfold2_input_hash"].isna().all()


def test_merge_retains_per_target_identifier_strings(tmp_path, monkeypatch):
    source = tmp_path / "run"
    designs = source / "intermediate_designs"
    _pair(designs, "candidate", "coordinate fixture")
    write_rows(
        designs / "aggregate_metrics_analyze.csv",
        [{"id": "candidate", "file_name": "candidate.cif"}],
    )
    rows = [
        {"target_id": stem, "mean_metric": "" if index % 2 else "1.25"}
        for index, stem in enumerate(IDS)
    ]
    write_rows(designs / "per_target_metrics_analyze.csv", rows)
    output = tmp_path / "merged"
    _merge(monkeypatch, [source], output)
    path = output / designs.name / "per_target_metrics_analyze.csv"
    assert raw_rows(path) == rows
    frame = pd.read_csv(path)
    assert pd.api.types.is_numeric_dtype(frame["mean_metric"])


def test_filter_retains_ids_sequence_families_and_missing_sequence(tmp_path):
    sequences = ["NA", "AANA", "GNA", "NA:G", "", "AGNA"]
    task = _load_filter(
        tmp_path,
        pd.DataFrame(
            {
                "id": IDS,
                "file_name": [f"{stem}.cif" for stem in IDS],
                "target_id": list(reversed(IDS)),
                "designed_sequence": sequences,
                "designed_chain_sequence": ["NA"] * len(IDS),
                "designed_sequence_0": ["NA"] * len(IDS),
                "full_sequence_0": ["NA"] * len(IDS),
                "bb_rmsd": [1.25, np.nan, 0, np.nan, 1.5, 2.0],
                "esmfold2_input_hash": [np.nan] * len(IDS),
            }
        ),
        [],
    )
    assert task.df_in["id"].tolist() == IDS
    assert task.df_in["target_id"].tolist() == list(reversed(IDS))
    assert task.df_in["file_name"].tolist() == [f"{stem}.cif" for stem in IDS]
    for key in ("designed_chain_sequence", "designed_sequence_0", "full_sequence_0"):
        assert task.df_in[key].tolist() == ["NA"] * len(IDS)
    assert task.df["has_x"].tolist() == [False, False, False, False, True, False]
    assert pd.api.types.is_numeric_dtype(task.df_in["bb_rmsd"])
    assert task.df_in["bb_rmsd"].isna().tolist() == [
        False,
        True,
        False,
        True,
        False,
        False,
    ]
    assert task.df_in["esmfold2_input_hash"].isna().all()


def test_novelty_retains_foldseek_query_names(tmp_path, monkeypatch):
    def write_foldseek(command, *, check):
        assert check
        path = Path(command[4])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerows([stem, "001", 0.1, 0.25, 0.75] for stem in IDS)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "boltzgen.task.analyze.analyze_utils.subprocess.run", write_foldseek
    )
    result = compute_novelty_foldseek(
        tmp_path,
        tmp_path,
        tmp_path / "reference",
        [f"{stem}.pdb" for stem in IDS],
        "foldseek",
    )
    assert result["query"].tolist() == IDS
    assert result["novelty"].tolist() == [0.5] * len(IDS)


def test_clustering_retains_foldseek_file_and_representative_names(
    tmp_path, monkeypatch
):
    def write_foldseek(command, *, check):
        assert check
        path = Path(command[3] + "_cluster.tsv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, delimiter="\t").writerows([stem, stem] for stem in IDS)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("boltzgen.task.analyze.analyze.subprocess.run", write_foldseek)
    monkeypatch.setattr(
        "boltzgen.task.analyze.analyze.torch.set_num_interop_threads", lambda _: None
    )
    task = Analyze(
        "csv",
        SimpleNamespace(),
        design_dir=str(tmp_path),
        allatom_fold_metrics=False,
        compute_lddts=False,
    )
    result = task.run_foldseek_clustering(
        pd.DataFrame({"id": IDS, "num_design": [3] * len(IDS)})
    )
    assert result["id"].tolist() == IDS
    assert result["clusters_05_tmscore"].tolist() == IDS


@pytest.mark.parametrize(
    "value",
    [
        "NA",
        "N/A",
        "null",
        "NULL",
        "nan",
        "NaN",
        "None",
        "001",
        "9007199254740993",
        "quoted,id",
        "line\r\nbreak",
        "\u03b1",
        "",
    ],
)
def test_text_null_tokens_and_numeric_missing_tokens_are_independent(tmp_path, value):
    from boltzgen.data.parse.table import read_metrics_table

    path = tmp_path / "metrics.csv"
    rows = [
        {
            "id": value,
            "file_name": value,
            "target_id": value,
            "designed_sequence": value,
            "full_sequence_0": value,
            "score": metric,
            "count": "0007",
            "flag": "True",
        }
        for metric in ("", "NA", "null", "NaN", "1.25", "inf", "-inf")
    ]
    write_rows(path, rows)
    frame = read_metrics_table(path)
    for key in ("id", "file_name", "target_id"):
        assert frame[key].tolist() == [value] * len(rows)
    for key in ("designed_sequence", "full_sequence_0"):
        if value:
            assert frame[key].tolist() == [value] * len(rows)
        else:
            assert frame[key].isna().all()
    assert frame["score"].iloc[:4].isna().all()
    assert frame["score"].iloc[4:].tolist() == [1.25, np.inf, -np.inf]
    assert frame["count"].tolist() == [7] * len(rows)
    assert frame["flag"].tolist() == [True] * len(rows)


@pytest.mark.parametrize("header", ["id,designed_sequence,score", "score,count"])
def test_header_only_metric_tables_remain_empty(tmp_path, header):
    from boltzgen.data.parse.table import read_metrics_table

    path = tmp_path / "empty.csv"
    path.write_text(header + "\n")
    frame = read_metrics_table(path)
    assert frame.empty
    assert frame.columns.tolist() == header.split(",")


def test_numeric_only_table_uses_normal_inference(tmp_path):
    from boltzgen.data.parse.table import read_metrics_table

    path = tmp_path / "numeric.csv"
    path.write_text("score,count\n1.25,7\nNA,8\n")
    pd.testing.assert_frame_equal(read_metrics_table(path), pd.read_csv(path))


def test_utf8_bom_keeps_identifier_column(tmp_path):
    from boltzgen.data.parse.table import read_metrics_table

    path = tmp_path / "bom.csv"
    path.write_text("id,designed_sequence,score\n001,NA,NA\n", encoding="utf-8-sig")
    frame = read_metrics_table(path)
    assert frame["id"].tolist() == ["001"]
    assert frame["designed_sequence"].tolist() == ["NA"]
    assert frame["score"].isna().all()
