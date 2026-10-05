"""Regression coverage for independent hard filters and confidence metric keys."""

# ruff: noqa: INP001

import json
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from boltzgen.data import const
from boltzgen.task.analyze.analyze_utils import get_best_folding_sample
from boltzgen.task.esmfold2.contract import (
    ESM_VERSION,
    ESMC_REVISION,
    MODEL_REVISION,
    SCORE_DIR,
    SCORE_KEY,
    file_sha256,
    fingerprint,
)
from boltzgen.task.filter.filter import Filter
from boltzgen.task.predict.writer import AffinityWriter


def _load_filter(
    tmp_path: Path,
    frame: pd.DataFrame,
    rules: list[dict],
    *,
    filter_target_aligned: bool = False,
    use_affinity: bool = True,
) -> Filter:
    frame = frame.copy()
    defaults = {
        "id": [f"design_{i}" for i in range(len(frame))],
        "designed_sequence": ["A" * i + "G" for i in range(len(frame))],
        "bb_rmsd": 2.5,
        "bb_rmsd_design": 2.5,
        "min_interaction_pae": 1.0,
        "min_design_to_target_pae": 1.0,
        "design_iiptm": 1.0,
        "design_ptm": 1.0,
        "design_to_target_iptm": 1.0,
        "quality": list(range(len(frame))),
    }
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
    if not use_affinity:
        # Exercise the production provenance check with a complete score record.
        score_dir = tmp_path / SCORE_DIR
        score_dir.mkdir(exist_ok=True)
        frame[SCORE_KEY] = 0.75
        frame["file_name"] = frame["id"] + ".cif"
        frame["esmfold2_input_hash"] = ""
        for index, row in frame.iterrows():
            design = tmp_path / row["file_name"]
            design.write_text(f"fixture design {row['id']}")
            request = {"design_id": row["id"], "design_sha256": file_sha256(design)}
            input_hash = fingerprint(request)
            frame.loc[index, "esmfold2_input_hash"] = input_hash
            metrics = {
                SCORE_KEY: 0.75,
                "esmfold2_design_to_target_ipsae": 0.75,
                "esmfold2_target_to_design_ipsae": 0.75,
            }
            result = {
                "schema_version": 1,
                "model_revision": MODEL_REVISION,
                "esmc_revision": ESMC_REVISION,
                "esm_version": ESM_VERSION,
                "input_hash": input_hash,
                "metrics": metrics,
            }
            (score_dir / f"{row['id']}.input.json").write_text(json.dumps(request))
            (score_dir / f"{row['id']}.json").write_text(json.dumps(result))
            (score_dir / f"{row['id']}.cif").write_text("fixture prediction")
            np.savez(score_dir / f"{row['id']}.npz", pae=np.zeros((1, 2, 2)))
    frame.to_csv(tmp_path / "aggregate_metrics_test.csv", index=False)
    task = Filter(
        design_dir=str(tmp_path),
        use_affinity=use_affinity,
        filter_designfolding=False,
        filter_cysteine=False,
        filter_biased=False,
        filter_target_aligned=filter_target_aligned,
        additional_filters=rules,
        metrics_override={
            "design_to_target_iptm": None,
            "design_ptm": None,
            "neg_min_design_to_target_pae": None,
            "plip_hbonds_refolded": None,
            "plip_saltbridge_refolded": None,
            "delta_sasa_refolded": None,
            "affinity_probability_binary1": None,
            SCORE_KEY: None,
            "quality": 1,
        },
    )
    task.load_dataframe()
    return task


@pytest.mark.parametrize("order", list(permutations(range(3))))
@pytest.mark.parametrize("use_affinity", [False, True])
def test_independent_rules_count_and_rank_csv(
    tmp_path: Path, order: tuple[int, ...], use_affinity: bool
) -> None:
    truth = np.array(list(product([False, True], repeat=3)))
    frame = pd.DataFrame(
        {
            "high": np.where(truth[:, 0], 0.5, 0.4),
            "low": np.where(truth[:, 1], 0.5, 0.6),
            "signed": np.where(truth[:, 2], 0.0, -1.0),
        }
    )
    frame.loc[len(frame)] = [np.nan, 0.5, 0.0]
    rules = [
        {"feature": "high", "lower_is_better": False, "threshold": 0.5},
        {"feature": "low", "lower_is_better": True, "threshold": 0.5},
        {"feature": "signed", "lower_is_better": False, "threshold": 0.0},
    ]
    task = _load_filter(
        tmp_path, frame, [rules[i] for i in order], use_affinity=use_affinity
    )
    task.filter_df()

    expected_counts = [int(row.sum()) + 3 for row in truth] + [5]
    assert task.df["num_filters_passed"].tolist() == expected_counts
    assert task.df["num_filters_passed"].dtype == np.dtype("int64")
    assert task.df["pass_filters"].tolist() == [*truth.all(axis=1).tolist(), False]
    for i, feature in enumerate(["high", "low", "signed"]):
        expected = [*truth[:, i].tolist(), i != 0]
        assert task.df[f"pass_{feature}_filter"].tolist() == expected

    task.sort_df()
    expected_order = sorted(
        range(len(frame)), key=lambda i: (expected_counts[i], i), reverse=True
    )
    assert task.df["id"].tolist() == [f"design_{i}" for i in expected_order]
    assert task.df["final_rank"].tolist() == list(range(1, len(frame) + 1))
    assert task.df.columns.is_unique


@pytest.mark.parametrize(
    ("rules", "passed_rule_counts"),
    [
        ([(False, 0.2), (False, 0.1)], [0, 1, 2, 2, 2, 0]),
        ([(False, 0.1), (True, 0.3)], [1, 2, 2, 2, 1, 0]),
        ([(False, 0.3), (True, 0.1)], [1, 1, 0, 1, 1, 0]),
        ([(False, 0.1), (False, 0.1)], [0, 2, 2, 2, 2, 0]),
    ],
    ids=["nested", "range", "contradictory", "identical"],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_repeated_feature_counts_flags_ranking_and_penalties(
    tmp_path: Path,
    rules: list[tuple[bool, float]],
    passed_rule_counts: list[int],
    reverse: bool,
) -> None:
    configured = [
        {"feature": "x", "lower_is_better": low, "threshold": threshold}
        for low, threshold in rules
    ]
    if reverse:
        configured.reverse()
    frame = pd.DataFrame({"x": [0.0, 0.1, 0.2, 0.3, 0.4, np.nan]})
    task = _load_filter(tmp_path, frame, configured)

    # Obtain the unpenalized score without changing the real scoring calculation.
    task.filters = task.filters[: -len(configured)]
    task.filter_df()
    task.absolute_metrics()
    unpenalized = task.df["absolute_score"].copy()
    assert (unpenalized > 0).all()
    task.filters.extend(configured)
    task.filter_df()

    expected_counts = [3 + count for count in passed_rule_counts]
    expected_pass = [count == len(configured) for count in passed_rule_counts]
    assert task.df["num_filters_passed"].tolist() == expected_counts
    assert task.df["pass_filters"].tolist() == expected_pass
    assert task.df["pass_x_filter"].tolist() == expected_pass

    task.absolute_metrics()
    expected_penalties = [0.1 ** (2 - count) for count in passed_rule_counts]
    np.testing.assert_allclose(
        task.df["absolute_score"] / unpenalized, expected_penalties
    )
    task.sort_df()
    expected_order = sorted(
        range(len(frame)), key=lambda i: (expected_counts[i], i), reverse=True
    )
    assert task.df["id"].tolist() == [f"design_{i}" for i in expected_order]

    output = tmp_path / "ranked_metrics.csv"
    task.df.to_csv(output, index=False)
    persisted = pd.read_csv(output).set_index("id").sort_index()
    assert persisted.columns.is_unique
    assert persisted["pass_x_filter"].tolist() == expected_pass
    assert persisted["pass_filters"].tolist() == expected_pass
    assert persisted["num_filters_passed"].tolist() == expected_counts

    # Re-running after sorting must align by row index and reset previous flags.
    task.filters.reverse()
    task.filter_df()
    actual = task.df.set_index("id").sort_index()
    assert actual["num_filters_passed"].tolist() == expected_counts
    assert actual["pass_x_filter"].tolist() == expected_pass
    assert actual["pass_filters"].tolist() == expected_pass


@pytest.mark.parametrize("negative", [False, True])
def test_fraction_penalties_keep_length_exemption_per_rule(
    tmp_path: Path, negative: bool
) -> None:
    task = _load_filter(
        tmp_path,
        pd.DataFrame(
            {
                "ALA_fraction": [0.0, 0.0, 0.2, np.nan],
                "num_design": [8, 9, 9, 9],
                "design_iiptm": 0.1 if negative else 1.0,
                "design_ptm": 0.1 if negative else 1.0,
                "min_design_to_target_pae": 20.0 if negative else 1.0,
            }
        ),
        [
            {"feature": "ALA_fraction", "lower_is_better": False, "threshold": 0.1},
            {"feature": "ALA_fraction", "lower_is_better": True, "threshold": 0.3},
        ],
    )
    task.filter_df()
    task.absolute_metrics()
    assert task.df["num_filters_passed"].tolist() == [4, 4, 5, 3]
    assert task.df["pass_ALA_fraction_filter"].tolist() == [False, False, True, False]
    scores = task.df["absolute_score"]
    assert (scores.iloc[2] < 0) == negative
    penalty = 10.0 if negative else 0.1
    np.testing.assert_allclose(scores / scores.iloc[2], [1.0, penalty, 1.0, penalty**2])


@pytest.mark.parametrize("reverse", [False, True])
def test_failed_rules_worsen_negative_absolute_scores(
    tmp_path: Path, reverse: bool
) -> None:
    rules = [
        {"feature": "x", "lower_is_better": False, "threshold": 0.1},
        {"feature": "x", "lower_is_better": False, "threshold": 0.4},
    ]
    if reverse:
        rules.reverse()
    task = _load_filter(
        tmp_path,
        pd.DataFrame(
            {
                "x": [0.5, 0.25, 0.0] * 2,
                "design_iiptm": [0.1] * 3 + [1.0] * 3,
                "design_ptm": [0.1] * 3 + [1.0] * 3,
                "min_design_to_target_pae": [20.0] * 3 + [1.0] * 3,
            }
        ),
        rules,
    )
    task.filter_df()
    task.absolute_metrics()
    scores = task.df["absolute_score"]
    assert scores.iloc[0] < 0 < scores.iloc[3]
    assert scores.iloc[2] < scores.iloc[1] < scores.iloc[0]
    assert scores.iloc[5] < scores.iloc[4] < scores.iloc[3]
    np.testing.assert_allclose(scores.iloc[:3] / scores.iloc[0], [1.0, 10.0, 100.0])
    np.testing.assert_allclose(scores.iloc[3:] / scores.iloc[3], [1.0, 0.1, 0.01])
    assert task.df["num_filters_passed"].tolist() == [5, 4, 3] * 2


def test_affinity_scores_available_outside_source_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _load_filter(
        tmp_path,
        pd.DataFrame({"x": [1.0, 0.0]}),
        [{"feature": "x", "lower_is_better": False, "threshold": 0.5}],
    )
    task.filter_df()
    monkeypatch.chdir(tmp_path)
    task.absolute_metrics()
    assert "absolute_score" in task.df
    assert "structure_confidence" in task.df
    assert task.df["absolute_score"].iloc[0] > 0
    np.testing.assert_allclose(
        task.df["absolute_score"] / task.df["absolute_score"].iloc[0], [1.0, 0.1]
    )


def test_builtin_rules_count_successes_after_failures(tmp_path: Path) -> None:
    task = _load_filter(
        tmp_path,
        pd.DataFrame(
            {
                "designed_sequence": ["AAAA", "AAXA"],
                "bb_rmsd": [2.5, np.nan],
                "bb_rmsd_design": [2.5, 2.5],
            }
        ),
        [],
    )
    task.filter_df()
    assert task.df["num_filters_passed"].tolist() == [3, 1]
    assert task.df["pass_filters"].tolist() == [True, False]


def test_additional_rule_for_builtin_feature(tmp_path: Path) -> None:
    task = _load_filter(
        tmp_path,
        pd.DataFrame({"bb_rmsd": [0.5, 1.0, 2.0, 3.0, np.nan]}),
        [{"feature": "filter_rmsd", "lower_is_better": True, "threshold": 1.0}],
    )
    for _ in range(2):
        task.filter_df()
        assert task.df["num_filters_passed"].tolist() == [4, 4, 3, 2, 2]
        expected_pass = [True, True, False, False, False]
        assert task.df["pass_filter_rmsd_filter"].tolist() == expected_pass
        assert task.df["pass_filters"].tolist() == expected_pass
        task.filters.reverse()


@pytest.mark.parametrize("include_missing", [False, True])
def test_target_aligned_boolean_rule_after_csv_roundtrip(
    tmp_path: Path, include_missing: bool
) -> None:
    values = [True, False, np.nan] if include_missing else [True, False]
    task = _load_filter(
        tmp_path,
        pd.DataFrame({"bb_target_aligned<2.5": values}),
        [],
        filter_target_aligned=True,
    )
    task.filter_df()
    expected = [True, False, False] if include_missing else [True, False]
    assert task.df["pass_bb_target_aligned<2.5_filter"].tolist() == expected
    assert task.df["pass_filters"].tolist() == expected
    assert task.df["num_filters_passed"].tolist() == [3 + int(p) for p in expected]


@pytest.mark.parametrize("empty_data", [False, True])
@pytest.mark.parametrize("no_rules", [False, True])
def test_empty_data_and_empty_rule_set(
    tmp_path: Path, empty_data: bool, no_rules: bool
) -> None:
    frame = pd.DataFrame({"quality": [] if empty_data else [0.0, 1.0]})
    task = _load_filter(tmp_path, frame, [])
    if no_rules:
        task.filters = []
    task.filter_df()
    assert task.df["num_filters_passed"].tolist() == [0 if no_rules else 3] * len(frame)
    assert task.df["pass_filters"].tolist() == [True] * len(frame)
    task.sort_df()
    assert len(task.df) == len(frame)
    assert task.df.columns.is_unique


def test_confidence_metric_keys_are_unique_and_persisted(tmp_path: Path) -> None:
    keys = const.eval_keys_confidence
    assert len(keys) == len(set(keys))
    assert keys.count("ligand_iptm") == 1
    assert len(const.eval_keys) == len(set(const.eval_keys))

    prediction = {key: torch.tensor([0.25, 0.75]) for key in keys}
    prediction["coords"] = torch.zeros((2, 1, 3))
    prediction["exception"] = False
    # This writer persists eval_keys without requiring an unrelated structure export.
    writer = AffinityWriter(str(tmp_path))
    writer.write_on_batch_end(prediction=prediction, batch={"id": ["test"]})
    with np.load(writer.outdir / "test.npz") as persisted:
        assert len(persisted.files) == len(set(persisted.files))
        assert set(persisted.files) == set(keys) | {"coords"}
        best = get_best_folding_sample(persisted)
    assert set(best) == set(keys) | {"coords"}
    assert best["ligand_iptm"] == pytest.approx(0.75)
