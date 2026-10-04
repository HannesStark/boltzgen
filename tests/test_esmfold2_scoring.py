"""Protocol, score selection, and provenance regressions without model weights."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boltzgen.task.esmfold2.contract import (
    ESMC_REVISION,
    ESM_VERSION,
    MODEL_REVISION,
    SCORE_KEY,
    fingerprint,
    load_result,
)
from boltzgen.task.esmfold2.ipsae import score_interface


@pytest.mark.parametrize("cached", [False, True])
def test_runtime_uses_cache_offline_and_provisions_only_when_needed(
    monkeypatch, cached
):
    import subprocess
    from boltzgen.task.esmfold2 import runtime

    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        if "--offline" in command and not cached:
            return subprocess.CompletedProcess(command, 2, "", "No cached environment")
        return subprocess.CompletedProcess(command, 0, "/isolated/python\n")

    monkeypatch.setattr(runtime.subprocess, "run", probe)
    assert runtime.resolve_python(require_cuda=True) == "/isolated/python"
    assert "--offline" in calls[0]
    assert len(calls) == (2 if cached else 3)
    assert calls[-1][0] == "/isolated/python"
    assert "torch.cuda.is_available()" in calls[-1][-1]
    if not cached:
        assert "--offline" not in calls[1]


def test_cached_runtime_validation_failure_does_not_trigger_online_install(monkeypatch):
    import subprocess
    from boltzgen.task.esmfold2 import runtime

    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        if "--offline" in command:
            return subprocess.CompletedProcess(command, 0, "/isolated/python\n")
        assert command[0] == "/isolated/python"
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runtime.subprocess, "run", probe)
    with pytest.raises(RuntimeError, match="Could not prepare"):
        runtime.resolve_python(require_cuda=True)
    assert len(calls) == 2


def test_runtime_override_failure_does_not_silently_install_another_runtime(
    monkeypatch,
):
    import subprocess
    from boltzgen.task.esmfold2 import runtime

    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runtime.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Could not prepare"):
        runtime.resolve_python("/managed/python")
    assert len(calls) == 1
    assert calls[0][0] == "/managed/python"


def test_ipsae_directionality_cutoff_and_nucleic_acid_normalization():
    pae = np.full((5, 5), 20.0)
    pae[:2, 2:] = [[1, 1, 10], [20, 20, 20]]
    pae[2:, :2] = [[2, 2], [20, 20], [20, 20]]
    result = score_interface(pae, [0, 1], [2, 3, 4])
    assert result["esmfold2_design_to_target_ipsae"] == pytest.approx(0.5)
    assert result["esmfold2_target_to_design_ipsae"] == pytest.approx(0.2)
    assert result[SCORE_KEY] == pytest.approx(0.2)
    assert score_interface(pae, [0, 1], [2, 3, 4], nucleic_acid=True)[
        SCORE_KEY
    ] == pytest.approx(0.5)
    assert score_interface(np.full((2, 2), 10.0), [0], [1])[SCORE_KEY] == 0
    pae[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_interface(pae, [0], [1])


@pytest.mark.parametrize(
    "protocol",
    [
        "protein-anything",
        "peptide-anything",
        "nanobody-anything",
        "antibody-anything",
        "protein-redesign",
        "protein-small_molecule",
    ],
)
@pytest.mark.parametrize("python_override", [None, "/esm/bin/python"])
def test_protocol_routing_keeps_boltz_structural_checks(
    monkeypatch, tmp_path, protocol, python_override
):
    from boltzgen.cli import boltzgen as cli

    # Hardware and checkpoint download are external boundaries; test the actual
    # parser, pipeline construction, and merged per-step Hydra configs.
    monkeypatch.setattr(cli.torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.delenv("BOLTZGEN_ESMFOLD2_PYTHON", raising=False)
    monkeypatch.setattr(
        cli,
        "get_artifact_path",
        lambda args, artifact, **kwargs: Path("/weights") / artifact.rsplit(":", 1)[-1],
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
        + (["--esmfold2_python", python_override] if python_override else [])
    )
    steps = {
        s.name: s.get_config()
        for s in cli.BinderDesignPipeline(args, Path("/mols")).steps
    }
    assert "folding" in steps
    assert steps["folding"].checkpoint.endswith("boltz2_conf_final.ckpt")
    small = protocol == "protein-small_molecule"
    assert ("affinity" in steps) == small
    assert ("esmfold2_scoring" in steps) != small
    assert steps["analysis"].esmfold2_metrics == (not small)
    assert steps["filtering"].use_affinity == small
    if small:
        assert steps["affinity"].checkpoint.endswith("boltz2_aff.ckpt")
    else:
        assert steps["esmfold2_scoring"].python == python_override
        assert steps["esmfold2_scoring"].diffusion_samples == 5
        assert not steps["analysis"].data.skip_existing
        redesign = protocol == "protein-redesign"
        assert steps["esmfold2_scoring"].scoring_mode == (
            "redesign" if redesign else "binder"
        )
        if redesign:
            import hydra

            task = hydra.utils.instantiate(steps["filtering"])
            assert task.esmfold2_score_key == "esmfold2_score"
            assert task.metrics == {"esmfold2_score": 1, "neg_filter_rmsd_design": 4}


def test_ranking_and_tiebreak_follow_esmfold2(tmp_path):
    from boltzgen.task.filter.filter import Filter

    task = Filter(
        design_dir=tmp_path, budget=1, top_budget=1, from_inverse_folded=False
    )
    assert SCORE_KEY in task.metrics
    assert "design_to_target_iptm" not in task.metrics
    assert "neg_min_design_to_target_pae" not in task.metrics
    assert "design_ptm" in task.metrics
    task.df = pd.DataFrame(
        {
            "id": ["boltz_favorite", "esm_favorite"],
            "num_filters_passed": [2, 2],
            "design_to_target_iptm": [0.99, 0.01],
            SCORE_KEY: [0.1, 0.9],
            "design_ptm": [0.9, 0.1],
            "plip_hbonds": [1, 1],
            "plip_saltbridge": [1, 1],
            "delta_sasa_original": [1, 1],
        }
    )
    task.sort_df()
    assert task.df.iloc[0]["id"] == "esm_favorite"
    affinity = Filter(design_dir=tmp_path / "ligand", use_affinity=True)
    assert SCORE_KEY not in affinity.metrics
    assert affinity.metrics["affinity_probability_binary1"] == 1
    assert affinity.metrics["design_to_target_iptm"] == 1.1


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize(
    "scoring_mode,target_chains,error",
    [
        ("binder", None, None),
        ("redesign", ["A"], "uses every polymer chain"),
        ("bogus", None, "scoring_mode must be"),
        ("binder", [], "nonempty, unique"),
        ("binder", ["A", "A"], "nonempty, unique"),
    ],
)
def test_configure_validates_settings_before_runtime_setup(
    monkeypatch, tmp_path, reuse, scoring_mode, target_chains, error
):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from boltzgen.cli import boltzgen as cli
    from boltzgen.task.esmfold2 import runtime
    from boltzgen.task.esmfold2.score import ESMFold2Score

    config = OmegaConf.create(
        dict(
            python=None,
            reuse=reuse,
            scoring_mode=scoring_mode,
            scoring_target_chains=target_chains,
        )
    )
    step = SimpleNamespace(
        name="esmfold2_scoring", check=lambda: None, get_config=lambda: config
    )
    pipeline = SimpleNamespace(steps=[step], pretty_print=lambda: None)
    monkeypatch.setattr(cli, "BinderDesignPipeline", lambda *args: pipeline)
    monkeypatch.setattr(cli, "get_artifact_path", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(cli, "load_canonicals", lambda **kwargs: {})
    monkeypatch.setattr(cli, "check_design_specs", lambda *args: None)
    calls = []
    monkeypatch.setattr(
        runtime,
        "resolve_python",
        lambda python, **kwargs: calls.append((python, kwargs)),
    )
    args = cli.build_parser().parse_args(
        ["configure", "input.yaml", "--output", str(tmp_path / "result")]
        + (["--reuse"] if reuse else [])
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            ESMFold2Score(
                data=None,
                design_dir=tmp_path,
                scoring_mode=scoring_mode,
                scoring_target_chains=target_chains,
            )
        with pytest.raises(ValueError, match=error):
            cli.configure_command(args)
        assert calls == []
        assert not (args.output / "config/esmfold2_scoring.yaml").exists()
        return
    cli.configure_command(args)
    assert calls == ([] if reuse else [(None, {"require_cuda": True})])
    assert (args.output / "config/esmfold2_scoring.yaml").is_file()


def test_missing_scores_and_changed_protocol_cannot_reuse_old_results(tmp_path):
    from boltzgen.task.filter.filter import Filter

    pd.DataFrame({"design_to_target_iptm": [0.9]}).to_csv(
        tmp_path / "aggregate_metrics_old.csv", index=False
    )
    with pytest.raises(ValueError, match="Missing ESMFold2"):
        Filter(design_dir=tmp_path).load_dataframe()
    result = {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "esmc_revision": ESMC_REVISION,
        "esm_version": ESM_VERSION,
        "input_hash": "old",
        "metrics": score_interface(np.zeros((2, 2)), [0], [1]),
    }
    path = tmp_path / "score.json"
    path.with_suffix(".cif").touch()
    path.with_suffix(".npz").touch()
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Stale"):
        load_result(path, "new")
    result["metrics"][SCORE_KEY] = float("nan")
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="invalid"):
        load_result(path)
    path.with_suffix(".cif").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        load_result(path)
    assert fingerprint({"samples": 1}) != fingerprint({"samples": 5})


def test_missing_full_source_is_an_error():
    from boltzgen.task.esmfold2.score import validate_context

    with pytest.raises(ValueError, match="Missing full-source"):
        validate_context(None)
    with pytest.raises(ValueError, match="Full sequence is unavailable"):
        validate_context(
            {
                "version": 1,
                "chains": [
                    {
                        "mol_type": 0,
                        "source_chain": "A",
                        "source": "cropped.pdb",
                        "complete": False,
                    }
                ],
            }
        )


def test_analysis_csv_preserves_scores_for_filtering(tmp_path, monkeypatch):
    from boltzgen.data import const
    from boltzgen.task.analyze.analyze import Analyze
    from boltzgen.task.filter.filter import Filter
    from boltzgen.task.esmfold2.contract import SCORE_DIR, file_sha256

    design = tmp_path / "target_0.cif"
    design.write_text("generated design")
    request = {"design_id": "target_0", "design_sha256": file_sha256(design)}
    metrics = score_interface(np.full((2, 2), 2.7), [0], [1])
    scores = tmp_path / SCORE_DIR
    scores.mkdir()
    (scores / "target_0.input.json").write_text(json.dumps(request))
    (scores / "target_0.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_revision": MODEL_REVISION,
                "esmc_revision": ESMC_REVISION,
                "esm_version": ESM_VERSION,
                "input_hash": fingerprint(request),
                "metrics": metrics,
                "selected_sample": 0,
            }
        )
    )
    (scores / "target_0.cif").write_text("ESMFold2 structure")
    np.savez(scores / "target_0.npz", pae=np.full((1, 2, 2), 2.7))
    analyze = Analyze(
        name="analyze", data=None, design_dir=str(tmp_path), esmfold2_metrics=True
    )
    row = dict(
        id="target_0",
        file_name=design.name,
        designed_sequence="AAA",
        esmfold2_input_hash=fingerprint(request),
        rmsd=0.1,
        rmsd_design=0.1,
        min_interaction_pae=2.7,
        **metrics,
    )
    np.savez(analyze.metrics_dir / "metrics_target_0.npz", **row)
    np.savez(
        analyze.metrics_dir / "data_target_0.npz",
        sample_id="target_0",
        target_id="target",
        design_seq=np.array([const.token_ids["ALA"]] * 3),
        ca_coords=np.ones((3, 3)),
    )
    monkeypatch.setattr(analyze, "compute_diversity", lambda *args: ({}, {}))
    monkeypatch.setattr(analyze, "compute_novelty", lambda: ({}, {}))
    monkeypatch.setattr(analyze, "make_histograms", lambda *args: ({}, {}))
    analyze.aggregate_metrics()
    task = Filter(design_dir=tmp_path, from_inverse_folded=False)
    task.load_dataframe()
    assert task.df.iloc[0][SCORE_KEY] == pytest.approx(metrics[SCORE_KEY], abs=1e-12)


def test_merge_preserves_scores_under_renamed_designs(tmp_path):
    from boltzgen.cli import boltzgen as cli
    from boltzgen.task.esmfold2.contract import SCORE_DIR, file_sha256

    source = tmp_path / "run"
    designs = source / "intermediate_designs_inverse_folded"
    scores = designs / SCORE_DIR
    scores.mkdir(parents=True)
    design = designs / "candidate.cif"
    design.write_text("generated design")
    request = {"design_id": "candidate", "design_sha256": file_sha256(design)}
    metrics = score_interface(np.zeros((2, 2)), [0], [1])
    result = {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "esmc_revision": ESMC_REVISION,
        "esm_version": ESM_VERSION,
        "input_hash": fingerprint(request),
        "metrics": metrics,
    }
    (scores / "candidate.input.json").write_text(json.dumps(request))
    (scores / "candidate.json").write_text(json.dumps(result))
    (scores / "candidate.cif").write_text("ESMFold2 structure")
    np.savez(scores / "candidate.npz", pae=np.zeros((1, 2, 2)))
    pd.DataFrame(
        [
            {
                "id": "candidate",
                "file_name": design.name,
                "esmfold2_input_hash": fingerprint(request),
                **metrics,
            }
        ]
    ).to_csv(designs / "aggregate_metrics_analyze.csv", index=False)
    output = tmp_path / "merged"
    args = cli.build_parser().parse_args(
        ["merge", str(source), "--output", str(output)]
    )
    cli.merge_command(args)
    merged = output / designs.name
    row = pd.read_csv(merged / "aggregate_metrics_analyze.csv").iloc[0]
    renamed = json.loads((merged / SCORE_DIR / f"{row['id']}.input.json").read_text())
    assert renamed["design_id"] == row["id"]
    assert row["esmfold2_input_hash"] == fingerprint(renamed)
    assert (
        load_result(merged / SCORE_DIR / f"{row['id']}.json", fingerprint(renamed))[
            "metrics"
        ]
        == metrics
    )


@pytest.mark.parametrize("metric", ["esmfold2_ipsae_min", "esmfold2_ptm"])
def test_redesign_provenance_ranking_and_monomer_metric(tmp_path, metric):
    from boltzgen.task.esmfold2.contract import (
        SCORE_DIR,
        REDESIGN_SCORE_KEY,
        file_sha256,
    )
    from boltzgen.task.filter.filter import Filter

    directory = tmp_path / SCORE_DIR
    directory.mkdir()
    rows = []
    for name, value in [("low", 0.1), ("high", 0.9)]:
        design = tmp_path / f"{name}.cif"
        design.write_text(name)
        request = {
            "design_id": name,
            "design_sha256": file_sha256(design),
            "scoring_mode": "redesign",
        }
        metrics = {REDESIGN_SCORE_KEY: value, metric: value}
        result = {
            "schema_version": 1,
            "model_revision": MODEL_REVISION,
            "esmc_revision": ESMC_REVISION,
            "esm_version": ESM_VERSION,
            "input_hash": fingerprint(request),
            "scoring_mode": "redesign",
            "score_metric": metric,
            "metrics": metrics,
        }
        (directory / f"{name}.input.json").write_text(json.dumps(request))
        (directory / f"{name}.json").write_text(json.dumps(result))
        (directory / f"{name}.cif").touch()
        np.savez(directory / f"{name}.npz", pae=np.ones((1, 3, 3)))
        rows.append(
            dict(
                id=name,
                file_name=design.name,
                designed_sequence="AAA" if name == "low" else "AAG",
                esmfold2_input_hash=fingerprint(request),
                esmfold2_score_metric=metric,
                rmsd=0.1,
                rmsd_design=0.1,
                min_interaction_pae=1,
                num_filters_passed=2,
                design_to_target_iptm=1 - value,
                **metrics,
            )
        )
    pd.DataFrame(rows).to_csv(tmp_path / "aggregate_metrics_analyze.csv", index=False)
    task = Filter(
        design_dir=tmp_path,
        from_inverse_folded=False,
        esmfold2_redesign=True,
        metrics_override={
            "design_ptm": None,
            "plip_hbonds": None,
            "plip_saltbridge": None,
            "delta_sasa_original": None,
        },
    )
    task.load_dataframe()
    task.absolute_metrics()
    assert "absolute_score" not in task.df
    task.sort_df()
    assert task.df.iloc[0]["id"] == "high"
    assert task.metrics == {REDESIGN_SCORE_KEY: 1}
    result_path = directory / "high.json"
    result = json.loads(result_path.read_text())
    csv_path = tmp_path / "aggregate_metrics_analyze.csv"
    other_metric = "esmfold2_ptm" if metric == SCORE_KEY else SCORE_KEY
    rows[1]["esmfold2_score_metric"] = other_metric
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="metric is stale"):
        task.load_dataframe()
    mixed_result = dict(result, score_metric=other_metric)
    mixed_result["metrics"] = {REDESIGN_SCORE_KEY: 0.9, other_metric: 0.9}
    result_path.write_text(json.dumps(mixed_result))
    with pytest.raises(ValueError, match="separate campaigns"):
        task.load_dataframe()
    rows[1]["esmfold2_score_metric"] = metric
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    result["metrics"][REDESIGN_SCORE_KEY] = 0.5
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Inconsistent"):
        load_result(result_path)


def test_redesign_ipsae_uses_nucleic_acid_floor_and_disconnected_chains():
    from boltzgen.task.esmfold2.ipsae import score_chain_vs_rest

    pae = np.full((3, 3), 2.0)
    reps = {"A": [0], "B": [1], "C": [2]}
    protein = score_chain_vs_rest(pae, reps, dict.fromkeys(reps, 0))
    mixed = score_chain_vs_rest(pae, reps, {"A": 0, "B": 0, "C": 2})
    assert all(value[SCORE_KEY] == pytest.approx(0.2) for value in protein.values())
    assert all(value[SCORE_KEY] == pytest.approx(0.5) for value in mixed.values())
    pae[2, :2] = pae[:2, 2] = 20
    scores = score_chain_vs_rest(pae, reps, dict.fromkeys(reps, 0))
    assert min(value[SCORE_KEY] for value in scores.values()) == 0


@pytest.mark.parametrize("failure", ["runtime", "worker"])
def test_failed_rescore_preserves_published_cache_until_valid_result(
    tmp_path, monkeypatch, failure
):
    from types import SimpleNamespace

    from boltzgen.task.esmfold2 import score as score_module
    from boltzgen.task.esmfold2.contract import (
        SCORE_DIR,
        file_sha256,
    )
    from boltzgen.task.esmfold2.score import ESMFold2Score

    class Dataset:
        def __init__(self, design):
            self.generated_paths = [design]
            self.metadata_paths = [tmp_path / "metadata.npz"]
            self.native_paths = [None]

        def getitem_from_paths(self, metadata, path, native):
            return object()

    design = tmp_path / "candidate.cif"
    design.write_text("candidate structure")
    score_dir = tmp_path / SCORE_DIR
    score_dir.mkdir()
    old_request = {
        "design_id": "candidate",
        "design_sha256": file_sha256(design),
        "options": {"sampling_steps": 200},
    }
    new_request = {
        "design_id": "candidate",
        "design_sha256": file_sha256(design),
        "options": {"sampling_steps": 201},
    }
    request_path = score_dir / "candidate.input.json"
    request_path.write_text(json.dumps(old_request, indent=2) + "\n")
    metrics = score_interface(np.zeros((2, 2)), [0], [1])
    cached_result = {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "esmc_revision": ESMC_REVISION,
        "esm_version": ESM_VERSION,
        "input_hash": fingerprint(old_request),
        "metrics": metrics,
    }
    result_path = score_dir / "candidate.json"
    result_path.write_text(json.dumps(cached_result))
    (score_dir / "candidate.cif").write_text("old structure")
    (score_dir / "candidate.npz").write_bytes(b"old PAE")
    score_task = ESMFold2Score(
        data=SimpleNamespace(predict_set=Dataset(design)),
        design_dir=str(tmp_path),
        reuse=True,
        sampling_steps=201,
    )
    monkeypatch.setattr(score_module, "make_request", lambda *args, **kwargs: new_request)

    def snapshot():
        return {path.name: path.read_bytes() for path in score_dir.iterdir() if path.is_file()}

    before = snapshot()
    if failure == "runtime":
        def fail_runtime(*args, **kwargs):
            raise RuntimeError("offline runtime unavailable")

        monkeypatch.setattr(score_module, "resolve_python", fail_runtime)
        with pytest.raises(RuntimeError, match="offline runtime unavailable"):
            score_task.run()
    else:
        class FailedWorker:
            def wait(self):
                return 1

            def poll(self):
                return 1

            def terminate(self):
                raise AssertionError("completed failed worker should not be terminated")

        def fail_worker(command, **kwargs):
            manifest = Path(command[-3])
            staged_request = Path(json.loads(manifest.read_text())[0])
            assert staged_request.parent.name.startswith(".esmfold2-stage-")
            assert staged_request != request_path
            (staged_request.parent / "candidate.cif").write_text("partial replacement")
            (staged_request.parent / "candidate.npz").write_bytes(b"partial PAE")
            return FailedWorker()

        monkeypatch.setattr(score_module, "resolve_python", lambda *args, **kwargs: "/fake/python")
        monkeypatch.setattr(score_module.subprocess, "Popen", fail_worker)
        with pytest.raises(RuntimeError, match=r"worker exit codes \[1\]"):
            score_task.run()

    assert snapshot() == before
    assert list(score_dir.glob(".esmfold2-stage-*")) == []
    assert load_result(result_path, fingerprint(old_request)) == cached_result

    # A successful worker publishes the staged files and new request atomically
    # with respect to the completion marker, replacing the stale cache.
    class SuccessfulWorker:
        def wait(self):
            return 0

        def poll(self):
            return 0

        def terminate(self):
            raise AssertionError("successful worker should not be terminated")

    def succeed_worker(command, **kwargs):
        manifest = Path(command[-3])
        staged_request = Path(json.loads(manifest.read_text())[0])
        staged_dir = staged_request.parent
        assert staged_dir.name.startswith(".esmfold2-stage-")
        result = {
            "schema_version": 1,
            "model_revision": MODEL_REVISION,
            "esmc_revision": ESMC_REVISION,
            "esm_version": ESM_VERSION,
            "input_hash": fingerprint(new_request),
            "metrics": metrics,
        }
        (staged_dir / "candidate.cif").write_text("new structure")
        (staged_dir / "candidate.npz").write_bytes(b"new PAE")
        (staged_dir / "candidate.json").write_text(json.dumps(result))
        return SuccessfulWorker()

    monkeypatch.setattr(score_module, "resolve_python", lambda *args, **kwargs: "/fake/python")
    monkeypatch.setattr(score_module.subprocess, "Popen", succeed_worker)
    score_task.run()
    assert json.loads(request_path.read_text()) == new_request
    assert load_result(result_path, fingerprint(new_request))["input_hash"] == fingerprint(new_request)
    assert (score_dir / "candidate.cif").read_text() == "new structure"
    assert (score_dir / "candidate.npz").read_bytes() == b"new PAE"
    assert list(score_dir.glob(".esmfold2-stage-*")) == []
