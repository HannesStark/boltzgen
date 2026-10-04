"""SolubleMPNN configuration and optional real-weight pipeline integration tests.

Set BOLTZGEN_TEST_MOLDIR and BOLTZGEN_TEST_SOLUBLEMPNN_CHECKPOINT to run the
CPU integration tests without network access during pytest.
"""

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import gemmi
import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from boltzgen.cli.boltzgen import (
    ARTIFACTS,
    SOLUBLEMPNN_SHA256,
    BinderDesignPipeline,
    build_parser,
    configure_command,
    get_artifact_path,
    protocol_configs,
    run_command,
)
from boltzgen.data import const
from boltzgen.model.modules.masker import BoltzMasker
from boltzgen.task.predict.predict_solublempnn import SolubleMPNN
from boltzgen.task.predict.writer import DesignWriter

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "src/boltzgen/resources/config"


@pytest.mark.parametrize("target_kind,target_sequence", [("protein", "AGCAAGCA"), ("rna", "ACGUACGU")])
def test_symmetry_metadata_follows_design_only_crop(tmp_path, target_kind, target_sequence):
    from types import SimpleNamespace

    from boltzgen.data.feature.featurizer import Featurizer
    from boltzgen.data.mol import load_canonicals
    from boltzgen.data.tokenize.tokenizer import Tokenizer
    from boltzgen.task.predict.data_from_generated import FromGeneratedDataset
    from boltzgen.task.predict.data_from_yaml import PredictionDataset, collate

    moldir = os.environ.get("BOLTZGEN_TEST_MOLDIR")
    if moldir is None:
        pytest.skip("Set BOLTZGEN_TEST_MOLDIR")
    specification = tmp_path / "input.yaml"
    specification.write_text(
        "entities:\n"
        "  - protein: {id: [A, B], sequence: 6, symmetric_group: 1}\n"
        f"  - {target_kind}: {{id: R, sequence: {target_sequence}}}\n"
    )
    canonicals = load_canonicals(Path(moldir))
    config = SimpleNamespace(
        yaml_path=str(specification), tokenizer=Tokenizer(),
        featurizer=Featurizer(), multiplicity=1,
    )
    batch = collate([PredictionDataset(config, canonicals.copy(), moldir, atom14=False)[0]])
    prediction = dict(batch)
    prediction["coords"] = torch.arange(batch["coords"][:, 0].numel(), dtype=torch.float32).reshape_as(batch["coords"][:, 0]) + 1
    prediction["exception"] = False
    output = tmp_path / "generated"
    writer = DesignWriter(str(output), res_atoms_only=False, atom14=False)
    writer.write_on_batch_end(prediction=prediction, batch=batch, sample_id="symmetry")
    assert writer.failed == 0
    source = output / "symmetry_0.cif"
    reader = FromGeneratedDataset(
        [source], [source.with_suffix(".npz")], [source], moldir, canonicals,
        Tokenizer(), Featurizer(), extra_mol_dir=output / const.molecules_dirname,
        return_designfolding=True, design=False,
    )
    features = reader[0]
    assert len(features["symmetric_group"]) == len(features["design_mask"]) == 12
    assert (features["symmetric_group"] == 1).all()
    assert (features["mol_type"] == const.chain_type_ids["PROTEIN"]).all()

    with np.load(source.with_suffix(".npz")) as archive:
        metadata = {key: archive[key] for key in archive.files if key != "symmetric_group"}
    np.savez_compressed(source.with_suffix(".npz"), **metadata)
    legacy = reader[0]
    assert len(legacy["symmetric_group"]) == 12
    assert (legacy["symmetric_group"] == 0).all()


@pytest.mark.parametrize("only", [False, True])
@pytest.mark.parametrize("backend", [None, "boltzif", "solublempnn"])
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
def test_pipeline_selects_model_without_fetching_other_weights(
    tmp_path, monkeypatch, only, backend, protocol
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    argv = [
        "configure",
        "input.yaml",
        "--output",
        str(tmp_path / "out"),
        "--protocol",
        protocol,
        "--inverse_fold_num_sequences",
        "3",
        "--num_workers",
        "0",
        "--reuse",
        "--design_checkpoints",
        str(checkpoint),
        "--folding_checkpoint",
        str(checkpoint),
        "--affinity_checkpoint",
        str(checkpoint),
    ]
    if backend is not None:
        argv.extend(["--inverse_fold_model", backend])
    selected_backend = backend or (
        "boltzif" if protocol == "protein-small_molecule" else "solublempnn"
    )
    if selected_backend == "solublempnn":
        argv.extend(["--solublempnn_checkpoint", str(checkpoint)])
    else:
        argv.extend(["--inverse_fold_checkpoint", str(checkpoint)])
    if only:
        argv.append("--only_inverse_fold")
    args = build_parser().parse_args(argv)
    assert args.inverse_fold_model == backend

    def unexpected_download(*args, **kwargs):
        pytest.fail("The unselected inverse-folding checkpoint must not be downloaded")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", unexpected_download)
    monkeypatch.setattr(torch.hub, "download_url_to_file", unexpected_download)
    if protocol == "protein-small_molecule" and backend == "solublempnn":
        # Reject the incompatible choice even on a host without a GPU, before
        # fetching any model weights or querying device capability.
        monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_download)
        with pytest.raises(ValueError, match="protein-small_molecule requires BoltzIF"):
            BinderDesignPipeline(args, tmp_path)
        with pytest.raises(ValueError, match="protein-small_molecule requires BoltzIF"):
            configure_command(args)
        return
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    pipeline = BinderDesignPipeline(args, tmp_path)
    config = next(
        step for step in pipeline.steps if step.name == "inverse_folding"
    ).get_config()
    assert config.checkpoint == str(checkpoint)
    assert config.data.cfg.multiplicity == 3
    assert config.writer.inverse_fold
    restrictions = (
        ["CYS"]
        if protocol in {"peptide-anything", "nanobody-anything", "antibody-anything"}
        else []
    )
    if selected_backend == "solublempnn":
        assert config._target_.endswith("PredictSolubleMPNN")
        assert config.inverse_fold_restriction == restrictions
        assert config.sampling_temperature == 0.1
        assert "override" not in config
    else:
        assert config._target_.endswith("predict.Predict")
        assert (
            config.override.inverse_fold_args.inverse_fold_restriction == restrictions
        )
    if protocol == "protein-small_molecule":
        affinity = next(
            step for step in pipeline.steps if step.name == "affinity"
        ).get_config()
        assert affinity.data.compute_affinity
        assert affinity.data.design_dir == config.output
    if only:
        assert config.data.cfg.skip_existing
        assert config.data.cfg.output_dir == str(tmp_path / "out/intermediate_designs")
    else:
        assert config.data.skip_existing
        assert config.data.skip_existing_kind == "inverse_fold"


@pytest.mark.parametrize("protocol", sorted(protocol_configs))
@pytest.mark.parametrize("command", ["configure", "run"])
def test_only_inverse_fold_cannot_skip_inverse_folding_early(
    tmp_path, monkeypatch, protocol, command
):
    from boltzgen.cli import boltzgen as cli

    args = build_parser().parse_args(
        [
            command,
            "input.yaml",
            "--output",
            str(tmp_path / "out"),
            "--protocol",
            protocol,
            "--only_inverse_fold",
            "--skip_inverse_folding",
        ]
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Conflicting inverse-fold flags must fail before setup")

    monkeypatch.setattr(cli, "get_artifact_path", unexpected)
    monkeypatch.setattr(cli, "load_canonicals", unexpected)
    monkeypatch.setattr(cli, "check_design_specs", unexpected)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected)

    with pytest.raises(ValueError, match="cannot be combined"):
        if command == "run":
            run_command(args)
        else:
            configure_command(args)
    with pytest.raises(ValueError, match="cannot be combined"):
        BinderDesignPipeline(args, tmp_path)
    assert not args.output.exists()


def test_solublempnn_download_cache_and_force(tmp_path, monkeypatch):
    calls = []

    def download(url, destination, hash_prefix):
        calls.append((url, hash_prefix))
        Path(destination).write_bytes(b"checkpoint")

    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    args = Namespace(cache=tmp_path, force_download=False)
    url = ARTIFACTS["solublempnn"][0]
    path = get_artifact_path(args, url)
    assert path.read_bytes() == b"checkpoint"
    assert get_artifact_path(args, url) == path
    assert calls == [(url, SOLUBLEMPNN_SHA256)]
    args.force_download = True
    assert get_artifact_path(args, url) == path
    assert len(calls) == 2
    assert get_artifact_path(args, str(path)) == path


def test_default_cache_works_in_clean_cli_process(tmp_path):
    # Importing hydra/Lightning in this test process eagerly imports Hub
    # constants, which can hide a broken lazy import in the actual CLI.
    cache = tmp_path / "hub"
    checkpoint = cache / "boltzgen/solublempnn_v_48_020.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"cached checkpoint")
    result = subprocess.run(
        [sys.executable, "-m", "boltzgen.cli.boltzgen", "download", "solublempnn"],
        env={**os.environ, "HF_HUB_CACHE": str(cache), "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(checkpoint) in result.stdout
    assert checkpoint.read_bytes() == b"cached checkpoint"


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_fails_before_loading_weights(temperature):
    with pytest.raises(ValueError, match="finite and positive"):
        SolubleMPNN("unused.pt", sampling_temperature=temperature)


def test_invalid_exclusions_fail_before_loading_weights():
    with pytest.raises(ValueError, match="every amino acid"):
        SolubleMPNN("unused.pt", inverse_fold_restriction=const.canonical_tokens)
    with pytest.raises(ValueError, match="canonical residue names"):
        SolubleMPNN("unused.pt", inverse_fold_restriction=["invalid"])


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        ("--solublempnn_sampling_temperature", value, "finite and positive")
        for value in ("0", "-1", "nan", "inf")
    ]
    + [
        ("--inverse_fold_avoid", value, "leave at least one amino acid")
        for value in ("ACDEFGHIKLMNPQRSTVWY", "X")
    ],
)
@pytest.mark.parametrize("command_entrypoint", [False, True])
def test_cli_rejects_invalid_sampling_options_before_device_or_model_access(
    tmp_path, monkeypatch, option, value, error, command_entrypoint
):
    args = build_parser().parse_args(
        [
            "configure",
            "input.yaml",
            "--output",
            str(tmp_path / "out"),
            option,
            value,
        ]
    )

    def unexpected_access(*args, **kwargs):
        pytest.fail("Invalid sampling options must fail before device/model access")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_access)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", unexpected_access)
    monkeypatch.setattr(torch.hub, "download_url_to_file", unexpected_access)
    with pytest.raises(ValueError, match=error):
        if command_entrypoint:
            configure_command(args)
        else:
            BinderDesignPipeline(args, tmp_path)
    assert not (tmp_path / "out").exists()


@pytest.fixture
def real_task(tmp_path, request):
    moldir = os.environ.get("BOLTZGEN_TEST_MOLDIR")
    checkpoint = os.environ.get("BOLTZGEN_TEST_SOLUBLEMPNN_CHECKPOINT")
    if not moldir or not checkpoint:
        pytest.skip("Set BOLTZGEN_TEST_MOLDIR and BOLTZGEN_TEST_SOLUBLEMPNN_CHECKPOINT")
    torch.set_num_threads(1)
    spec = tmp_path / "redesign.yaml"
    specification = (
        "entities:\n  - file:\n"
        f"      path: {ROOT / 'example/vanilla_protein/1g13.cif'}\n"
        "      include:\n        - chain:\n            id: A\n            res_index: 1..30\n"
        "      design:\n        - chain:\n            id: A\n            res_index: 10..20\n"
    )
    if ligand := getattr(request, "param", None):
        specification += f"  - ligand:\n      id: B\n      ccd: {ligand}\n"
    spec.write_text(specification)
    cfg = OmegaConf.load(CONFIGS / "inverse_fold_only_solublempnn.yaml")
    cfg.output = str(tmp_path / "inverse_folded")
    cfg.checkpoint = checkpoint
    cfg.data.cfg.moldir = moldir
    cfg.data.cfg.yaml_path = [str(spec)]
    cfg.data.cfg.multiplicity = 2
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.logger = False
    cfg.trainer.enable_progress_bar = False
    cfg.inverse_fold_restriction = [aa for aa in const.canonical_tokens if aa != "ALA"]
    return hydra.utils.instantiate(cfg)


def test_more_devices_than_samples_runs_without_empty_prediction_ranks(real_task):
    real_task.data.predict_set.dataset.multiplicity = 1
    real_task.trainer["devices"] = 2
    real_task.run()
    assert real_task.writer.failed == 0
    assert len(list(Path(real_task.output).glob("*.cif"))) == 1


def test_position_constraints_survive_sampling_and_generated_reload(
    real_task, tmp_path
):
    batch = next(iter(real_task.data.predict_dataloader()))
    positions = batch["design_mask"][0].nonzero().flatten()
    constraints = torch.zeros_like(batch["aa_constraint_mask"])
    constraints[0, positions[:2]] = 1
    constraints[0, positions[0], const.canonical_tokens.index("ALA")] = 0
    constraints[0, positions[1], const.canonical_tokens.index("CYS")] = 0
    batch["aa_constraint_mask"] = constraints
    model = SolubleMPNN(real_task.checkpoint, inverse_fold_restriction=["CYS"]).eval()
    with pytest.warns(RuntimeWarning, match="Relaxing per-residue constraints"):
        prediction = model.predict_step(batch)
    tokens = prediction["res_type"].argmax(-1)
    assert tokens[0, positions[0]] == const.token_ids["ALA"]
    assert (tokens[0, positions] != const.token_ids["CYS"]).all()
    real_task.writer.write_on_batch_end(
        prediction=prediction, batch=batch, sample_id="constrained"
    )
    config = OmegaConf.load(CONFIGS / "inverse_fold_solublempnn.yaml")
    config.output = str(tmp_path / "reloaded")
    config.checkpoint = real_task.checkpoint
    config.data.cfg.moldir = real_task.data.cfg.moldir
    config.data.cfg.num_workers = 0
    config.data.design_dir = real_task.output
    config.trainer.accelerator = "cpu"
    config.trainer.logger = False
    config.trainer.enable_progress_bar = False
    config.inverse_fold_restriction = ["CYS"]
    task = hydra.utils.instantiate(config)
    loaded = next(iter(task.data.predict_dataloader()))
    assert torch.equal(loaded["aa_constraint_mask"], constraints)
    with pytest.warns(RuntimeWarning, match="Relaxing per-residue constraints"):
        task.run()
    assert task.writer.failed == 0
    structure = gemmi.read_structure(str(next(Path(task.output).glob("*.cif"))))
    assert list(structure[0][0])[positions[0]].name == "ALA"


@pytest.mark.parametrize("temperature", [5e-324, 0.1, 1.7976931348623157e308])
def test_tied_sampling_intersects_constraints_and_preserves_fixed_tokens(
    real_task, temperature
):
    batch = next(iter(real_task.data.predict_dataloader()))
    positions = batch["design_mask"][0].nonzero().flatten()[:4]
    batch["symmetric_group"].zero_()
    batch["symmetric_group"][0, positions] = 1
    batch["feature_residue_index"][0, positions[:2]] = 100
    batch["feature_residue_index"][0, positions[2:]] = 101
    # A fixed group member remains fixed and is not included in sequence tying.
    batch["symmetric_group"][0, 0] = 1
    batch["feature_residue_index"][0, 0] = 100
    # Exclude an earlier fixed UNK token, exercising compressed MPNN indices.
    batch["res_type_clone"][0, 1].zero_()
    batch["res_type_clone"][0, 1, const.token_ids["UNK"]] = 1
    mask = batch["aa_constraint_mask"]
    mask[0, positions[:2]] = 1
    for position, allowed in zip(positions[:2], [("ALA", "SER"), ("SER", "GLY")]):
        mask[0, position, [const.canonical_tokens.index(aa) for aa in allowed]] = 0
    model = SolubleMPNN(real_task.checkpoint, sampling_temperature=temperature).eval()
    prediction = model.predict_step(batch)
    tokens = prediction["res_type"].argmax(-1)
    assert (tokens[0, positions[:2]] == const.token_ids["SER"]).all()
    assert tokens[0, positions[2]] == tokens[0, positions[3]]
    fixed = ~batch["design_mask"].bool()
    assert torch.equal(prediction["res_type"][fixed], batch["res_type_clone"][fixed])
    # Each position has an allowed identity, but the pair has no common one.
    mask[0, positions[0]] = 1
    mask[0, positions[0], const.canonical_tokens.index("ALA")] = 0
    with pytest.raises(ValueError, match="no common allowed amino acid"):
        model.predict_step(batch)
    model.tie_symmetric_sequences = False
    assert model.predict_step(batch)["exception"] is False


def test_yaml_homomer_tying_writes_matching_sequences(real_task, tmp_path):
    spec = tmp_path / "homomer.yaml"
    spec.write_text(
        "entities:\n  - file:\n"
        f"      path: {ROOT / 'example/vanilla_protein/1g13.cif'}\n"
        "      include:\n"
        "        - chain:\n            id: A\n            res_index: 1..30\n            symmetric_group: 1\n"
        "        - chain:\n            id: B\n            res_index: 1..30\n            symmetric_group: 1\n"
        "      design:\n"
        "        - chain:\n            id: A\n            res_index: 10..20\n"
        "        - chain:\n            id: B\n            res_index: 10..20\n"
    )
    config = OmegaConf.load(CONFIGS / "inverse_fold_only_solublempnn.yaml")
    config.output = str(tmp_path / "homomer")
    config.checkpoint = real_task.checkpoint
    config.data.cfg.moldir = real_task.data.cfg.moldir
    config.data.cfg.yaml_path = [str(spec)]
    config.data.cfg.multiplicity = 1
    config.data.num_workers = 0
    config.trainer.accelerator = "cpu"
    config.trainer.logger = False
    config.trainer.enable_progress_bar = False
    task = hydra.utils.instantiate(config)
    batch = next(iter(task.data.predict_dataloader()))
    assert (batch["symmetric_group"][batch["design_mask"].bool()] == 1).all()
    task.run()
    assert task.writer.failed == 0
    structure = gemmi.read_structure(str(next(Path(task.output).glob("*.cif"))))
    sequences = [[residue.name for residue in chain][9:20] for chain in structure[0]]
    assert len(sequences) == 2
    assert sequences[0] == sequences[1]

    reloaded_config = OmegaConf.load(CONFIGS / "inverse_fold_solublempnn.yaml")
    reloaded_config.output = str(tmp_path / "homomer_reloaded")
    reloaded_config.checkpoint = real_task.checkpoint
    reloaded_config.data.cfg.moldir = real_task.data.cfg.moldir
    reloaded_config.data.cfg.num_workers = 0
    reloaded_config.data.cfg.pin_memory = False
    reloaded_config.data.design_dir = task.output
    reloaded_config.trainer.accelerator = "cpu"
    reloaded_config.trainer.logger = False
    reloaded_config.trainer.enable_progress_bar = False
    reloaded = hydra.utils.instantiate(reloaded_config)
    loaded = next(iter(reloaded.data.predict_dataloader()))
    assert torch.equal(loaded["symmetric_group"], batch["symmetric_group"])
    reloaded.run()
    assert reloaded.writer.failed == 0
    structure = gemmi.read_structure(str(next(Path(reloaded.output).glob("*.cif"))))
    sequences = [[residue.name for residue in chain][9:20] for chain in structure[0]]
    assert len(sequences) == 2
    assert sequences[0] == sequences[1]


def test_memory_failure_skips_one_sample_and_continues(real_task, monkeypatch):
    batch = next(iter(real_task.data.predict_dataloader()))
    model = SolubleMPNN(real_task.checkpoint).eval()
    sample = model.model.sample
    cleared = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))

    def fail_once(features):
        monkeypatch.setattr(model.model, "sample", sample)
        raise torch.OutOfMemoryError("CUDA out of memory. Injected test failure.")

    monkeypatch.setattr(model.model, "sample", fail_once)
    assert model.predict_step(batch) == {"exception": True}
    assert cleared == [True]
    assert model.predict_step(batch)["exception"] is False

    def other_failure(features):
        raise RuntimeError("Unrelated model failure")

    monkeypatch.setattr(model.model, "sample", other_failure)
    with pytest.raises(RuntimeError, match="Unrelated model failure"):
        model.predict_step(batch)


@pytest.mark.parametrize("temperature", [5e-324, 1e-40, 1e39, 1.7976931348623157e308])
def test_extreme_positive_temperatures_keep_exclusions(real_task, temperature):
    batch = next(iter(real_task.data.predict_dataloader()))
    model = SolubleMPNN(
        real_task.checkpoint,
        sampling_temperature=temperature,
        inverse_fold_restriction=real_task.inverse_fold_restriction,
    ).eval()
    prediction = model.predict_step(batch)
    designed = batch["design_mask"].bool()
    assert (prediction["res_type"].argmax(-1)[designed] == const.token_ids["ALA"]).all()
    assert torch.equal(
        prediction["res_type"][~designed], batch["res_type_clone"][~designed]
    )


def test_generated_unknown_backbone_is_redesigned(real_task, tmp_path):
    """An ambiguous atom14 output must still reach sequence design and folding."""
    real_task.data.predict_set.backbone_only = False
    real_task.data.predict_set.atom14 = True
    batch = next(iter(real_task.data.predict_dataloader()))
    position = int(batch["design_mask"][0].nonzero()[0])
    atoms = batch["atom_to_token"][0, :, position].nonzero().flatten()
    assert len(atoms) == 14
    assert batch["fake_atom_mask"][0, atoms[-1]]
    coords = batch["coords"][:, 0].clone()
    # No canonical atom14 placement code uses CA for fake atoms. The normal
    # generation writer converts this ambiguous geometry to a backbone UNK.
    coords[0, atoms[-1]] = coords[0, atoms[1]]
    prediction = BoltzMasker(mask=True, mask_backbone=False)(batch)
    prediction.update(exception=False, coords=coords)
    generated = tmp_path / "generated"
    writer = DesignWriter(
        str(generated), res_atoms_only=False, atom14=True, write_native=False
    )
    writer.write_on_batch_end(prediction=prediction, batch=batch, sample_id="unknown")
    structure = gemmi.read_structure(str(generated / "unknown_0.cif"))
    assert list(structure[0][0])[position].name == "UNK"

    config = OmegaConf.load(CONFIGS / "inverse_fold_solublempnn.yaml")
    config.output = str(tmp_path / "redesigned")
    config.checkpoint = real_task.checkpoint
    config.data.cfg.moldir = real_task.data.cfg.moldir
    config.data.cfg.num_workers = 0
    config.data.cfg.pin_memory = False
    config.data.design_dir = str(generated)
    config.trainer.accelerator = "cpu"
    config.trainer.logger = False
    config.trainer.enable_progress_bar = False
    config.inverse_fold_restriction = real_task.inverse_fold_restriction
    task = hydra.utils.instantiate(config)
    task.run()
    assert task.writer.failed == 0
    structure = gemmi.read_structure(str(Path(task.output) / "unknown_0.cif"))
    assert list(structure[0][0])[position].name == "ALA"

    config = OmegaConf.load(CONFIGS / "fold.yaml")
    config.data.cfg.moldir = real_task.data.cfg.moldir
    config.data.cfg.num_workers = 0
    config.data.design_dir = task.output
    config.data.output_dir = str(tmp_path / "folded")
    data = hydra.utils.instantiate(config.data)
    folded_batch = next(iter(data.predict_dataloader()))
    assert not any(folded_batch["exception"])
    assert (
        folded_batch["res_type_clone"].argmax(-1)[0, position] == const.token_ids["ALA"]
    )


@pytest.mark.parametrize("real_task", [None, "RPB"], indirect=True)
def test_real_weights_write_and_reload_for_downstream_folding(real_task, tmp_path):
    """Exercise parser -> featurizer -> sampler -> CIF/NPZ writer -> fold loader."""
    original = json.loads(
        next(iter(real_task.data.predict_dataloader()))["source_context"][0]
    )
    real_task.run()
    assert real_task.writer.failed == 0
    output = Path(real_task.output)
    assert len(list(output.glob("*.cif"))) == 2
    assert len(list(output.glob("*.npz"))) == 2
    # Run the same data/task path used after diffusion generation, with a
    # different forced sequence so a bypassed second stage cannot pass.
    config = OmegaConf.load(CONFIGS / "inverse_fold_solublempnn.yaml")
    config.output = str(tmp_path / "from_generated")
    config.checkpoint = real_task.checkpoint
    config.data.cfg.moldir = real_task.data.cfg.moldir
    config.data.cfg.num_workers = 0
    config.data.cfg.pin_memory = False
    config.data.design_dir = str(output)
    config.trainer.accelerator = "cpu"
    config.trainer.logger = False
    config.trainer.enable_progress_bar = False
    config.inverse_fold_restriction = [
        aa for aa in const.canonical_tokens if aa != "GLY"
    ]
    generated_task = hydra.utils.instantiate(config)
    generated_task.run()
    assert generated_task.writer.failed == 0
    assert len(list(Path(generated_task.output).glob("*.cif"))) == 2

    fold_config = OmegaConf.load(CONFIGS / "fold.yaml")
    fold_config.data.cfg.moldir = real_task.data.cfg.moldir
    fold_config.data.cfg.num_workers = 0
    fold_config.data.cfg.pin_memory = False
    fold_config.data.design_dir = generated_task.output
    fold_config.data.output_dir = str(tmp_path / "folded")
    data = hydra.utils.instantiate(fold_config.data)
    batch = next(iter(data.predict_dataloader()))
    assert not any(batch["exception"])
    designed = batch["design_mask"].bool()
    assert designed.any()
    assert (
        batch["res_type_clone"].argmax(-1)[designed] == const.token_ids["GLY"]
    ).all()
    with np.load(next(output.glob("*.npz"))) as metadata:
        assert metadata["design_mask"].sum() == int(designed.sum())
        first_context = json.loads(str(metadata["source_context"].item()))
        expected_ligand_tokens = (
            metadata["mol_type"] == const.chain_type_ids["NONPOLYMER"]
        ).sum()
    assert (
        batch["mol_type"] == const.chain_type_ids["NONPOLYMER"]
    ).sum() == expected_ligand_tokens
    final_context = json.loads(batch["source_context"][0])
    protein = original["chains"][0]
    assert len(protein["residue_names"]) > len(protein["indices"])
    design_positions = designed[0, :len(protein["indices"])].nonzero().flatten().tolist()
    for context, identity in [(first_context, "ALA"), (final_context, "GLY")]:
        for index, (before, after) in enumerate(zip(
            original["chains"], context["chains"], strict=True
        )):
            assert after["indices"] == before["indices"]
            expected_names = before["residue_names"].copy()
            if index == 0:
                for position in design_positions:
                    expected_names[before["indices"][position]] = identity
            assert after["residue_names"] == expected_names
            assert after["complete"] == before["complete"]
    if expected_ligand_tokens:
        # Check ligand identity and its pose relative to the protein backbone
        # across serialization and a second inverse-folding pass.
        geometry = []
        identities = []
        for directory in (output, Path(generated_task.output)):
            structure = gemmi.read_structure(str(sorted(directory.glob("*.cif"))[0]))
            atoms = [
                (chain.name, residue, atom)
                for chain in structure[0]
                for residue in chain
                for atom in residue
                if residue.name == "RPB" or atom.name in {"N", "CA", "C", "O"}
            ]
            assert (
                sum(residue.name == "RPB" for _, residue, _ in atoms)
                == expected_ligand_tokens
            )
            identities.append(
                [
                    (chain, str(residue.seqid), atom.name)
                    for chain, residue, atom in atoms
                ]
            )
            coords = np.array(
                [[atom.pos.x, atom.pos.y, atom.pos.z] for _, _, atom in atoms]
            )
            geometry.append(np.linalg.norm(coords[:, None] - coords[None], axis=-1))
        assert identities[0] == identities[1]
        np.testing.assert_allclose(geometry[0], geometry[1], atol=0.005)

        affinity_config = OmegaConf.load(CONFIGS / "affinity.yaml")
        affinity_config.data.cfg.moldir = real_task.data.cfg.moldir
        affinity_config.data.cfg.num_workers = 0
        affinity_config.data.cfg.pin_memory = False
        affinity_config.data.design_dir = generated_task.output
        affinity_data = hydra.utils.instantiate(affinity_config.data)
        affinity_batch = next(iter(affinity_data.predict_dataloader()))
        assert not any(affinity_batch["exception"])
        assert (
            affinity_batch["mol_type"] == const.chain_type_ids["NONPOLYMER"]
        ).sum() == expected_ligand_tokens


def test_real_sampler_preserves_fixed_tokens_and_masks(real_task):
    batch = next(iter(real_task.data.predict_dataloader()))
    original = batch["res_type_clone"].clone()
    # Narrow the redesign region and include a nonprotein context token.
    design = batch["design_mask"].clone().bool()
    positions = design[0].nonzero().flatten()
    design[0, positions[0]] = False
    batch["inverse_fold_design_mask"] = design
    batch["mol_type"][0, 0] = const.chain_type_ids["NONPOLYMER"]
    model = SolubleMPNN(
        real_task.checkpoint,
        inverse_fold_restriction=real_task.inverse_fold_restriction,
    ).eval()
    prediction = model.predict_step(batch)
    assert not prediction["exception"]
    assert torch.equal(prediction["res_type"][~design], original[~design])
    assert (prediction["res_type"].argmax(-1)[design] == const.token_ids["ALA"]).all()
    assert torch.equal(batch["res_type_clone"], original)
    assert torch.equal(prediction["coords"], batch["coords"][:, 0])
    assert torch.equal(prediction["inverse_fold_design_mask"], design)
    assert model.predict_step({"skip": [True]}) == {"exception": True}

    atom = batch["token_to_bb4_atoms"][0, positions[-1], 3].long().argmax()
    batch["atom_resolved_mask"][0, atom] = False
    assert model.predict_step(batch) == {"exception": True}
    batch["atom_resolved_mask"][0, atom] = True
    assert model.predict_step(batch)["exception"] is False
    # Unsupported mask expansion is a configuration error even if the added
    # position also lacks a usable protein backbone.
    batch["inverse_fold_design_mask"][0, 0] = True
    with pytest.raises(ValueError, match="must be within design_mask"):
        model.predict_step(batch)
