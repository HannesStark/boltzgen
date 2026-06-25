"""Smoke tests for diffusion guidance integration."""

from __future__ import annotations

import json
import types
from pathlib import Path

import torch

from boltzgen.model.modules.diffusion import AtomDiffusion
from boltzgen.task.predict.data_from_yaml import _load_guidance_feats


def _build_guidance_module() -> AtomDiffusion:
    module = AtomDiffusion.__new__(AtomDiffusion)
    module.guidance_eps = 1e-6
    module._guidance_log_once = set()
    return module


def test_guidance_contract_defaults() -> None:
    feats = _load_guidance_feats(json_path=None, inline=None)
    assert feats["guidance_bbb_weight"] == 0.0
    assert feats["guidance_membrane_weight"] == 0.0
    assert feats["guidance_bbb_ckpt"] == ""
    assert feats["guidance_bbb_sigma_gate"] == 4.0
    assert feats["guidance_bbb_hidden"] == 64
    assert feats["guidance_bbb_layers"] == 3


def test_guidance_contract_json_merge(tmp_path: Path) -> None:
    payload = {
        "guidance_bbb_weight": 0.25,
        "guidance_membrane_weight": 0.75,
        "guidance_bbb_ckpt": "/tmp/best.ckpt",
        "guidance_bbb_hidden": 32,
    }
    guidance_json = tmp_path / "guidance_feats.json"
    guidance_json.write_text(json.dumps(payload), encoding="utf-8")

    feats = _load_guidance_feats(
        json_path=str(guidance_json),
        inline={"guidance_bbb_layers": 5},
    )

    assert feats["guidance_bbb_weight"] == 0.25
    assert feats["guidance_membrane_weight"] == 0.75
    assert feats["guidance_bbb_ckpt"] == "/tmp/best.ckpt"
    assert feats["guidance_bbb_hidden"] == 32
    assert feats["guidance_bbb_layers"] == 5


def test_bbb_guidance_baseline_is_noop() -> None:
    module = _build_guidance_module()
    out = module._compute_bbb_guidance(
        atom_coords=torch.randn(1, 8, 3),
        feats={},
        atom_mask=torch.ones(1, 8, dtype=torch.bool),
        sigma=2.0,
    )
    assert out is None


def test_bbb_guidance_missing_ckpt_is_noop() -> None:
    module = _build_guidance_module()
    out = module._compute_bbb_guidance(
        atom_coords=torch.randn(1, 8, 3),
        feats={"guidance_bbb_weight": 0.2, "guidance_membrane_weight": 0.0},
        atom_mask=torch.ones(1, 8, dtype=torch.bool),
        sigma=2.0,
    )
    assert out is None


def test_bbb_guidance_valid_path_applies_force(monkeypatch) -> None:
    module = _build_guidance_module()
    captured = {}

    fake_struct_guidance = types.ModuleType("bbb_geo.infer.struct_guidance")

    class BBBGuidanceConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def compute_bbb_guidance_force(atom_coords, feats, atom_mask, sigma, cfg):
        captured["sigma"] = sigma
        captured["cfg"] = cfg
        return torch.ones_like(atom_coords)

    fake_struct_guidance.BBBGuidanceConfig = BBBGuidanceConfig
    fake_struct_guidance.compute_bbb_guidance_force = compute_bbb_guidance_force

    monkeypatch.setitem(
        __import__("sys").modules, "bbb_geo", types.ModuleType("bbb_geo")
    )
    monkeypatch.setitem(
        __import__("sys").modules, "bbb_geo.infer", types.ModuleType("bbb_geo.infer")
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "bbb_geo.infer.struct_guidance",
        fake_struct_guidance,
    )

    atom_coords = torch.randn(1, 8, 3)
    out = module._compute_bbb_guidance(
        atom_coords=atom_coords,
        feats={
            "guidance_bbb_weight": 0.3,
            "guidance_membrane_weight": 0.7,
            "guidance_bbb_ckpt": "/tmp/best.ckpt",
            "guidance_bbb_sigma_gate": 4.0,
            "guidance_bbb_hidden": 64,
            "guidance_bbb_layers": 3,
            "guidance_max_force": 1.0,
        },
        atom_mask=torch.ones(1, 8, dtype=torch.bool),
        sigma=2.0,
    )

    assert out is not None
    assert torch.all(out == 1.0)
    assert captured["cfg"].ckpt_path == "/tmp/best.ckpt"
