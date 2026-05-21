from __future__ import annotations

from pathlib import Path
from typing import Any

from marco_copilot.config import dump_yaml


def _default_spec(task: dict[str, Any], campaign: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "target": task["target"],
        "binder": {
            "scaffold": task["scaffold"],
            "length": task.get("binder_length", campaign.get("defaults", {}).get("binder_length", 125)),
        },
        "constraints": {
            "epitope": task.get("epitope", []),
            "mode": campaign.get("mode", "cross-reactive"),
        },
    }


def generate_specs(campaign: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for task in campaign.get("tasks", []):
        name = task["name"]
        spec = task.get("spec") or _default_spec(task, campaign)
        path = out_dir / f"{name}.yaml"
        dump_yaml(path, spec)
        created.append(path)
    return created
