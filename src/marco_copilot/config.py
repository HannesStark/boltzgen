from __future__ import annotations

from pathlib import Path
from typing import Any
import json

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if yaml is None:
            data = json.loads(handle.read()) if handle.readable() else {}
        else:
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}")
    return data


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if yaml is None:
            handle.write(json.dumps(payload, indent=2))
        else:
            yaml.safe_dump(payload, handle, sort_keys=False)


def load_campaign(path: Path) -> dict[str, Any]:
    return load_yaml(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
