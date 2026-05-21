from __future__ import annotations

from pathlib import Path
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


def load_campaign(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        if yaml is None:
            raise RuntimeError("pyyaml is required for campaign parsing")
        return yaml.safe_load(f) or {}
