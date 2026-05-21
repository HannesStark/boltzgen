from __future__ import annotations

from pathlib import Path
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


def generate_specs(campaign: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for item in campaign.get("specs", []):
        name = item["name"]
        path = out_dir / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            if yaml is None:
                import json
                f.write(json.dumps(item.get("spec", {}), indent=2))
            else:
                yaml.safe_dump(item.get("spec", {}), f, sort_keys=False)
        created.append(path)
    return created
