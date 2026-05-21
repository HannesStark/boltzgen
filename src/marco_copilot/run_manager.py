from __future__ import annotations

from pathlib import Path
import stat

from marco_copilot.config import write_json


def create_run_scripts(
    spec_paths: list[Path],
    runs_dir: Path,
    budget: int,
    num_designs: int,
    protocol: str = "nanobody-anything",
    devices: int = 2,
) -> list[Path]:
    """Create per-spec run scripts.

    Notes:
    - BoltzGen manual documents `--devices` default as all available devices.
    - We pass `--devices` explicitly to ensure dual-GPU usage on 2-GPU nodes.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    scripts = []
    for spec in spec_paths:
        run_name = spec.stem
        run_dir = runs_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        script = run_dir / "run.sh"
        script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "DEVICES=\"${DEVICES:-" + str(devices) + "}\"\n"
            f"boltzgen check {spec}\n"
            f"boltzgen run {spec} --output {run_dir} --protocol {protocol} --budget {budget} --num_designs {num_designs} --devices \"$DEVICES\" --reuse\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        write_json(
            run_dir / "metadata.json",
            {
                "spec": str(spec),
                "budget": budget,
                "num_designs": num_designs,
                "protocol": protocol,
                "devices": devices,
                "status": "prepared",
            },
        )
        scripts.append(script)
    return scripts
