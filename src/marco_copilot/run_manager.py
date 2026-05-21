from __future__ import annotations

from pathlib import Path


def create_run_scripts(spec_paths: list[Path], runs_dir: Path, budget: int, num_designs: int) -> list[Path]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    scripts = []
    for spec in spec_paths:
        run_name = spec.stem
        run_dir = runs_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        script = run_dir / "run.sh"
        script.write_text(
            "#!/usr/bin/env bash\n"
            f"boltzgen run {spec} --output {run_dir} --budget {budget} --num_designs {num_designs}\n",
            encoding="utf-8",
        )
        scripts.append(script)
    return scripts
