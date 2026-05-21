from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "marco_boltzgen_design" / "runs" / "run_nanobody_campaign.sh"


def test_run_nanobody_campaign_uses_dual_gpu_default(tmp_path: Path):
    spec = tmp_path / "spec.yaml"
    spec.write_text("# Protocol: nanobody-hotspot\nversion: 1\n", encoding="utf-8")

    outdir = tmp_path / "out"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_file = tmp_path / "boltzgen_args.txt"

    fake_boltzgen = fake_bin / "boltzgen"
    fake_boltzgen.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$@\" > \"$BOLTZGEN_ARGS_LOG\"\n",
        encoding="utf-8",
    )
    fake_boltzgen.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["BOLTZGEN_ARGS_LOG"] = str(log_file)

    subprocess.run(
        ["bash", str(SCRIPT), str(spec), str(outdir)],
        check=True,
        env=env,
        cwd=REPO_ROOT,
        text=True,
    )

    args = log_file.read_text(encoding="utf-8")
    assert "run" in args
    assert f"{spec}" in args
    assert "--protocol nanobody-hotspot" in args
    assert "--devices 2" in args
    assert "--reuse" in args
