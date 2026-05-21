from pathlib import Path

from marco_copilot.run_manager import create_run_scripts


def test_run_script_contains_devices(tmp_path: Path):
    spec = tmp_path / "s1.yaml"
    spec.write_text("version: 1\n", encoding="utf-8")
    scripts = create_run_scripts([spec], tmp_path / "runs", budget=20, num_designs=10, devices=2)
    assert len(scripts) == 1
    text = scripts[0].read_text(encoding="utf-8")
    assert '--devices "$DEVICES"' in text
    assert 'DEVICES="${DEVICES:-2}"' in text
