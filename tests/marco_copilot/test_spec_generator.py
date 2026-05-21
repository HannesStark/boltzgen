from pathlib import Path
from marco_copilot.spec_generator import generate_specs


def test_generate_specs(tmp_path: Path):
    campaign = {"specs": [{"name": "s1", "spec": {"a": 1}}]}
    out = tmp_path / "specs"
    files = generate_specs(campaign, out)
    assert len(files) == 1
    assert files[0].exists()
