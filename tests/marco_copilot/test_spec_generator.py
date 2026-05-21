from pathlib import Path
from marco_copilot.spec_generator import generate_specs


def test_generate_specs(tmp_path: Path):
    campaign = {
        "mode": "cross-reactive",
        "tasks": [{"name": "s1", "target": "t1.pdb", "scaffold": "s1.cif", "epitope": [1, 2]}],
    }
    out = tmp_path / "specs"
    files = generate_specs(campaign, out)
    assert len(files) == 1
    assert files[0].exists()
    text = files[0].read_text()
    assert "t1.pdb" in text
