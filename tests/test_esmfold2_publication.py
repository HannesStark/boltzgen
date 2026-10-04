import json
import os

import pytest

from boltzgen.task.esmfold2 import score
from boltzgen.task.esmfold2.contract import (
    ESMC_REVISION,
    ESM_VERSION,
    MODEL_REVISION,
    SCHEMA_VERSION,
    fingerprint,
    load_result,
)


def test_interrupted_publication_removes_old_completion_marker(tmp_path, monkeypatch):
    outdir = tmp_path / "scores"
    staging_dir = tmp_path / "stage"
    outdir.mkdir()
    staging_dir.mkdir()
    design_id = "candidate"
    old_request = {"design_id": design_id, "sequence": "old"}
    new_request = {"design_id": design_id, "sequence": "new"}
    request_path = outdir / f"{design_id}.input.json"
    request_path.write_text(json.dumps(old_request))
    result_path = outdir / f"{design_id}.json"
    old_result = {
        "schema_version": SCHEMA_VERSION,
        "model_revision": MODEL_REVISION,
        "esmc_revision": ESMC_REVISION,
        "esm_version": ESM_VERSION,
        "input_hash": fingerprint(old_request),
        "metrics": {
            "esmfold2_ipsae_min": 0.5,
            "esmfold2_design_to_target_ipsae": 0.5,
            "esmfold2_target_to_design_ipsae": 0.4,
        },
    }
    result_path.write_text(json.dumps(old_result))
    (outdir / f"{design_id}.cif").write_text("old coordinates")
    (outdir / f"{design_id}.npz").write_bytes(b"old pae")
    staged_request_path = staging_dir / f"{design_id}.input.json"
    staged_request_path.write_text(json.dumps(new_request))
    staged_result = dict(old_result, input_hash=fingerprint(new_request))
    (staging_dir / f"{design_id}.json").write_text(json.dumps(staged_result))
    (staging_dir / f"{design_id}.cif").write_text("new coordinates")
    (staging_dir / f"{design_id}.npz").write_bytes(b"new pae")

    replace = os.replace
    calls = 0

    def fail_after_first_artifact(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interruption during publication")
        return replace(source, destination)

    monkeypatch.setattr(score.os, "replace", fail_after_first_artifact)
    with pytest.raises(OSError, match="simulated interruption"):
        score._publish_validated_result(
            staging_dir,
            outdir,
            request_path,
            staged_request_path,
            new_request,
        )

    assert not result_path.exists()
    assert json.loads(request_path.read_text()) == old_request
    assert (outdir / f"{design_id}.cif").read_text() == "new coordinates"
    assert (outdir / f"{design_id}.npz").read_bytes() == b"old pae"
    with pytest.raises(FileNotFoundError):
        load_result(result_path, fingerprint(old_request))
