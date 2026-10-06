"""Pinned model identity and the on-disk scoring contract."""

import hashlib
import json
from pathlib import Path
import shutil

MODEL_REPO = "biohub/ESMFold2"
MODEL_REVISION = "8fc3ff471022fdce52c77030685eb775de0c00a3"
ESMC_REPO = "biohub/ESMC-6B"
ESMC_REVISION = "af1602ba7406f521b11bf8f81d52af378cde09e4"
ESM_VERSION = "3.4.1.post1"
SCORE_DIR = "esmfold2_scores"
SCORE_KEY = "esmfold2_ipsae_min"
REDESIGN_SCORE_KEY = "esmfold2_score"
PTM_KEY = "esmfold2_ptm"
SCHEMA_VERSION = 1
ACCELERATION_REVISION = "anthropic-native-v1"


def validate_acceleration(mode: str) -> None:
    """Validate execution mode before provisioning the GPU runtime."""
    if mode not in ("auto", "fused", "off"):
        raise ValueError("ESMFold2 acceleration must be auto, fused, or off")


def validate_fused_size(tokens: int, samples: int, pair_width: int) -> None:
    """Keep the pinned fused pair-bias kernel within its signed int32 offsets."""
    if samples * tokens * tokens * pair_width >= 2**31:
        raise ValueError(
            "This crop and sample count exceed the pinned ESMFold2 fused kernel's "
            "32-bit indexing limit; use --esmfold2_acceleration auto or off"
        )


def validate_scoring_mode(mode: str, target_chains: list[str] | None) -> None:
    """Reject incompatible scoring settings before any inference work."""
    if mode not in ("binder", "redesign"):
        raise ValueError("scoring_mode must be binder or redesign")
    if mode == "redesign" and target_chains is not None:
        raise ValueError(
            "Redesign scoring uses every polymer chain; scoring_target_chains is only for binder scoring"
        )
    if target_chains is not None and (
        not target_chains or len(target_chains) != len(set(target_chains))
    ):
        raise ValueError(
            "scoring_target_chains must name nonempty, unique target polymer chains"
        )


def fingerprint(value: dict) -> str:
    """Hash the complete input, molecular context, and inference settings."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a design artifact without depending on timestamps."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_renamed_result(
    source: Path, destination: Path, old_id: str, new_id: str, design_path: Path
) -> str:
    """Preserve score provenance when the merge command renames a design."""
    request = json.loads((source / f"{old_id}.input.json").read_text())
    old_hash = fingerprint(request)
    result = load_result(source / f"{old_id}.json", old_hash)
    if request["design_sha256"] != file_sha256(design_path):
        raise ValueError(f"Cannot merge stale ESMFold2 score for {design_path}")
    request["design_id"] = new_id
    result["input_hash"] = fingerprint(request)
    result["merged_from"] = {"design_id": old_id, "input_hash": old_hash}
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("cif", "npz"):
        shutil.copy2(source / f"{old_id}.{suffix}", destination / f"{new_id}.{suffix}")
    for suffix, value in (("input.json", request), ("json", result)):
        (destination / f"{new_id}.{suffix}").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
    return result["input_hash"]


def load_result(path: Path, expected_input: str | None = None) -> dict:
    """Reject absent, stale, incompatible, or non-finite scoring results."""
    import math

    result = json.loads(path.read_text())
    if (
        result.get("schema_version") != SCHEMA_VERSION
        or result.get("model_revision") != MODEL_REVISION
        or result.get("esmc_revision") != ESMC_REVISION
        or result.get("esm_version") != ESM_VERSION
    ):
        raise ValueError(
            f"Incompatible ESMFold2 result: {path}; rerun esmfold2_scoring"
        )
    if expected_input is not None and result.get("input_hash") != expected_input:
        raise ValueError(f"Stale ESMFold2 result: {path}; rerun esmfold2_scoring")
    if not all(path.with_suffix(ext).is_file() for ext in (".cif", ".npz")):
        raise ValueError(
            f"Incomplete ESMFold2 artifacts: {path}; rerun esmfold2_scoring"
        )
    mode = result.get("scoring_mode", "binder")
    if mode == "binder":
        keys = [
            SCORE_KEY,
            "esmfold2_design_to_target_ipsae",
            "esmfold2_target_to_design_ipsae",
        ]
    elif mode == "redesign":
        metric = result.get("score_metric")
        if metric not in (SCORE_KEY, PTM_KEY):
            raise ValueError(f"Invalid ESMFold2 redesign metric in {path}")
        keys = [REDESIGN_SCORE_KEY, metric]
        if result.get("metrics", {}).get(REDESIGN_SCORE_KEY) != result.get(
            "metrics", {}
        ).get(metric):
            raise ValueError(f"Inconsistent ESMFold2 redesign score in {path}")
    else:
        raise ValueError(f"Invalid ESMFold2 scoring mode in {path}")
    for key in keys:
        value = result.get("metrics", {}).get(key)
        if (
            not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Missing or invalid {key} in {path}")
    return result
