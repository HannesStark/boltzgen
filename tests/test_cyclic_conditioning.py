"""Explicit bonds only imply backbone cyclization at the N/C termini."""

from pathlib import Path

import numpy as np
import pytest
from test_structure_transform_boundaries import parse_entities

from boltzgen.data.data import Input, Structure
from boltzgen.data.feature.featurizer import process_token_features
from boltzgen.data.tokenize.tokenizer import Tokenizer


def cyclic_features(structure: Structure) -> np.ndarray:
    tokenized = Tokenizer().tokenize(structure)
    features = process_token_features(
        Input(
            tokenized.tokens,
            tokenized.bonds,
            tokenized.token_to_res,
            structure,
            {},
            {},
        ),
        random=np.random.default_rng(0),
    )
    return features["cyclic"].numpy()


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("disulfide", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_terminal_disulfide_does_not_set_backbone_cyclization(
    tmp_path: Path, prefix: bool, cyclic: bool, disulfide: bool, reverse: bool
) -> None:
    entities = [{"protein": {"id": "P", "sequence": "GG"}}] if prefix else []
    entities.append({"protein": {"id": "D", "sequence": "CGC", "cyclic": cyclic}})
    constraints = (
        [{"bond": {"atom1": ["D", 1, "SG"], "atom2": ["D", 3, "SG"]}}]
        if disulfide
        else None
    )
    structure = parse_entities(tmp_path, entities, constraints).structure
    if reverse:
        for field in ("chain", "res", "atom"):
            first = structure.bonds[f"{field}_1"].copy()
            structure.bonds[f"{field}_1"] = structure.bonds[f"{field}_2"]
            structure.bonds[f"{field}_2"] = first
    # Reconstructed/file-derived structures can lack the cached cyclic period.
    structure.chains["cyclic_period"] = 0
    bonds_before = structure.bonds.copy()
    expected = ([0, 0] if prefix else []) + [3 if cyclic else 0] * 3
    np.testing.assert_array_equal(cyclic_features(structure), expected)
    np.testing.assert_array_equal(structure.bonds, bonds_before)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (["A", 1, "C"], ["A", 3, "N"]),
        (["A", 2, "N"], ["A", 3, "C"]),
        (["A", 1, "N"], ["B", 3, "C"]),
    ],
)
def test_other_nc_bonds_do_not_close_the_backbone(
    tmp_path: Path, first: list, second: list
) -> None:
    structure = parse_entities(
        tmp_path,
        [{"protein": {"id": ["A", "B"], "sequence": "GGG"}}],
        [{"bond": {"atom1": first, "atom2": second}}],
    ).structure
    np.testing.assert_array_equal(cyclic_features(structure), np.zeros(6))


def test_cyclic_period_uses_asym_id_independently_of_chain_table_index(
    tmp_path: Path,
) -> None:
    structure = parse_entities(
        tmp_path,
        [
            {"protein": {"id": "A", "sequence": "GGG"}},
            {"protein": {"id": "B", "sequence": "GGG", "cyclic": True}},
        ],
    ).structure
    structure.chains["asym_id"] = [7, 3]
    np.testing.assert_array_equal(cyclic_features(structure), [0, 0, 0, 3, 3, 3])
