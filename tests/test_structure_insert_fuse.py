"""Regression tests for Structure.insert and Structure.fuse index handling.

Both methods locate the target chain with ``np.where(...)[0]``, which returns
an array of matches. Treating that size-1 array as a scalar emits a
DeprecationWarning on older numpy and raises

    TypeError: only 0-dimensional arrays can be converted to Python scalars

on numpy >= 2.4, at parse time before any GPU work. These tests exercise both
methods directly and assert the resulting index bookkeeping, so the failure is
caught regardless of the installed numpy version.
"""

import numpy as np
import pytest

from boltzgen.data.data import (
    Atom,
    Bond,
    Chain,
    Coords,
    Ensemble,
    Interface,
    Residue,
    Structure,
)


def make_structure(n_chains=2, res_per_chain=3, atoms_per_res=2):
    """Build a minimal multi-chain Structure with consistent indices."""
    n_res = n_chains * res_per_chain
    n_atoms = n_res * atoms_per_res

    atoms = np.zeros(n_atoms, dtype=Atom)
    atoms["is_present"] = True

    coords = np.zeros(n_atoms, dtype=Coords)
    coords["coords"] = np.arange(n_atoms * 3, dtype=np.float32).reshape(n_atoms, 3)

    residues = np.zeros(n_res, dtype=Residue)
    for i in range(n_res):
        residues[i]["name"] = "ALA"
        residues[i]["res_idx"] = i % res_per_chain
        residues[i]["atom_idx"] = i * atoms_per_res
        residues[i]["atom_num"] = atoms_per_res
        residues[i]["atom_center"] = i * atoms_per_res
        residues[i]["atom_disto"] = i * atoms_per_res
        residues[i]["is_standard"] = True
        residues[i]["is_present"] = True

    chains = np.zeros(n_chains, dtype=Chain)
    for c in range(n_chains):
        chains[c]["name"] = chr(ord("A") + c)
        chains[c]["entity_id"] = c
        chains[c]["asym_id"] = c
        chains[c]["atom_idx"] = c * res_per_chain * atoms_per_res
        chains[c]["atom_num"] = res_per_chain * atoms_per_res
        chains[c]["res_idx"] = c * res_per_chain
        chains[c]["res_num"] = res_per_chain

    ensemble = np.zeros(1, dtype=Ensemble)
    ensemble[0]["atom_num"] = n_atoms

    return Structure(
        atoms=atoms,
        bonds=np.zeros(0, dtype=Bond),
        residues=residues,
        chains=chains,
        coords=coords,
        mask=np.ones(n_chains, dtype=bool),
        ensemble=ensemble,
        interfaces=np.zeros(0, dtype=Interface),
    )


def assert_chain_offsets_consistent(structure):
    """Chain res_idx/atom_idx must be the running totals of preceding chains."""
    expected_res = 0
    expected_atom = 0
    for chain in structure.chains:
        assert int(chain["res_idx"]) == expected_res
        assert int(chain["atom_idx"]) == expected_atom
        expected_res += int(chain["res_num"])
        expected_atom += int(chain["atom_num"])
    assert expected_res == len(structure.residues)
    assert expected_atom == len(structure.atoms)


class TestInsert:
    """Structure.insert."""

    @pytest.mark.parametrize(
        ("chain_name", "res_idx"),
        [("A", 0), ("A", 1), ("A", 3), ("B", 0), ("B", 1), ("B", 3)],
    )
    def test_insert_positions(self, chain_name, res_idx):
        """Insertion works at the start, middle and end of either chain."""
        structure = make_structure()
        num_residues = 2
        result = Structure.insert(structure, chain_name, res_idx, num_residues)

        assert len(result.residues) == len(structure.residues) + num_residues
        assert_chain_offsets_consistent(result)

        # only the target chain grew
        for original, updated in zip(structure.chains, result.chains):
            grew = int(updated["res_num"]) - int(original["res_num"])
            assert grew == (num_residues if original["name"] == chain_name else 0)

    def test_insert_preserves_existing_coords(self):
        """Coordinates before the insertion point are untouched."""
        structure = make_structure()
        result = Structure.insert(structure, "A", 1, 2)
        n_before = 1 * 2  # residues before insertion * atoms per residue
        np.testing.assert_array_equal(
            result.coords["coords"][:n_before],
            structure.coords["coords"][:n_before],
        )

    def test_insert_three_chains(self):
        """Chains after the target are shifted, chains before are not."""
        structure = make_structure(n_chains=3)
        result = Structure.insert(structure, "B", 1, 1)
        assert_chain_offsets_consistent(result)
        assert int(result.chains[0]["res_num"]) == 3
        assert int(result.chains[1]["res_num"]) == 4
        assert int(result.chains[2]["res_num"]) == 3


class TestFuse:
    """Structure.fuse."""

    @pytest.mark.parametrize("chain_name", ["A", "B"])
    def test_fuse_appends_to_chain(self, chain_name):
        structure1 = make_structure()
        structure2 = make_structure(n_chains=1, res_per_chain=2)
        result = Structure.fuse(structure1, structure2, chain_name)

        assert len(result.residues) == len(structure1.residues) + len(
            structure2.residues
        )
        assert len(result.atoms) == len(structure1.atoms) + len(structure2.atoms)
        assert_chain_offsets_consistent(result)

        for original, updated in zip(structure1.chains, result.chains):
            grew = int(updated["res_num"]) - int(original["res_num"])
            expected = len(structure2.residues) if original["name"] == chain_name else 0
            assert grew == expected

    def test_fuse_with_res_reindex(self):
        structure1 = make_structure()
        structure2 = make_structure(n_chains=1, res_per_chain=2)
        result = Structure.fuse(structure1, structure2, "A", res_reindex=True)
        assert_chain_offsets_consistent(result)
        # the fused residues continue the target chain's numbering
        chain_a_res = result.residues[: int(result.chains[0]["res_num"])]
        assert [int(r["res_idx"]) for r in chain_a_res] == [0, 1, 2, 3, 4]
