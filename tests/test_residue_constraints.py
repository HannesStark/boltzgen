"""Unit tests for per-residue amino acid constraint parsing.

Tests parse_residue_constraints(), _normalize_aa_spec(), and
_convert_aa_names_to_indices() from boltzgen.data.parse.schema.
"""

import numpy as np
import pytest

from boltzgen.data import const
from boltzgen.data.parse.schema import (
    _convert_aa_names_to_indices,
    _normalize_aa_spec,
    parse_residue_constraints,
)

# Shorthand fixtures
CANONICAL = const.canonical_tokens  # 20 three-letter codes
LETTER_MAP = const.prot_letter_to_token  # e.g. {"A": "ALA", ...}


# ============================================================================
# _normalize_aa_spec
# ============================================================================

class TestNormalizeAASpec:
    """Tests for _normalize_aa_spec helper."""

    def test_single_letter(self):
        assert _normalize_aa_spec("A") == ["A"]

    def test_multi_letter_string(self):
        assert _normalize_aa_spec("AGS") == ["A", "G", "S"]

    def test_long_string(self):
        assert _normalize_aa_spec("AVILMFYW") == list("AVILMFYW")

    def test_three_letter_code(self):
        assert _normalize_aa_spec("ALA") == ["ALA"]

    def test_three_letter_not_valid(self):
        # "AGS" is 3 chars but NOT a valid 3-letter code → split into 1-letter
        assert _normalize_aa_spec("AGS") == ["A", "G", "S"]

    def test_list_format_single_letters(self):
        assert _normalize_aa_spec(["A", "G", "S"]) == ["A", "G", "S"]

    def test_list_format_three_letter(self):
        assert _normalize_aa_spec(["ALA", "GLY"]) == ["ALA", "GLY"]

    def test_lowercase_normalised(self):
        assert _normalize_aa_spec("ags") == ["A", "G", "S"]

    def test_whitespace_stripped(self):
        assert _normalize_aa_spec(" AG ") == ["A", "G"]

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            _normalize_aa_spec(123)


# ============================================================================
# _convert_aa_names_to_indices
# ============================================================================

class TestConvertAANamesToIndices:
    """Tests for _convert_aa_names_to_indices helper."""

    def test_single_letter_a(self):
        indices = _convert_aa_names_to_indices(["A"], CANONICAL, LETTER_MAP)
        assert indices == [CANONICAL.index("ALA")]

    def test_single_letter_c(self):
        indices = _convert_aa_names_to_indices(["C"], CANONICAL, LETTER_MAP)
        assert indices == [CANONICAL.index("CYS")]

    def test_three_letter_code(self):
        indices = _convert_aa_names_to_indices(["ALA", "GLY"], CANONICAL, LETTER_MAP)
        assert indices == [CANONICAL.index("ALA"), CANONICAL.index("GLY")]

    def test_mixed_formats(self):
        indices = _convert_aa_names_to_indices(["A", "GLY"], CANONICAL, LETTER_MAP)
        assert indices == [CANONICAL.index("ALA"), CANONICAL.index("GLY")]

    def test_all_20_aas(self):
        all_letters = list("ACDEFGHIKLMNPQRSTVWY")
        indices = _convert_aa_names_to_indices(all_letters, CANONICAL, LETTER_MAP)
        assert len(indices) == 20
        assert len(set(indices)) == 20  # all unique

    def test_invalid_letter_raises(self):
        with pytest.raises(ValueError, match="Unknown amino acid"):
            _convert_aa_names_to_indices(["X"], CANONICAL, LETTER_MAP)

    def test_invalid_three_letter_raises(self):
        with pytest.raises(ValueError, match="Unknown amino acid"):
            _convert_aa_names_to_indices(["ZZZ"], CANONICAL, LETTER_MAP)


# ============================================================================
# parse_residue_constraints — valid inputs
# ============================================================================

class TestParseResidueConstraintsValid:
    """Tests for parse_residue_constraints with valid YAML specs."""

    def test_empty_list_returns_zeros(self):
        mask = parse_residue_constraints([], 10, CANONICAL, LETTER_MAP)
        assert mask.shape == (10, 20)
        assert mask.sum() == 0.0

    def test_single_allowed(self):
        spec = [{"position": 1, "allowed": "A"}]
        mask = parse_residue_constraints(spec, 5, CANONICAL, LETTER_MAP)
        ala_idx = CANONICAL.index("ALA")
        # Position 0 (1-indexed=1): only ALA allowed (0.0), rest blocked (1.0)
        assert mask[0, ala_idx] == 0.0
        assert mask[0].sum() == 19.0  # 19 blocked, 1 allowed
        # Other positions untouched
        assert mask[1:].sum() == 0.0

    def test_single_disallowed(self):
        spec = [{"position": 3, "disallowed": "CM"}]
        mask = parse_residue_constraints(spec, 5, CANONICAL, LETTER_MAP)
        cys_idx = CANONICAL.index("CYS")
        met_idx = CANONICAL.index("MET")
        # Position 2 (1-indexed=3): CYS and MET blocked
        assert mask[2, cys_idx] == 1.0
        assert mask[2, met_idx] == 1.0
        assert mask[2].sum() == 2.0  # only 2 blocked

    def test_range_positions(self):
        spec = [{"position": "3..5", "disallowed": "C"}]
        mask = parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)
        cys_idx = CANONICAL.index("CYS")
        # Positions 2,3,4 (1-indexed 3,4,5) should have CYS blocked
        for pos in [2, 3, 4]:
            assert mask[pos, cys_idx] == 1.0
        # Other positions untouched
        for pos in [0, 1, 5, 6, 7, 8, 9]:
            assert mask[pos, cys_idx] == 0.0

    def test_allowed_multiple_aas(self):
        spec = [{"position": 8, "allowed": "AGS"}]
        mask = parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)
        ala_idx = CANONICAL.index("ALA")
        gly_idx = CANONICAL.index("GLY")
        ser_idx = CANONICAL.index("SER")
        # Position 7 (1-indexed=8): only A,G,S allowed
        assert mask[7, ala_idx] == 0.0
        assert mask[7, gly_idx] == 0.0
        assert mask[7, ser_idx] == 0.0
        assert mask[7].sum() == 17.0  # 17 blocked

    def test_list_format_allowed(self):
        spec = [{"position": 1, "allowed": ["A", "G"]}]
        mask = parse_residue_constraints(spec, 5, CANONICAL, LETTER_MAP)
        ala_idx = CANONICAL.index("ALA")
        gly_idx = CANONICAL.index("GLY")
        assert mask[0, ala_idx] == 0.0
        assert mask[0, gly_idx] == 0.0
        assert mask[0].sum() == 18.0

    def test_multiple_constraints_no_overlap(self):
        spec = [
            {"position": 1, "allowed": "A"},
            {"position": 5, "allowed": "P"},
        ]
        mask = parse_residue_constraints(spec, 5, CANONICAL, LETTER_MAP)
        ala_idx = CANONICAL.index("ALA")
        pro_idx = CANONICAL.index("PRO")
        assert mask[0, ala_idx] == 0.0
        assert mask[0].sum() == 19.0
        assert mask[4, pro_idx] == 0.0
        assert mask[4].sum() == 19.0
        # Middle positions untouched
        assert mask[1:4].sum() == 0.0

    # ------------------------------------------------------------------
    # Intersection semantics (overlapping constraints)
    # ------------------------------------------------------------------

    def test_overlapping_allowed_intersection(self):
        """Two allowed constraints on same position → only common AAs survive."""
        spec = [
            {"position": 1, "allowed": "AG"},
            {"position": 1, "allowed": "GS"},
        ]
        mask = parse_residue_constraints(spec, 5, CANONICAL, LETTER_MAP)
        gly_idx = CANONICAL.index("GLY")
        ala_idx = CANONICAL.index("ALA")
        ser_idx = CANONICAL.index("SER")
        # Only G is in both sets
        assert mask[0, gly_idx] == 0.0  # allowed
        assert mask[0, ala_idx] == 1.0  # blocked (not in 2nd)
        assert mask[0, ser_idx] == 1.0  # blocked (not in 1st)
        assert mask[0].sum() == 19.0  # only GLY allowed

    def test_overlapping_allowed_range_intersection(self):
        """Overlapping ranges intersect at overlap positions."""
        spec = [
            {"position": "1..5", "allowed": "AG"},
            {"position": "3..7", "allowed": "GS"},
        ]
        mask = parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)
        gly_idx = CANONICAL.index("GLY")
        ala_idx = CANONICAL.index("ALA")
        ser_idx = CANONICAL.index("SER")
        # Positions 0,1 (1-indexed 1,2): only AG (first constraint only)
        assert mask[0, ala_idx] == 0.0
        assert mask[0, gly_idx] == 0.0
        assert mask[0].sum() == 18.0
        # Positions 2,3,4 (1-indexed 3,4,5): intersection of {A,G} and {G,S} = {G}
        for pos in [2, 3, 4]:
            assert mask[pos, gly_idx] == 0.0
            assert mask[pos, ala_idx] == 1.0
            assert mask[pos, ser_idx] == 1.0
            assert mask[pos].sum() == 19.0
        # Positions 5,6 (1-indexed 6,7): only GS (second constraint only)
        assert mask[5, gly_idx] == 0.0
        assert mask[5, ser_idx] == 0.0
        assert mask[5].sum() == 18.0

    def test_allowed_then_disallowed_same_position(self):
        """allowed + disallowed on same position: disallowed narrows the set."""
        spec = [
            {"position": 5, "allowed": "AGILMV"},
            {"position": 5, "disallowed": "CM"},
        ]
        mask = parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)
        met_idx = CANONICAL.index("MET")
        ala_idx = CANONICAL.index("ALA")
        # M was in allowed set but then blocked by disallowed
        assert mask[4, met_idx] == 1.0
        # A was in allowed set and not disallowed
        assert mask[4, ala_idx] == 0.0

    def test_disallowed_then_allowed_same_position(self):
        """Order independent: disallowed then allowed gives same result."""
        spec_ab = [
            {"position": 5, "allowed": "AGILMV"},
            {"position": 5, "disallowed": "CM"},
        ]
        spec_ba = [
            {"position": 5, "disallowed": "CM"},
            {"position": 5, "allowed": "AGILMV"},
        ]
        mask_ab = parse_residue_constraints(spec_ab, 10, CANONICAL, LETTER_MAP)
        mask_ba = parse_residue_constraints(spec_ba, 10, CANONICAL, LETTER_MAP)
        np.testing.assert_array_equal(mask_ab, mask_ba)

    def test_disjoint_allowed_sets_all_blocked(self):
        """Two allowed sets with no overlap → all 20 AAs blocked."""
        spec = [
            {"position": 1, "allowed": "AG"},
            {"position": 1, "allowed": "VILM"},
        ]
        mask = parse_residue_constraints(spec, 5, CANONICAL, LETTER_MAP)
        # All 20 blocked at position 0
        assert mask[0].sum() == 20.0

    def test_multiple_disallowed_accumulate(self):
        """Multiple disallowed on same position: union of blocked sets."""
        spec = [
            {"position": 1, "disallowed": "CM"},
            {"position": 1, "disallowed": "WK"},
        ]
        mask = parse_residue_constraints(spec, 5, CANONICAL, LETTER_MAP)
        cys_idx = CANONICAL.index("CYS")
        met_idx = CANONICAL.index("MET")
        trp_idx = CANONICAL.index("TRP")
        lys_idx = CANONICAL.index("LYS")
        assert mask[0, cys_idx] == 1.0
        assert mask[0, met_idx] == 1.0
        assert mask[0, trp_idx] == 1.0
        assert mask[0, lys_idx] == 1.0
        assert mask[0].sum() == 4.0

    def test_dtype_and_shape(self):
        spec = [{"position": 1, "allowed": "A"}]
        mask = parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)
        assert mask.dtype == np.float32
        assert mask.shape == (10, 20)


# ============================================================================
# parse_residue_constraints — error paths
# ============================================================================

class TestParseResidueConstraintsErrors:
    """Tests for parse_residue_constraints with invalid YAML specs."""

    def test_missing_position(self):
        spec = [{"allowed": "A"}]
        with pytest.raises(ValueError, match="position.*required"):
            parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)

    def test_position_out_of_bounds_high(self):
        spec = [{"position": 11, "allowed": "A"}]
        with pytest.raises(ValueError, match="out of bounds"):
            parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)

    def test_position_out_of_bounds_zero(self):
        # Position 0 is invalid (1-indexed); parse_range catches this
        spec = [{"position": 0, "allowed": "A"}]
        with pytest.raises(ValueError, match="1 indexed|out of bounds"):
            parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)

    def test_both_allowed_and_disallowed(self):
        spec = [{"position": 1, "allowed": "A", "disallowed": "C"}]
        with pytest.raises(ValueError, match="cannot specify both"):
            parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)

    def test_neither_allowed_nor_disallowed(self):
        spec = [{"position": 1}]
        with pytest.raises(ValueError, match="must specify either"):
            parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)

    def test_empty_allowed(self):
        spec = [{"position": 1, "allowed": ""}]
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)

    def test_invalid_amino_acid_code(self):
        spec = [{"position": 1, "allowed": "X"}]
        with pytest.raises(ValueError, match="Unknown amino acid"):
            parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)

    def test_invalid_amino_acid_in_disallowed(self):
        spec = [{"position": 1, "disallowed": "XZ"}]
        with pytest.raises(ValueError, match="Unknown amino acid"):
            parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)


# ============================================================================
# Regression: original test case (no overlaps)
# ============================================================================

class TestOriginalTestCase:
    """Regression test matching residue_constraints_test.yaml."""

    def test_original_yaml_constraints(self):
        """Matches the constraints from example/residue_constraints_test.yaml."""
        spec = [
            {"position": 1, "allowed": "A"},
            {"position": "3..5", "disallowed": "CM"},
            {"position": 8, "allowed": "AGS"},
            {"position": 10, "allowed": "P"},
        ]
        mask = parse_residue_constraints(spec, 10, CANONICAL, LETTER_MAP)

        ala_idx = CANONICAL.index("ALA")
        cys_idx = CANONICAL.index("CYS")
        met_idx = CANONICAL.index("MET")
        gly_idx = CANONICAL.index("GLY")
        ser_idx = CANONICAL.index("SER")
        pro_idx = CANONICAL.index("PRO")

        # Position 1: only A
        assert mask[0, ala_idx] == 0.0
        assert mask[0].sum() == 19.0

        # Positions 3-5: C and M blocked
        for pos in [2, 3, 4]:
            assert mask[pos, cys_idx] == 1.0
            assert mask[pos, met_idx] == 1.0
            assert mask[pos].sum() == 2.0

        # Position 8: only A, G, S
        assert mask[7, ala_idx] == 0.0
        assert mask[7, gly_idx] == 0.0
        assert mask[7, ser_idx] == 0.0
        assert mask[7].sum() == 17.0

        # Position 10: only P
        assert mask[9, pro_idx] == 0.0
        assert mask[9].sum() == 19.0

        # Unconstrained positions (2, 6, 7, 9 in 0-indexed) are all zeros
        for pos in [1, 5, 6, 8]:
            assert mask[pos].sum() == 0.0
