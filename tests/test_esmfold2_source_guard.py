"""Source fingerprints must work in managed runtimes with newer Python too."""

import pytest

from boltzgen.task.esmfold2.acceleration import check_source


def _source_fixture(value: int, *, offset: int = 1) -> int:
    return value + offset


def test_source_fingerprint_preserves_python312_representation():
    # Recorded with Python 3.12, where ast.dump includes empty fields by default.
    # This test also runs in the main environment without installing ESM.
    check_source(
        _source_fixture,
        "39814f3bae8ee76a2986b097d92db4d7d8733d3b385adaeb6debeb01e33d2ee1",
    )
    with pytest.raises(ValueError, match="Unsupported ESMFold2 source"):
        check_source(_source_fixture, "different-source")
