"""Optional test configuration for mocking heavy dependencies.

Enable with:
    pytest --mock-heavy-deps tests/test_residue_constraints.py

By default no mocking is performed, so integration tests run against
real dependencies.
"""
import sys
from unittest.mock import MagicMock


def _install_mock(name: str) -> None:
    """Install a mock module (and parent packages) into sys.modules."""
    parts = name.split(".")
    for i in range(len(parts)):
        mod_name = ".".join(parts[: i + 1])
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


# Heavy dependencies that schema.py imports transitively but are NOT
# needed by the three constraint-parsing functions under test.
_MOCK_MODULES = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.utils",
    "torch.utils.data",
    "pytorch_lightning",
    "hydra",
    "hydra.core",
    "hydra.core.config_store",
    "einops",
    "einx",
    "mashumaro",
    "biotite",
    "biotite.structure",
    "biotite.structure.io",
    "biotite.structure.io.pdbx",
    "pydssp",
    "logomaker",
    "hydride",
    "gemmi",
    "pdbeccdutils",
    "pdbeccdutils.core",
    "pdbeccdutils.core.ccd_reader",
    "edit_distance",
    "huggingface_hub",
    "nvidia_ml_py",
    "cuequivariance_ops_cu12",
    "cuequivariance_ops_torch_cu12",
    "cuequivariance_torch",
    "numba",
    "sklearn",
    "sklearn.cluster",
    "sklearn.neighbors",
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "tqdm",
    "Bio",
    "Bio.PDB",
]


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--mock-heavy-deps",
        action="store_true",
        default=False,
        help="Mock heavy optional dependencies for parser-only unit tests.",
    )


def pytest_configure(config) -> None:
    if config.getoption("--mock-heavy-deps"):
        for mod in _MOCK_MODULES:
            _install_mock(mod)
