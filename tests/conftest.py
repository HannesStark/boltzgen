"""Test configuration: mock heavy dependencies for CPU-only unit tests.

The per-residue constraint functions under test (_normalize_aa_spec,
_convert_aa_names_to_indices, parse_residue_constraints) only use numpy
and the boltzgen.data.const module. However, schema.py transitively
imports torch, pytorch_lightning, etc. via other boltzgen modules.

This conftest patches those heavy imports so tests can run without GPU
libraries installed — achieving the "Level 1: No GPU, fast" goal.
"""
import sys
from types import ModuleType
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

for mod in _MOCK_MODULES:
    _install_mock(mod)
