from importlib.util import find_spec

# The Ascend runtime and RDKit can crash when torch-npu is loaded first. Load
# RDKit before importing the accelerator helper, which imports torch.
if find_spec("torch_npu") is not None:
    from rdkit import Chem as _rdkit_chem  # noqa: F401

from boltzgen.utils.accelerator import enable_npu_compat

enable_npu_compat()
