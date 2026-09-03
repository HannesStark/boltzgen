from importlib.util import find_spec

# The Ascend runtime and RDKit can crash when torch-npu is loaded before RDKit.
# Import RDKit at package startup so a normal installation is self-contained
# and does not depend on an environment-level sitecustomize hook.
if find_spec("torch_npu") is not None:
    from rdkit import Chem as _rdkit_chem  # noqa: F401
