"""Automatically provision ESMFold2 without changing BoltzGen's environment."""

from pathlib import Path
import subprocess

from boltzgen.task.esmfold2.contract import ESM_VERSION


def resolve_python(python: str | None = None, *, require_cuda: bool = False) -> str:
    """Return a checked interpreter, provisioning a cached runtime when needed.

    uv owns dependency resolution, environment locking and Python installation.
    Its tool cache is independent of the active environment and current project.
    A caller-supplied interpreter remains available for managed/offline installs.
    """
    if python is None:
        from uv import find_uv_bin

        requirements = (
            Path(__file__).resolve().parents[2] / "resources/runtime/esmfold2.txt"
        )
        command = [
            find_uv_bin(),
            "tool",
            "run",
            "--isolated",
            "--no-config",
            "--python",
            "3.12",
            "--from",
            f"esm=={ESM_VERSION}",
            "--with-requirements",
            str(requirements),
            "python",
        ]
    probe = (
        "import sys; from importlib.metadata import version; "
        "assert sys.version_info >= (3, 12); "
        f"assert version('esm') == {ESM_VERSION!r}; "
        "import torch; from esm.models.esmfold2 import EsmFold2Model; "
    )
    if require_cuda:
        probe += (
            "assert torch.cuda.is_available(), "
            "'ESMFold2 requires a visible GPU and CUDA 13 compatible driver'; "
            "torch.empty(1, device='cuda').add_(1); torch.cuda.synchronize(); "
        )
    probe += "print(sys.executable)"
    try:
        if python is None:
            print("Checking the cached ESMFold2 runtime...", flush=True)
            # uv normally refreshes stale index entries even for a cached tool.
            # Try the complete cache first so subsequent runs work offline.
            discovery = ["-I", "-c", "import sys; print(sys.executable)"]
            cached = subprocess.run(
                [command[0], "--offline", *command[1:], *discovery],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if cached.returncode != 0:
                print(
                    "Preparing the ESMFold2 runtime: downloading Python and dependencies "
                    "as needed (about 6 GB on first use). This may take several minutes. "
                    "The runtime is cached for later runs; UV_CACHE_DIR controls its location.",
                    flush=True,
                )
                cached = subprocess.run(
                    [*command, *discovery],
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                )
            python = cached.stdout.strip().splitlines()[-1]
        print(f"Checking ESMFold2 dependencies and device using {python}...", flush=True)
        # Validate separately: import/CUDA failures in an existing environment
        # must surface directly, never be mistaken for an installer cache miss.
        result = subprocess.run(
            [python, "-I", "-c", probe],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Could not prepare the ESMFold2 runtime. See the installer/import/CUDA "
            "error above and check the interpreter, dependencies, and GPU driver. "
            "An uncached installation also needs network access and space in the uv "
            "cache. Managed installations may set --esmfold2_python."
        ) from exc
    resolved = result.stdout.strip().splitlines()[-1]
    print(f"ESMFold2 runtime ready: {resolved}", flush=True)
    return resolved


def worker_command(python: str, manifest: Path, device: str) -> list[str]:
    """Launch only our worker source in the isolated dependency environment."""
    return [
        python,
        "-I",
        str(Path(__file__).with_name("worker.py")),
        str(manifest),
        "--device",
        device,
    ]
