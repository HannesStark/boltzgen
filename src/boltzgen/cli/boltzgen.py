#!/usr/bin/env python3
"""
This script orchestrates work. It sets up an output directory with yaml files of pipeline steps that need to be run, and launches processes that run the pipeline steps.

Mainly it:
1) **Writes to yaml files** when `configure_command(...)` is executed
   - For each `PipelineStep`, the resolved Hydra config is written to
     `OUTPUT/config/<step>.yaml`.
   - A manifest `OUTPUT/steps.yaml` is also written, listing the enabled steps
     and their config files in execution order.

2) **Executes from YAML** when `execute_command(...)` is executed
   - Each step is launched **as a subprocess** (`python main.py <config.yaml>`)
     unless `--no_subprocess` is set (not the default).
   - If `--no_subprocess` is specified, the config is instantiated in-process
     and the `Task.run(...)` method is called directly.

The actual code that is exectued in each pipeline step is found in `main.py` which a wrapper for running the .run() function of our `Task` class.
If you run the pipeline (for example via `boltzgen run design_spec.yaml ...`) then this function reads the yaml files of the individual pipeline steps and executes the pipeline steps.

The possible tasks (and code files you want to inspect to understand what they are running):
    - Predict src/boltzgen/task/predict/predict.py (GPU: Running BoltzGen diffusion, inverse folding, refolding, designfolding, or affinity prediction)
    - Analyze src/boltzgen/task/analyze/analyze.py (CPU: Compute CPU Metrics and aggregate metrics from GPU steps)
    - Filter src/boltzgen/task/filter/filter.py (CPU: Very fast (20s) computes ranking and writes final output files)
"""

from boltzgen.utils.quiet import quiet_startup

quiet_startup()

import collections
import huggingface_hub
import argparse
from dataclasses import dataclass
import shlex
import subprocess
import os
import time
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml
import hydra
import omegaconf
import torch

from boltzgen.data import const
from boltzgen.data.mol import load_canonicals
from boltzgen.data.parse.schema import YamlDesignParser
from boltzgen.data.write.mmcif import to_mmcif
from boltzgen.task.task import Task
from importlib.metadata import PackageNotFoundError, version as pkg_version

### Paths and constants ####
# Get the path to the project root (where main.py and configs/ are located)
# Since we're now in src/boltzgen/, we need to go up 3 levels
path_to_script = Path(__file__)
project_root = (
    path_to_script.parent.parent
)  # Go up from src/boltzgen/cli to project root
config_dir = project_root / "resources/config"
main_script = project_root / "resources/main.py"

step_names = [
    "design",
    "inverse_folding",
    "design_folding",
    "folding",
    "affinity",
    "analysis",
    "filtering",
]

### Protocol-specific configuration overrides (which can be overridden by user) ####
protocol_configs = {
    "protein-anything": {},  # base config corresponds to protein-anything
    "peptide-anything": {
        # Note that in inverse folding step we also avoid cysteines by default; this is implemented elsewhere.
        "analysis": ["largest_hydrophobic=false", "largest_hydrophobic_refolded=false"],
        "filtering": [
            "filter_cysteine=true",
            "alpha=0.01",
            "refolding_rmsd_threshold=2",
        ],
    },
    "protein-small_molecule": {
        "analysis": ["affinity_metrics=true"],
        "filtering": ["use_affinity=true"],
    },
    "nanobody-anything": {
        # Similar to peptide-anything we also avoid cysteins in inverse folding by default.
        "analysis": ["largest_hydrophobic=false", "largest_hydrophobic_refolded=false"],
        "filtering": ["filter_cysteine=true"],
    },
}
assert all(
    step_name in step_names for cfg in protocol_configs.values() for step_name in cfg
)


### Model checkpoints and other artifacts ####
ARTIFACTS: dict[str, tuple[str, str]] = {
    "design-diverse": (
        "huggingface:boltzgen/boltzgen-1:boltzgen1_diverse.ckpt",
        "model",
    ),
    "design-adherence": (
        "huggingface:boltzgen/boltzgen-1:boltzgen1_adherence.ckpt",
        "model",
    ),
    "inverse-fold": (
        "huggingface:boltzgen/boltzgen-1:boltzgen1_ifold.ckpt",
        "model",
    ),
    "folding": (
        "huggingface:boltzgen/boltzgen-1:boltz2_conf_final.ckpt",
        "model",
    ),
    "affinity": ("huggingface:boltzgen/boltzgen-1:boltz2_aff.ckpt", "model"),
    "moldir": ("huggingface:boltzgen/inference-data:mols.zip", "dataset"),
}


### CLI arguments ###
def add_configure_arguments(
    parser: argparse.ArgumentParser, *, output_required: bool = False
) -> None:
    # General configuration options
    p = parser.add_argument_group("general configuration")
    p.add_argument(
        "--protocol",
        type=str,
        choices=list(protocol_configs.keys()),
        default="protein-anything",
        help="Protocol to use for the design. This determines default settings and in some cases what steps "
        "are run. Default: %(default)s",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=output_required,
        help="Output directory for pipeline results",
    )
    p.add_argument(
        "--config",
        nargs="+",
        action="append",
        help="Override pipeline step configuration, in format <step_name> <arg1>=<value1> <arg2>=<value2> ..."
        "(example: '--config folding num_workers=4 trainer.devices=4'). Can be used multiple times.",
    )
    p.add_argument(
        "--devices",
        type=int,
        help="Number of devices to use. Default is all devices available.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        help="Number of DataLoader worker processes.",
        default=1,
    )
    p.add_argument(
        "--config_dir",
        type=Path,
        help=f"Path to the directory of default config files. Default: %(default)s",
        default=config_dir,
    )
    p.add_argument(
        "--use_kernels",
        help="Whether to use kernels. One of 'auto', 'true', or 'false'. Default: %(default)s. "
        "If 'auto', will use kernels if the device capability is >= 8.",
        choices=["auto", "true", "false"],
        default="auto",
    )
    p.add_argument(
        "--moldir",
        type=str,
        help="Path to the moldir. Default: %(default)s",
        default=ARTIFACTS["moldir"][0],
    )

    # Design configuration options
    p = parser.add_argument_group("design")
    p.add_argument(
        "--num_designs",
        type=int,
        help="Number of total designs to generate. This commonly would be something like 10,000"
        "After generating 10,000 designs we then filter down to --budget many designs in the filter step",
        default=10000,
    )
    p.add_argument(
        "--diffusion_batch_size",
        type=int,
        default=None,
        help="Number of diffusion samples to generate per trunk run. If not specified, "
        "this defaults to 1 if --num-designs is less than 100, and 10 otherwise. Note that "
        "for design tasks that randomly sample the binder length (or use randomness in other "
        "ways), all designs generated in the same batch will share the same length. "
        "Having a large diffusion batch size compared to the total number of designs to "
        "generate will therefore not evenly sample the possible lengths.",
    )
    p.add_argument(
        "--design_checkpoints",
        type=str,
        nargs="+",
        help="Path to the boltzgen checkpoint(s). One or more checkpoints are supported. Just specifying an individual path here will work."
        "Each will be used for an equal fraction of the designs. By default, two checkpoints are used. "
        "Default: %(default)s",
        default=[
            ARTIFACTS["design-diverse"][0],
            ARTIFACTS["design-adherence"][0],
        ],
    )
    p.add_argument(
        "--step_scale",
        type=str,
        help="Fixed step scale to use (e.g. 1.8). Default is to use a schedule",
        default=None,
    )
    p.add_argument(
        "--noise_scale",
        type=str,
        help="Fixed noise scale to use (e.g. 0.98). Default is to use a schedule",
        default=None,
    )

    # Inverse folding configuration options
    p = parser.add_argument_group("inverse folding")
    p.add_argument(
        "--skip_inverse_folding",
        action="store_true",
        help="Skip inverse folding step",
    )
    p.add_argument(
        "--inverse_fold_num_sequences",
        type=int,
        help="Number of sequences per backbone to generate in the inverse fold step. Default: %(default)s",
        default=1,
    )
    p.add_argument(
        "--inverse_fold_checkpoint",
        type=str,
        help="Path or huggingface repo and filename for the inverse fold checkpoint. Default: %(default)s",
        default=ARTIFACTS["inverse-fold"][0],
    )
    p.add_argument(
        "--inverse_fold_avoid",
        type=str,
        default=None,
        help="Disallowed residues as a string of one letter amino acid codes, e.g. 'KEC'. "
        "This is implemented at the inverse fold step, so it only affects results if inverse folding is "
        "enabled. Default: none for protein design, 'C' for peptide and nanobody design. Pass an empty list if you want Cysteins to be generated if you are using a nanobody or peptide protocol",
    )
    p.add_argument(
        "--only_inverse_fold",
        action="store_true",
        help="Skip design step and only run inverse folding. Requires a fully specified structure.",
    )

    # Folding and affinity prediction configuration options
    p = parser.add_argument_group("folding and affinity prediction")
    p.add_argument(
        "--folding_checkpoint",
        type=str,
        help="Path to the folding checkpoint. Default: %(default)s",
        default=ARTIFACTS["folding"][0],
    )
    p.add_argument(
        "--affinity_checkpoint",
        type=str,
        help="Path to the affinity predictor checkpoint. Default: %(default)s",
        default=ARTIFACTS["affinity"][0],
    )

    # Filtering configuration options
    p = parser.add_argument_group("filtering")
    p.add_argument(
        "--budget",
        type=int,
        help="How many designs should be in the final diversity optimized set. This is used in the filtering step.",
        default=30,
    )
    p.add_argument(
        "--alpha",
        type=float,
        help="Trade-off for sequence diversity selection: 0.0=quality-only, 1.0=diversity-only. Default is "
        "0.01 (peptide-anything protocol) or 0.001 (other protocols).",
        default=None,
    )
    p.add_argument(
        "--filter_biased",
        choices=["true", "false"],
        help="Remove amino-acid composition outliers (default caps on ALA/GLY/GLU/LEU/VAL). Default: %(default)s.",
        default="true",
    )
    p.add_argument(
        "--metrics_override",
        nargs="+",
        help="Per-metric inverse-importance weights for ranking. "
        "Format: metric_name=weight (e.g., plip_hbonds_refolded=4 delta_sasa_refolded=2). "
        "A larger value down-weights that metric's rank. Use 'metric_name=none' to remove a metric.",
        default=None,
    )
    p.add_argument(
        "--additional_filters",
        nargs="+",
        help="Extra hard filters. Format: feature>threshold or feature<threshold "
        "(e.g., 'design_ALA>0.3' 'design_GLY<0.2'). Use '>' if higher is better, '<' if lower is better. "
        "Make sure to single-quote the strings so your shell doesn't get confused by < and > characters.",
        default=None,
    )
    p.add_argument(
        "--size_buckets",
        nargs="+",
        help="Optional constraint for maximum number of designs in size ranges. "
        "Format: min-max:count (e.g., 10-20:5 20-30:10 30-40:5).",
        default=None,
    )
    p.add_argument(
        "--refolding_rmsd_threshold",
        type=float,
        help="Threshold used for RMSD-based filters (lower is better).",
        default=None,
    )


def add_models_download_options(p: argparse.ArgumentParser) -> None:
    p = p.add_argument_group("model and data download options")
    p.add_argument(
        "--force_download",
        help="Force a (re)-download of models and data.",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--models_token",
        type=str,
        help="Secret token to use for our models hosting service (Hugging Face). Default: %(default)s",
        default=os.environ.get("HF_TOKEN"),
    )
    p.add_argument(
        "--cache",
        type=Path,
        help="Directory where downloaded models will be stored. Default: ~/.cache",
        default=None,
    )


def add_execute_core_arguments(p: argparse.ArgumentParser) -> None:
    p = p.add_argument_group("execution options")
    p.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing results across all steps. Generate only as many new designs are "
        "needed to achieve the specified total number of designs.",
    )
    p.add_argument(
        "--no_subprocess",
        dest="subprocess",
        action="store_false",
        help="Run each step in the main process. Will cause issues when devices >1.",
        default=True,
    )
    p.add_argument(
        "--steps",
        nargs="+",
        choices=step_names,
        help="Run only the specified pipeline steps (default: run all steps)",
    )


def build_run_parser(subparsers) -> argparse.ArgumentParser:
    run_parser = subparsers.add_parser(
        "run",
        description="Boltzgen binder design pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Run the binder design pipeline",
        epilog=__doc__,
    )
    group = run_parser.add_argument_group("design specification")
    group.add_argument(
        "design_spec",
        nargs="+",
        type=Path,
        help="Path(s) to design specification YAML file(s), or a directory containing prepared configs",
    )
    add_configure_arguments(run_parser, output_required=False)
    add_execute_core_arguments(run_parser)
    add_models_download_options(run_parser)
    return run_parser


def build_execute_parser(subparsers) -> argparse.ArgumentParser:
    execute_parser = subparsers.add_parser(
        "execute",
        description="Execute a pre-configured pipeline from a directory of config files",
        help="Run pipeline from pre-generated configuration files",
    )
    execute_parser.add_argument(
        "output",
        type=Path,
        help="Directory containing pre-configured pipeline files (generated by 'configure' command)",
    )
    add_execute_core_arguments(execute_parser)
    return execute_parser


def build_configure_parser(subparsers) -> argparse.ArgumentParser:
    configure_parser = subparsers.add_parser(
        "configure",
        description="Generate resolved pipeline configuration files without executing steps",
        help="Create configuration files for later execution",
    )
    group = configure_parser.add_argument_group("design specification")
    group.add_argument(
        "design_spec",
        nargs="+",
        type=Path,
        help="Path(s) to design specification YAML file(s)",
    )

    group = configure_parser.add_argument_group("steps to configure")
    group.add_argument(
        "--steps",
        nargs="+",
        choices=step_names,
        help="Configure only the specified pipeline steps (default: all steps)",
    )

    add_configure_arguments(configure_parser, output_required=True)
    add_models_download_options(configure_parser)

    configure_parser.set_defaults(subprocess=False)

    return configure_parser


def build_download_parser(subparsers) -> argparse.ArgumentParser:
    download_parser = subparsers.add_parser(
        "download",
        help="Download boltzgen model weights and supporting assets",
    )
    group = download_parser.add_argument_group(
        "artifacts to download (positional argument)"
    )
    group.add_argument(
        "artifacts",
        nargs="+",
        default=[],
        choices=sorted(ARTIFACTS.keys()) + ["all"],
        help="Subset of artifacts to download, or 'all' to download all artifacts.",
    )
    add_models_download_options(download_parser)
    return download_parser


def build_check_parser(subparsers) -> argparse.ArgumentParser:
    check_parser = subparsers.add_parser(
        "check",
        description="Check design specification files for validity and optionally output mmCIF",
        help="Validate design specification files",
    )
    check_parser.add_argument(
        "design_spec",
        nargs="+",
        type=Path,
        help="Path(s) to design specification YAML file(s)",
    )
    check_parser.add_argument(
        "--output",
        type=Path,
        help="Output directory to write mmCIF files (optional)",
    )
    check_parser.add_argument(
        "--moldir",
        type=str,
        help="Path to the moldir. Default: %(default)s",
        default=ARTIFACTS["moldir"][0],
    )

    add_models_download_options(check_parser)
    return check_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boltzgen",
        description="Boltzgen command line interface",
    )
    # Support: boltzgen -v / --version
    def get_package_version() -> str:
        try:
            return pkg_version("boltzgen")
        except PackageNotFoundError:
            return "unknown"
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"boltzgen {get_package_version()}",
        help="Print version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_run_parser(subparsers)
    build_configure_parser(subparsers)
    build_execute_parser(subparsers)
    build_download_parser(subparsers)
    build_check_parser(subparsers)
    return parser


#### Commands ####
def run_command(args: argparse.Namespace) -> None:
    """
    Run the **complete binder design pipeline** end-to-end by running:
        1. `configure_command(args)` – generates the per-step YAML configurations.
        2. `execute_command(args)` – launches each pipeline step based on those YAMLs.

    Typical CLI usage:
        $ boltzgen run path/to/design.yaml --output out_dir --protocol protein-anything
    """
    # Validate required arguments for running from design specs
    if not args.output:
        print("No output directory specified. Exiting.")
        return

    print("\n=== Configuring pipeline ===")
    configure_command(args)

    print("\n=== Executing pipeline ===")
    execute_command(args)


def download_command(args: argparse.Namespace) -> list[Path]:
    """
    Download **BoltzGen model checkpoints and the molecules directory** (hosted on HuggingFace).

    Parameters
    ----------
    args.artifacts : list[str]
        List of artifact keys to download, or `["all"]` to fetch all available assets.
    args.models_token : str
        Optional authentication token for private model access (default: env var `HF_TOKEN`).
    args.cache : Path
        Cache directory for storing downloaded artifacts.

    Returns
    -------
    list[Path]
        Local file paths of successfully downloaded artifacts.

    Usually this is executed by `boltzgen run ...` but it can be used like:
        $ boltzgen download all
        $ boltzgen download design-diverse inverse-fold
    """
    selections = sorted(set(args.artifacts))
    if "all" in selections:
        selections = list(ARTIFACTS.keys())

    download_paths: list[Path] = []
    for name in selections:
        artifact, repo_type = ARTIFACTS[name]
        resolved_path = get_artifact_path(args, artifact, repo_type=repo_type)
        print(f"Downloading {name} to {resolved_path}")
        download_paths.append(resolved_path)

    return download_paths


def configure_command(args: argparse.Namespace) -> None:
    """
    Generate **resolved per-step YAML configuration files** for the binder-design pipeline.

    This command constructs a `BinderDesignPipeline` from user and protocol parameters,
    validates design specifications, and writes out all configuration files required for running the pipeline. It does NOT run the pipeline.

    Outputs
    -------
    * `<output_dir>/config/<step>.yaml` — fully resolved config for each pipeline step.
    * `<output_dir>/steps.yaml` — manifest of steps and config file paths.

    This stage does **not execute** any model code; it only prepares the YAMLs used by
    `execute_command(...)`.

    Usually this is executed by `boltzgen run ...` but it can be used like:
        $ boltzgen configure path/to/design.yaml --output out_dir --protocol peptide-anything
    """
    moldir = get_artifact_path(args, args.moldir, repo_type="dataset")
    mols = load_canonicals(moldir=moldir)

    # Setup output directory
    output_dir = args.output
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {output_dir}")
    else:
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True)

    check_design_specs(args, moldir, mols)

    pipeline = BinderDesignPipeline(args, moldir)
    if args.steps is not None:
        pipeline.filter_steps(args.steps)

    pipeline.pretty_print()

    # Check that tasks can be instantiated
    for step in pipeline.steps:
        step.check()

    # Make the config subdir in output
    config_dir = output_dir / "config"
    if config_dir.exists():
        # Rename it to be previous-config-XXX
        counter = 1
        while config_dir.with_name(f"previous-config-{counter}").exists():
            counter += 1
        prev_config_dir = config_dir.with_name(f"previous-config-{counter}")
        print(f"Renaming existing config directory to {prev_config_dir}")
        config_dir.rename(prev_config_dir)
    config_dir.mkdir(parents=True)

    # Prepare step configurations and collect step info
    steps_info = []
    for step in pipeline.steps:
        config = step.get_config()
        config_filename = f"{step.name}.yaml"
        config_path = config_dir / config_filename
        with open(config_path, "w") as f:
            omegaconf.OmegaConf.save(config, f)

        # Add step info for steps.yaml (use relative path from output directory)
        steps_info.append(
            {"name": step.name, "config_file": str(config_path.relative_to(output_dir))}
        )

    # Write steps.yaml file
    steps_yaml_path = output_dir / "steps.yaml"
    steps_data = {"steps": steps_info}

    with open(steps_yaml_path, "w") as f:
        yaml.dump(steps_data, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration complete. Configs written to {config_dir}")
    print(f"Steps manifest written to {steps_yaml_path}")


def check_command(args: argparse.Namespace) -> None:
    """
    Validate **design specification YAML files** and write cif file for visualization.

    This command parses input design specs using `YamlDesignParser`, verifies structure
    integrity, highlights unresolved residues, and  outputs colored mmCIF visualizations.

    Parameters
    ----------
    args.design_spec : list[Path]
        Input YAML file(s) describing the binder design.
    args.output : Path, optional
        Directory to write mmCIF outputs (optional).
    args.moldir : str
        Path or Hugging Face reference to the molecule dataset.

    Typical CLI usage:
        $ boltzgen check path/to/design.yaml --output checked/

    This function does **not** execute the pipeline — it only validates inputs.
    """
    moldir = get_artifact_path(args, args.moldir, repo_type="dataset")
    mols = load_canonicals(moldir=moldir)

    if args.output:
        output_dir = args.output
        if output_dir.exists():
            if not output_dir.is_dir():
                raise ValueError(
                    f"Output path exists and is not a directory: {output_dir}"
                )
        else:
            print(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True)

    check_design_specs(args, moldir, mols)


def execute_command(args: argparse.Namespace) -> None:
    """
    Execute a **pre-configured binder design pipeline** from a directory of YAML files.

    Reads the `steps.yaml` manifest written by `configure_command(...)` and executes
    each step sequentially, either in subprocesses or directly in-process.

    Options
    --------
    * `--no_subprocess` : Run in the main process instead of spawning Python subprocesses.
    * `--steps` : Restrict to specific pipeline steps.
    * `--reuse` : Skip recomputation for existing results.

    Expected directory structure (produced by `configure_command(...)`):
        output_dir/
            ├── config/
            │     ├── design.yaml
            │     ├── folding.yaml
            │     └── ...
            └── steps.yaml

    Usually this is executed by `boltzgen run ...` but it can be used like:
        $ boltzgen execute --output out_dir
    """
    config_dir = args.output

    if not config_dir.exists() or not config_dir.is_dir():
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

    # Look for steps.yaml file
    steps_yaml_path = config_dir / "steps.yaml"
    if not steps_yaml_path.exists():
        raise FileNotFoundError(
            f"Steps manifest not found: {steps_yaml_path}. Run 'boltzgen configure' first."
        )

    # Load steps from steps.yaml
    with open(steps_yaml_path, "r") as f:
        steps_data = yaml.safe_load(f)

    if not isinstance(steps_data, dict) or "steps" not in steps_data:
        raise ValueError(f"Invalid steps.yaml format in {steps_yaml_path}")

    # Filter steps if specific steps are requested
    enabled_steps = set(args.steps) if args.steps else None
    resolved_steps: List[Tuple[str, Path]] = []

    for step_info in steps_data["steps"]:
        step_name = step_info["name"]
        config_filename = step_info["config_file"]

        # Skip if this step is not in the enabled steps list
        if enabled_steps is not None and step_name not in enabled_steps:
            continue

        # Build full path to config file
        config_path = config_dir / config_filename
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        resolved_steps.append((step_name, config_path))

    if not resolved_steps:
        if enabled_steps:
            print(f"No matching steps found for: {', '.join(enabled_steps)}")
        else:
            print(f"No steps found in {steps_yaml_path}")
        return

    total_steps = len(resolved_steps)
    for index, (step_name, config_path) in enumerate(resolved_steps, start=1):
        print("**************************************************")
        print(f"Pipeline step {index} of {total_steps}: {step_name}")

        os.environ["BOLTZGEN_PIPELINE_PROGRESS"] = f"Step {index}/{total_steps}"
        os.environ["BOLTZGEN_PIPELINE_STEP"] = step_name

        start = time.time()
        if args.subprocess:
            command = [
                "python",
                str(main_script),
                str(config_path),
            ]
            print(f"Running command: {shlex.join(command)}")
            subprocess.check_call(command)
        else:
            config = omegaconf.OmegaConf.load(config_path)
            task = hydra.utils.instantiate(config)
            if not isinstance(task, Task):
                raise TypeError("Config must be an instance of Task.")
            task.run(config)

        elapsed = time.time() - start
        print(f"✓ Step {step_name} completed successfully in {elapsed:.1f}s")

    if "BOLTZGEN_PIPELINE_PROGRESS" in os.environ:
        del os.environ["BOLTZGEN_PIPELINE_PROGRESS"]
    if "BOLTZGEN_PIPELINE_STEP" in os.environ:
        del os.environ["BOLTZGEN_PIPELINE_STEP"]


#### Pipeline implementation ####
@dataclass
class PipelineStep:
    name: str
    config_path: str
    args: List[str]

    def check(self):
        if self.name not in step_names:
            raise ValueError(
                f"Invalid step name: {self.name}. Available steps: {step_names}"
            )
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def get_config(self) -> omegaconf.DictConfig:
        config = omegaconf.OmegaConf.create({})
        config_file = omegaconf.OmegaConf.load(self.config_path)
        config = omegaconf.OmegaConf.merge(config, config_file)
        args_cfg = omegaconf.OmegaConf.from_dotlist(self.args)
        config = omegaconf.OmegaConf.merge(config, args_cfg)
        return config


class BinderDesignPipeline:
    """
    The class itself does **not** execute work or write files; instead it builds a
    structured list of `PipelineStep` objects (name + config path + dot-list args)
    corresponding to the selected protocol and user flags. Those steps are then:

    1) **Writtent to yaml files** when `configure_command(...)` is executed
       - For each `PipelineStep`, the resolved Hydra config is written to
         `OUTPUT/config/<step>.yaml`.
       - A manifest `OUTPUT/steps.yaml` is also written, listing the enabled steps
         and their config files in execution order.

    2) **Executed from YAML** when `execute_command(...)` is executed
       - Each step is launched **as a subprocess** (`python main.py <config.yaml>`)
         unless `--no_subprocess` is set (not the default).
       - If `--no_subprocess` is specified, the config is instantiated in-process
         and the `Task.run(...)` method is called directly.

    The actual code that is exectued in each pipeline step is found in `main.py` which a wrapper for running the .run() function of our `Task` class.
    If you run the pipeline (for example via `boltzgen run design_spec.yaml ...`) then this function reads the yaml files of the individual pipeline steps and executes the pipeline steps.

    The possible tasks are:
        - Predict src/boltzgen/task/predict/predict.py (GPU: Running BoltzGen diffusion, inverse folding, refolding, designfolding, or affinity prediction)
        - Analyze src/boltzgen/task/analyze/analyze.py (CPU: Compute CPU Metrics and aggregate metrics from GPU steps)
        - Filter src/boltzgen/task/filter/filter.py (CPU: Very fast (20s) computes ranking and writes final output files)

    Usage
    -----
    The class is constructed inside `configure_command(...)` and `run_command(...)`,
    not usually called directly by end users.

    Attributes
    ----------
    steps : list[PipelineStep]
        The ordered list of configured steps. Each entry carries the
        `name`, a `config_path` (base template under `--config_dir`), and a
        list of Hydra **dot-list** overrides that will be merged and written to YAML
        by `configure_command(...)`.

    Methods
    -------
    filter_steps(enabled_steps: list[str]) -> None
        Restrict the pipeline to a subset of steps (used by `--steps ...`).

    Notes
    -----
    * Subprocess execution sets environment variables
      `BOLTZGEN_PIPELINE_PROGRESS` and `BOLTZGEN_PIPELINE_STEP` for downstream
      logging/UX.
    * The pipeline distinguishes *where* configurations are **written** (in
      `configure_command`) from **how** they are **executed** (in `execute_command`).
      This separation makes it easy to tweak or re-run any step by editing the
      generated YAMLs.
    """

    def __init__(self, args: argparse.Namespace, moldir: Path):
        protocol = args.protocol
        if protocol not in protocol_configs:
            raise ValueError(
                f"Invalid protocol: {protocol}. Valid protocols: {list(protocol_configs.keys())}"
            )

        # Handle use_kernels argument
        device_capability = torch.cuda.get_device_capability()
        use_kernels = None
        if args.use_kernels == "auto":
            use_kernels = device_capability[0] >= 8
        elif args.use_kernels == "true":
            use_kernels = True
        elif args.use_kernels == "false":
            use_kernels = False
        else:
            raise ValueError(f"Invalid use_kernels value: {args.use_kernels}")
        print(f"Using kernels: {use_kernels} [device capability: {device_capability}]")

        protocol_config = protocol_configs[protocol]
        print(f"Config overrides for protocol {protocol}: {protocol_config}")

        # Set protocol-specific and user-specified step specific args
        config_args_by_step = parse_config_args(
            protocol_config, args.config, step_names
        )

        devices = (
            args.devices if args.devices is not None else torch.cuda.device_count()
        )
        print(f"Using {devices} devices")

        self.steps = []

        # Design generation
        output_dir = args.output / "intermediate_designs"
        print(f"Raw designs will be saved to: {output_dir}")
        diffusion_batch_size = args.diffusion_batch_size
        if diffusion_batch_size is None:
            diffusion_batch_size = 1 if args.num_designs < 100 else 10
        num_batches = math.ceil(args.num_designs / diffusion_batch_size)
        print(f"Using diffusion batch size: {diffusion_batch_size}")
        print(f"Number of diffusion batches: {num_batches}")

        # Configure design checkpoint arguments.
        checkpoint_args = []
        # First checkpoint is specified differently than others.
        first_checkpoint_path = get_artifact_path(args, args.design_checkpoints[0])
        checkpoint_args.append(f"checkpoint={first_checkpoint_path}")
        if len(args.design_checkpoints) > 1:
            fraction_per_checkpoint = 1.0 / len(args.design_checkpoints)
            checkpoint_args.append(
                f"override.checkpoints.first_checkpoint_num_samples={fraction_per_checkpoint}"
            )
            checkpoint_args.append(
                f"override.checkpoints.checkpoint_list=["
                + ",".join(
                    f"{{'checkpoint': {{'num_samples': {fraction_per_checkpoint}, 'path': '{get_artifact_path(args, checkpoint)}'}}}}"
                    for checkpoint in args.design_checkpoints[1:]
                )
                + "]"
            )
        design_step_and_noise_scale_args = []
        if args.step_scale is not None:
            design_step_and_noise_scale_args.append(
                f"override.diffusion_process_args.step_scale={args.step_scale}"
            )
            # Also disable the schedule when applying a fixed step scale
            design_step_and_noise_scale_args.append(
                f"override.step_scale_schedule=null"
            )
        if args.noise_scale is not None:
            design_step_and_noise_scale_args.append(
                f"override.diffusion_process_args.noise_scale={args.noise_scale}"
            )
            design_step_and_noise_scale_args.append(
                f"override.noise_scale_schedule=null"
            )

        if args.only_inverse_fold:
            # Designs from inverse folding
            self.steps.append(
                PipelineStep(
                    name="inverse_folding",
                    config_path=args.config_dir / "inverse_fold_only.yaml",
                    args=[
                        f"output={output_dir}",
                        f"data.cfg.yaml_path=[{', '.join(str(s) for s in args.design_spec)}]",
                        f"trainer.devices={devices}",
                        f"data.cfg.multiplicity={getattr(args, 'inverse_fold_num_sequences', 10)}",
                        f"override.use_kernels={use_kernels}",
                        f"checkpoint={get_artifact_path(args, args.inverse_fold_checkpoint)}",
                        f"data.cfg.moldir={moldir}",
                    ]
                    + config_args_by_step.get("inverse_folding", []),
                )
            )
        else:
            # Designs from diffusion model
            self.steps.append(
                PipelineStep(
                    name="design",
                    config_path=args.config_dir / "design.yaml",
                    args=[
                        f"output={output_dir}",
                        f"data.cfg.yaml_path=[{', '.join(str(s) for s in args.design_spec)}]",
                        f"trainer.devices={devices}",
                        f"data.num_workers={args.num_workers}",
                        f"data.cfg.skip_existing={args.reuse}",
                        f"data.cfg.multiplicity={num_batches}",
                        f"diffusion_samples={diffusion_batch_size}",
                        f"override.use_kernels={use_kernels}",
                        f"data.cfg.moldir={moldir}",
                    ]
                    + design_step_and_noise_scale_args
                    + checkpoint_args
                    + config_args_by_step["design"],
                )
            )

            # Inverse folding of diffusion-generated backbones.
            if not args.skip_inverse_folding:
                exclude_residues = []
                inverse_fold_avoid = (
                    args.inverse_fold_avoid
                    if args.inverse_fold_avoid is not None
                    else (
                        "C"
                        if protocol in ["peptide-anything", "nanobody-anything"]
                        else ""
                    )
                )

                for one_letter_code in inverse_fold_avoid:
                    exclude_residues.append(const.prot_letter_to_token[one_letter_code])

                if len(exclude_residues) > 0:
                    print(
                        f"Inverse fold will avoid the following residues: {exclude_residues}"
                    )

                input_dir = output_dir
                output_dir = args.output / "intermediate_designs_inverse_folded"
                print(f"Inverse-folded designs will be saved to: {output_dir}")
                self.steps.append(
                    PipelineStep(
                        name="inverse_folding",
                        config_path=args.config_dir / "inverse_fold.yaml",
                        args=[
                            f"output={output_dir}",
                            f"data.design_dir={input_dir}",
                            f"data.cfg.multiplicity={args.inverse_fold_num_sequences}",
                            f"data.cfg.num_workers={args.num_workers}",
                            f"data.skip_existing={args.reuse}",
                            f"data.skip_existing_kind=inverse_fold",
                            f"override.use_kernels={use_kernels}",
                            f"checkpoint={get_artifact_path(args, args.inverse_fold_checkpoint)}",
                            f"data.cfg.moldir={moldir}",
                            f"trainer.devices={devices}",
                            f"override.inverse_fold_args.inverse_fold_restriction=[{', '.join(exclude_residues)}]",
                        ]
                        + config_args_by_step["inverse_folding"],
                    )
                )

        # Folding
        input_dir = output_dir
        self.steps.append(
            PipelineStep(
                name="folding",
                config_path=args.config_dir / "fold.yaml",
                args=[
                    f"output={output_dir}",
                    f"data.design_dir={input_dir}",
                    f"trainer.devices={devices}",
                    f"data.cfg.num_workers={args.num_workers}",
                    f"data.skip_existing={args.reuse}",
                    f"data.skip_existing_kind=folded",
                    f"override.use_kernels={use_kernels}",
                    f"checkpoint={get_artifact_path(args, args.folding_checkpoint)}",
                    f"data.cfg.moldir={moldir}",
                ]
                + config_args_by_step["folding"],
            )
        )

        # Design folding
        input_dir = output_dir
        do_design_folding = protocol in ["protein-anything", "protein-small_molecule"]
        if do_design_folding:
            self.steps.append(
                PipelineStep(
                    name="design_folding",
                    config_path=args.config_dir / "fold.yaml",
                    args=[
                        f"output={output_dir}",
                        f"data.design_dir={input_dir}",
                        f"trainer.devices={devices}",
                        f"data.cfg.num_workers={args.num_workers}",
                        f"data.skip_existing={args.reuse}",
                        f"data.skip_existing_kind=design_folded",
                        f"override.use_kernels={use_kernels}",
                        f"checkpoint={get_artifact_path(args, args.folding_checkpoint)}",
                        f"data.cfg.moldir={moldir}",
                        f"writer.designfolding=True",
                        f"data.cfg.return_designfolding=True",
                    ]
                    + config_args_by_step["design_folding"],
                )
            )

        # Affinity
        use_affinity = protocol in ["protein-small_molecule"]
        if use_affinity:
            self.steps.append(
                PipelineStep(
                    name="affinity",
                    config_path=args.config_dir / "affinity.yaml",
                    args=[
                        f"output={output_dir}",
                        f"data.design_dir={input_dir}",
                        f"trainer.devices={devices}",
                        f"data.cfg.num_workers={args.num_workers}",
                        f"data.skip_existing={args.reuse}",
                        f"data.skip_existing_kind=affinity",
                        f"override.use_kernels={use_kernels}",
                        f"checkpoint={get_artifact_path(args, args.affinity_checkpoint)}",
                        f"data.cfg.moldir={moldir}",
                    ]
                    + config_args_by_step["affinity"],
                )
            )

        # Analysis
        self.steps.append(
            PipelineStep(
                name="analysis",
                config_path=args.config_dir / "analysis.yaml",
                args=[
                    f"design_dir={input_dir}",
                    f"data.skip_existing={args.reuse}",
                    f"data.skip_existing_kind=analyzed",
                    f"data.cfg.moldir={moldir}",
                    f"designfolding_metrics={do_design_folding}",
                    f"delta_sasa_original={args.skip_inverse_folding}",
                    f"noncovalents_original={args.skip_inverse_folding}",
                    f"allatom_fold_metrics={args.skip_inverse_folding}",
                ]
                + config_args_by_step["analysis"],
            )
        )

        # Filtering
        output_dir = args.output / "final_ranked_designs"
        print(f"Final ranked designs will be saved to: {output_dir}")

        # Build filter arguments
        filter_args = [
            f"design_dir={input_dir}",
            f"outdir={args.output}",
            f"from_inverse_folded={not args.skip_inverse_folding}",
            f"use_affinity={use_affinity}",
            f"filter_designfolding={do_design_folding}",
            f"budget={args.budget}",
        ]

        # Add optional filtering arguments
        if args.alpha is not None:
            filter_args.append(f"alpha={args.alpha}")
        if args.filter_biased is not None:
            filter_args.append(f"filter_biased={args.filter_biased}")
        if args.refolding_rmsd_threshold is not None:
            filter_args.append(
                f"refolding_rmsd_threshold={args.refolding_rmsd_threshold}"
            )
        if args.metrics_override is not None:
            parsed_metrics = parse_metrics_override(args.metrics_override)
            print(f"Filtering metrics override: {parsed_metrics}")
            filter_args.append(f"metrics_override={parsed_metrics}")
        if args.additional_filters is not None:
            parsed_filters = parse_additional_filters(args.additional_filters)
            print(f"Filtering additional filters: {parsed_filters}")
            filter_args.append(f"additional_filters={parsed_filters}")
        if args.size_buckets is not None:
            parsed_size_buckets = parse_size_buckets(args.size_buckets)
            print(f"Filtering size buckets: {parsed_size_buckets}")
            filter_args.append(f"size_buckets={parsed_size_buckets}")

        self.steps.append(
            PipelineStep(
                name="filtering",
                config_path=args.config_dir / "filtering.yaml",
                args=filter_args + config_args_by_step["filtering"],
            )
        )

    def filter_steps(self, enabled_steps: List[str]):
        self.steps = [s for s in self.steps if s.name in enabled_steps]

    def pretty_print(self):
        for i, step in enumerate(self.steps):
            print(f"[{i + 1}] {step.name:25s}")


### Misc utiltiies ###
def check_design_specs(args: argparse.Namespace, moldir: Path, mols: Dict[str, Any]):
    last_banner = ""
    for design_spec in args.design_spec:
        banner = f"************** Checking design spec: {design_spec} **************"
        last_banner = banner
        print(banner)
        check_design_spec(args, moldir, design_spec, mols)
    if last_banner:
        print("*" * len(last_banner))


def check_design_spec(
    args: argparse.Namespace, moldir: Path, design_spec: Path, mols: Dict[str, Any]
):
    """
    Validate a single design specification, color/annotate its structure, and
    write an mmCIF visualization.

    This function parses a YAML design spec with `YamlDesignParser`, applies visual
    annotations to the parsed structure (via B-factors and per-residue color features),
    reports unresolved residues/atoms, and writes a colored mmCIF file if `--output`
    was provided on the CLI.

    - Writes `<output>/<design_spec.stem>.cif` if `args.output` is set.

        * B-factor encodes design/binding (100 for designed, +80 if binding).
        * `design_color_features`: 1.0 for binding residues, 0.8 otherwise.

    Examples
    --------
    From the CLI (via the `check` subcommand):

        $ boltzgen check path/to/design.yaml --output checked/

    From Python:

        moldir = PATH TO MOLDIR
        mols = load_canonicals(moldir=moldir)
        check_design_spec(args, moldir, Path("design.yaml"), mols)
    """
    parser = YamlDesignParser(moldir)
    parsed = parser.parse_yaml(design_spec, mols, moldir)
    structure = parsed.structure
    design_info = parsed.design_info
    design_color_features = np.ones_like(design_info.res_binding_type) * 0.8
    design_color_features[design_info.res_binding_type.astype(bool)] = 1.0
    extract_mask = np.zeros(len(structure.residues), dtype=bool)
    for i, residue in enumerate(structure.residues):
        structure.atoms["bfactor"][
            residue["atom_idx"] : residue["atom_idx"] + residue["atom_num"]
        ] = 100 * design_info.res_design_mask[i] + 80 * design_info.res_binding_type[i]

        atom_positions = structure.atoms["coords"][
            residue["atom_idx"] : residue["atom_idx"] + residue["atom_num"]
        ]
        zero_position_count = ((atom_positions**2).sum(axis=1) < 1e-6).sum()
        if not zero_position_count == len(atom_positions):
            extract_mask[i] = True

    structure_write = structure
    if extract_mask.sum() > 0:
        structure_write = structure.extract_residues(structure, extract_mask)
        design_color_features = design_color_features[extract_mask]

    mmcif = to_mmcif(
        structure_write,
        design_coloring=True,
        color_features=design_color_features,
    )

    # Check for unresolved residues and atoms to log a warning
    unresolved_residues = (~structure.residues["is_present"]).nonzero()[0]
    unresolved_atoms = (~structure.atoms["is_present"]).nonzero()[0]

    print(f"Total designed residues: {design_info.res_design_mask.sum()}")

    if len(unresolved_residues) > 0 or len(unresolved_atoms) > 0:
        atom_to_residue = {}
        for residue_idx, residue in enumerate(structure.residues):
            start = residue["atom_idx"]
            end = start + residue["atom_num"]
            for i in range(start, end):
                atom_to_residue[i] = residue_idx

        residue_to_chain = {}
        residue_in_chain = {}
        for chain in structure.chains:
            start = chain["res_idx"]
            end = start + chain["res_num"]
            j = 1
            for i in range(start, end):
                residue_to_chain[i] = chain["name"]
                residue_in_chain[i] = j
                j += 1

        atom_to_chain = {}
        for chain in structure.chains:
            start = chain["atom_idx"]
            end = start + chain["atom_num"]
            for i in range(start, end):
                atom_to_chain[i] = chain["name"]

        msg = f"There are {len(unresolved_residues)} unresolved residues and {len(unresolved_atoms)} unresolved atoms in the target."
        print(msg)

    if args.output is not None:
        output_path = args.output / (design_spec.stem + ".cif")
    else:
        output_path = design_spec.stem + ".cif"
    with open(output_path, "w") as f:
        f.write(mmcif)
    print(f"Design specification visualization is written to {str(output_path)}")


def get_artifact_path(
    args, artifact: str, repo_type: str = "model", verbose: bool = True
) -> Path:
    """Get the local path to an artifact that is either a local file or hosted on Hugging Face."""
    if artifact.startswith("huggingface:"):
        try:
            _, repo_id, filename = artifact.split(":")
        except ValueError:
            raise ValueError(
                f"Invalid artifact: {artifact}. Expected format: huggingface:<repo_id>:<filename>"
            )
        result = huggingface_hub.hf_hub_download(
            repo_id,
            filename,
            repo_type=repo_type,
            library_name="boltzgen",
            force_download=args.force_download,
            token=args.models_token,
            cache_dir=args.cache,
        )
        result = Path(result)
    else:
        result = Path(artifact)
    if not result.exists():
        raise FileNotFoundError(f"Model not found: {result}")
    if verbose:
        print(f"Using {repo_type} artifact: {result}")
    return result


def parse_config_args(base_config, config_args, valid_step_names):
    config_args_by_step = collections.defaultdict(list)
    config_args_by_step.update(base_config)
    if config_args:
        for config in config_args:
            if len(config) < 2:
                raise ValueError(
                    f"Invalid config: {config}. Expected format: <step_name> <arg1>=<value1> <arg2>=<value2> ..."
                )
            step_name = config[0]
            if step_name not in valid_step_names:
                raise ValueError(
                    f"Invalid step name: {step_name}. Available steps: {valid_step_names}"
                )
            key_value_pairs = config[1:]
            config_args_by_step[step_name].extend(key_value_pairs)
    return config_args_by_step


### Filtering argument parsing functions ####
def parse_metrics_override(value_list):
    """Parse metrics_override from key=value pairs."""
    if not value_list:
        return None

    metrics_override = {}
    for item in value_list:
        if "=" in item:
            key, value = item.split("=", 1)
            if value == "" or value.lower() == "none":
                metrics_override[key] = None  # Remove metric
            else:
                try:
                    metrics_override[key] = float(value)
                except ValueError:
                    raise ValueError(
                        f"Invalid weight value for metric '{key}': '{value}'. Must be a number."
                    )
        else:
            raise ValueError(
                f"Invalid metrics_override format: '{item}'. Use 'metric_name=weight' format."
            )
    return metrics_override


def parse_additional_filters(value_list):
    """Parse additional_filters from feature>threshold or feature<threshold format."""
    if not value_list:
        return None

    additional_filters = []
    for item in value_list:
        if ">" in item:
            feature, threshold_str = item.split(">", 1)
            try:
                threshold = float(threshold_str)
                additional_filters.append(
                    {
                        "feature": feature,
                        "threshold": threshold,
                        "lower_is_better": False,  # > means higher is better
                    }
                )
            except ValueError:
                raise ValueError(
                    f"Invalid threshold value: '{threshold_str}'. Must be a number."
                )
        elif "<" in item:
            feature, threshold_str = item.split("<", 1)
            try:
                threshold = float(threshold_str)
                additional_filters.append(
                    {
                        "feature": feature,
                        "threshold": threshold,
                        "lower_is_better": True,  # < means lower is better
                    }
                )
            except ValueError:
                raise ValueError(
                    f"Invalid threshold value: '{threshold_str}'. Must be a number."
                )
        else:
            raise ValueError(
                f"Invalid additional_filters format: '{item}'. Use 'feature>threshold' or 'feature<threshold' format."
            )
    return additional_filters


def parse_size_buckets(value_list):
    """Parse size_buckets from min-max:count format."""
    if not value_list:
        return None

    size_buckets = []
    for item in value_list:
        if ":" in item and "-" in item:
            range_part, count_str = item.split(":", 1)
            if "-" in range_part:
                min_str, max_str = range_part.split("-", 1)
                try:
                    min_size = int(min_str)
                    max_size = int(max_str)
                    count = int(count_str)
                    size_buckets.append(
                        {"num_designs": count, "min": min_size, "max": max_size}
                    )
                except ValueError as e:
                    if "invalid literal" in str(e):
                        raise ValueError(
                            f"Invalid size_buckets format: '{item}'. All values must be integers. Use 'min-max:count' format."
                        )
                    else:
                        raise e
            else:
                raise ValueError(
                    f"Invalid size_buckets format: '{item}'. Use 'min-max:count' format."
                )
        else:
            raise ValueError(
                f"Invalid size_buckets format: '{item}'. Use 'min-max:count' format."
            )
    return size_buckets


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_command(args)
    elif args.command == "configure":
        configure_command(args)
    elif args.command == "execute":
        execute_command(args)
    elif args.command == "download":
        download_command(args)
    elif args.command == "check":
        check_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
