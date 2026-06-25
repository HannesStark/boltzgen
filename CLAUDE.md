# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BoltzGen is an all-atom protein design pipeline that uses diffusion models for de novo binder design. The system generates protein backbones, designs sequences via inverse folding, refolds structures using Boltz-2, and filters/ranks candidates based on structural and biochemical metrics.

## Key Commands

### Environment Setup
```bash
# Create conda environment (Python >=3.10, recommended 3.11 or 3.12)
conda create -n bg python=3.12
conda activate bg

# Install from source (for development)
pip install -e .

# Install with dev dependencies (includes wandb, ruff, pytest)
pip install -e .[dev]

# Install with CUDA kernel acceleration (requires CUDA 12)
pip install -e .[cuequivariance]
```

### Running BoltzGen Pipeline
```bash
# Full pipeline: design → inverse_fold → folding → analysis → filtering
boltzgen run example/vanilla_protein/1g13prot.yaml \
  --output workbench/test_run \
  --protocol protein-anything \
  --num_designs 10 \
  --budget 2

# Check design specification before running
boltzgen check example/vanilla_protein/1g13prot.yaml --output checked/

# Run specific pipeline steps only
boltzgen run design.yaml \
  --output workbench/test \
  --steps design inverse_folding \
  --num_designs 50

# Rerun filtering with different parameters (fast, ~15 seconds)
boltzgen run design.yaml \
  --output workbench/test \
  --steps filtering \
  --refolding_rmsd_threshold 3.0 \
  --alpha 0.2

# Resume interrupted run
boltzgen run design.yaml --output workbench/test --reuse
```

### Separate Configure and Execute
```bash
# Generate config files without running
boltzgen configure design.yaml \
  --output workbench/test \
  --protocol peptide-anything \
  --num_designs 100

# Edit config files in workbench/test/config/ if needed, then execute
boltzgen execute workbench/test
```

### Download Models
```bash
# Download all models and data (not usually needed - happens automatically)
boltzgen download all

# Download specific components
boltzgen download design-diverse inverse-fold folding
```

### Training Models
```bash
# Training requires dev installation and additional data
pip install -e .[dev]

# Download training data first (see README Training section)
# Then train models:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/boltzgen/resources/main.py \
  src/boltzgen/resources/config/train/boltzgen_small.yaml \
  name=boltzgen_small
```

### Linting
```bash
# Run ruff linter
ruff check src/
ruff format src/
```

### Docker
```bash
# Build image
docker build -t boltzgen .

# Run with GPU support
docker run --rm --gpus all \
  -v "$(realpath workdir)":/workdir \
  -v "$(realpath cache)":/cache \
  -v "$(realpath example)":/example \
  boltzgen run /example/vanilla_protein/1g13prot.yaml \
  --output /workdir/test \
  --protocol protein-anything \
  --num_designs 2
```

## Architecture

### Pipeline Flow
The BoltzGen pipeline orchestrates multiple GPU and CPU tasks through a YAML-based configuration system:

1. **Configure Phase** (`boltzgen configure` or first step of `boltzgen run`)
   - Parses design specification YAML files
   - Generates per-step configuration files in `output/config/`
   - Creates `output/steps.yaml` manifest
   - Validates design specs and downloads required models

2. **Execute Phase** (`boltzgen execute` or second step of `boltzgen run`)
   - Reads `steps.yaml` manifest
   - Launches each pipeline step as subprocess (or in-process with `--no_subprocess`)
   - Each step runs via `src/boltzgen/resources/main.py` which instantiates and runs a Task

### Core Components

**CLI Entry Point** (`src/boltzgen/cli/boltzgen.py`)
- Main entry point for all `boltzgen` commands
- Builds argparse parsers for: run, configure, execute, download, check
- Constructs `BinderDesignPipeline` which generates step configurations
- Handles protocol-specific overrides (protein-anything, peptide-anything, protein-small_molecule, nanobody-anything)

**Task Execution** (`src/boltzgen/resources/main.py`)
- Wrapper that loads Hydra configs and instantiates Task subclasses
- Invoked as subprocess for each pipeline step
- The `_target_` field in YAML configs specifies which Task class to run

**Task Implementations**
- `src/boltzgen/task/predict/predict.py` - GPU tasks: design, inverse_folding, folding, design_folding, affinity
- `src/boltzgen/task/analyze/analyze.py` - CPU task: compute structural and biochemical metrics
- `src/boltzgen/task/filter/filter.py` - CPU task: rank and filter designs based on metrics
- `src/boltzgen/task/train/train.py` - GPU task: train BoltzGen or inverse folding models

**Model Layers** (`src/boltzgen/model/`)
- `models/` - Main model architectures
- `modules/` - Reusable network modules
- `layers/` - Atomic layer implementations (including triangular attention)
- `loss/` - Loss functions for training
- `optim/` - Optimizers and learning rate schedulers

**Data Processing** (`src/boltzgen/data/`)
- `parse/` - YAML design spec parser (`schema.py`)
- `write/` - mmCIF and other output writers
- `feature/` - Feature extraction and preprocessing
- `filter/` - Data filtering (static and dynamic)
- `crop/`, `sample/`, `select/`, `tokenize/`, `template/` - Data pipeline components

**Molecular Dynamics** (`src/boltzgen/molecular_dynamics/`)
- System preparation, membrane simulations, free energy calculations
- Partition coefficient predictions, structural property analysis
- Newly added module for MD-based analysis

### Design Specification System

Design specs are YAML files with two main sections:

**entities**: Define components
- `protein`: Designed or fixed protein sequences (e.g., `sequence: 80..140` or `sequence: AAVTTTTPPP`)
- `ligand`: Small molecules via CCD codes or SMILES
- `file`: Import structures from PDB/mmCIF files with fine-grained control over:
  - `include`/`exclude`: Which chains/residues to use
  - `binding_types`: Where the design should bind
  - `structure_groups`: Visibility levels for relative positioning
  - `design`: Which imported residues should be redesigned
  - `secondary_structure`: Structural constraints (helix/sheet/loop)
  - `design_insertions`: Insert new designed residues

**constraints**: Structural rules
- `bond`: Covalent bonds between atoms (e.g., disulfide bonds, peptide staples)
- `total_len`: Length constraints for the entire system

**Important indexing**: All residue indices start at 1 and use mmCIF `label_asym_id` (NOT `auth_asym_id`). Verify using https://molstar.org/viewer/.

### Pipeline Steps

1. **design** - Generate backbone structures using BoltzGen diffusion model
2. **inverse_folding** - Design sequences for generated backbones using inverse folding model
3. **folding** - Refold designed binders with their targets using Boltz-2
4. **design_folding** - Refold designed binders alone (protein-anything and protein-small_molecule only)
5. **affinity** - Predict binding affinity (protein-small_molecule only)
6. **analysis** - Compute structural metrics (RMSD, SASA, H-bonds, etc.) and aggregate results
7. **filtering** - Rank designs by quality metrics, apply hard filters, optimize for diversity

### Protocols

Four protocols provide sensible defaults for different design tasks:

- **protein-anything**: Design proteins to bind proteins/peptides (includes design_folding)
- **peptide-anything**: Design peptides/cyclic peptides (no Cys in inverse folding, no design_folding, filter Cys, different alpha)
- **protein-small_molecule**: Design proteins to bind small molecules (includes affinity prediction)
- **nanobody-anything**: Design single-domain antibodies (no Cys, no design_folding, filter Cys)

Protocol settings can be overridden via `--config <step> <arg>=<value>`.

### Key Design Patterns

**Hydra Configuration System**
- All pipeline steps use Hydra for configuration management
- Base configs in `src/boltzgen/resources/config/`
- Configs are merged: base → protocol overrides → user `--config` overrides → CLI args
- The `_target_` field specifies the Python class to instantiate

**Subprocess vs In-Process Execution**
- Default: each step runs as subprocess (`python main.py config.yaml`)
- Alternative: `--no_subprocess` runs steps in main process
- Subprocess isolation prevents GPU memory issues with multiple devices

**Model Artifact Management**
- Models hosted on HuggingFace with `huggingface:repo:file` notation
- `get_artifact_path()` handles download/caching to `~/.cache` (or `$HF_HOME`)
- `--force_download` to re-download, `--local_files_only` for offline mode

**Reuse and Resumption**
- `--reuse` flag skips existing outputs, generates only missing designs
- Each step checks `skip_existing_kind` (e.g., `inverse_fold`, `folded`, `analyzed`)
- No progress lost if pipeline interrupted

## Development Workflow

### Making Changes to Pipeline Steps

1. **Modify Task Implementation**: Edit files in `src/boltzgen/task/`
2. **Update Config Schema**: If adding parameters, update corresponding YAML in `src/boltzgen/resources/config/`
3. **Test Locally**: Run with small `--num_designs` first
4. **No Need to Reinstall**: Changes to `.py` files are picked up immediately with `pip install -e .`

### Adding New Filtering Metrics

1. Compute metric in `src/boltzgen/task/analyze/analyze.py`
2. Add to output CSV in `analyze_utils.py`
3. Reference in filtering config or use `--metrics_override` / `--additional_filters`

### Training New Models

1. Download training data (see README "Training BoltzGen models")
2. Adjust paths in `src/boltzgen/resources/config/train/*.yaml`
3. Configure wandb for experiment tracking
4. Launch training with appropriate GPU count

### Design Spec YAML Creation

1. Start from example in `example/` directory
2. Use `boltzgen check design.yaml` to validate
3. Visualize output CIF in PyMOL, Chimera, or https://molstar.org/viewer/
4. Verify binding site coloring matches intent
5. Iterate on YAML until satisfied

## Important Notes

- **GPU Requirements**: Design, inverse folding, and folding steps require CUDA-capable GPU
- **Memory**: Diffusion batch size defaults to 1 (small jobs) or 10 (large jobs). Adjust `--diffusion_batch_size` based on GPU memory
- **Filtering is Fast**: The filtering step runs in ~15-20 seconds on CPU, so rerun with different thresholds as needed
- **File References in YAML**: All paths in design spec YAML are relative to the YAML file's directory
- **Model Cache**: First run downloads ~6GB of models to `~/.cache` (or `$HF_HOME`)
- **Filter Notebook**: Use `filter.ipynb` for interactive filtering exploration (more convenient than CLI)
- **Residue Numbering**: Always use 1-indexed `label_asym_id`, verify in MolStar viewer
- **Multiple Checkpoints**: `--design_checkpoints` accepts multiple models, each used for equal fraction of designs

## Common Pitfalls

- Using `auth_asym_id` instead of `label_asym_id` for residue indices
- Not running `boltzgen check` before full pipeline (wastes GPU time on malformed specs)
- Setting `--num_designs` too low for meaningful diversity (recommend 10k-60k for production)
- Forgetting to activate conda environment before running
- Not adjusting filtering thresholds after initial run (always iterate on filtering)
- Using `--diffusion_batch_size` larger than `--num_designs` (prevents length diversity)
