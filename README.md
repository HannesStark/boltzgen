
<div align="center">
  <div>&nbsp;</div>
  <img src="assets/boltzgen.png" alt="BoltzGen logo" width="60%">

[Paper](https://hannes-stark.com/assets/boltzgen.pdf) | 
[Slack](https://boltz.bio/join-slack) | 
[Video](https://www.youtube.com/watch?v=9d_QWUUI1Qo) <br> <br>
 ![alt text](assets/cover.png)
</div>

# Installation
In an environment with python >=3.11:
```bash
pip install boltzgen
```

<details>
  <summary style="font-size: 1.3em; font-weight: 600;">
    Click for detailed installation instructions
  </summary>

### 1 - Install Miniconda

Choose the installer for your operating system, download it, and follow the on-screen prompts:

* **Windows:** <https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation>
* **MacOS / Linux:** <https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation>

After installation, **open a terminal / command prompt** (you may need to search for “Anaconda Prompt” on Windows).

### 2 - Create a Miniconda Python environment

Run the command below in a terminal to create a fresh environment called `bg` with Python 3.12:

```bash
conda create -n bg python=3.12
```
* **MacOS**

Create a new conda environment for boltzgen with python 3.12, numba, numpy and lvmlite:

```
conda create --name bg python=3.12 llvmlite==0.44.0 numba==0.61.0 numpy==2.0.2
```
Temporary fix for loading multiple libomp
```
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 3 - Activate the environment (do this every time you work with BoltzGen)

```bash
conda activate bg
```

> If you open a **new** terminal session later, you must run `conda activate bg` again before using BoltzGen.

### 4 - Install BoltzGen

Run the command below to install BoltzGen from PyPI:

```bash
pip install boltzgen
```

Alternatively, if you prefer to install an editable, locally-managed copy, download the BoltzGen repository, change directory into the boltzgen directory, and install BoltzGen from source:

```bash
pip install -e .
```
</details>

<details>
  <summary style="font-size: 1.3em; font-weight: 600;">
    Click for optional Docker instructions if you prefer Docker
  </summary>

To build and run the docker image:

```bash
# Build
docker build -t boltzgen .

# Run an example
mkdir -p workdir  # output
mkdir -p cache    # where models will be downloaded to
docker run --rm --gpus all -v "$(realpath workdir)":/workdir -v "$(realpath cache)":/cache -v "$(realpath example)":/example boltzgen \
    boltzgen run /example/vanilla_protein/1g13prot.yaml --output /workdir/test \
 	--protocol protein-anything \
  	--num_designs 2
```

In the example above, the model weights are downloaded the first time the image is run. To bake the weights into the image at build time, run:

```bash
docker build -t boltzgen:weights --build-arg DOWNLOAD_WEIGHTS=true .
```
</details>
<br>


# Running BoltzGen
![alt text](assets/fig1.png)


`boltzgen run` takes a [design specification](#how-to-make-a-design-specification-yaml) `.yaml` and produces a set of ranked designs.\
⚠️ it downloads models (~6GB) to `~/.cache`. This can by changed by passing `--cache YOUR_PATH` or by setting `$HF_HOME`.\
⚠️ If your run is ever interrupted, you can restart it with `--reuse`. No progress is lost.

```bash
boltzgen run example/vanilla_protein/1g13prot.yaml \
  --output workbench/test_run \
  --protocol protein-anything \
  --num_designs 10 \
  --budget 2
# --num_designs is the number of intermediate designs. In practice you will want between 10,000 - 60,000
# --budget is how many designs should be in the final diversity optimized set
```
All command line args are explained in ["All Command Line Arguments"](#all-command-line-arguments).\
**Step-by-step guide for making your designs:**
1. Make your `.yaml` file that specifies your target and what you want to design. We provide many examples in
   `example` such as `example/vanilla_peptide_with_target_binding_site/beetletert.yaml`. Details in
   ["How to make a design specification .yaml"](#how-to-make-a-design-specification-yaml).
2. Check whether your design specification is as intended.  
   1. Run `boltzgen check example/vanilla_peptide_with_target_binding_site/beetletert.yaml`.  
   2. Visualize the resulting mmcif file in a protein structure viewer (e.g. PyMOL, Chimera, or online: https://molstar.org/viewer/).
   3. Your viewer should show the binding site in a different color than the rest of the target.
3. Run the `boltzgen run ...` command as above on your `.yaml` file. 
4. Your filtered, ranked set of designs will be in `--output`. <img src="assets/fig_seconds_per_design.png" alt="Seconds per design" align="right" width="35%">
5. You likely want to rerun the filtering step with different settings (takes ~15 sec). Use
   `boltzgen run --steps filtering --output ...` or the Jupyter notebook `filter.ipynb` which is often more convenient.
   Detailed explanation in ["Rerunning the Filtering"](#rerunning-the-filtering-recommended).

**How many designs to generate?** \
More is better. The "minimum" depends on your target.  
BoltzGen should be run on a GPU. On the right you can see the time required for each step in the pipeline for a single design on an A100 GPU.

We suggest first running with e.g. `--num_design 50`, checking that everything behaves as desired, and then increasing `--num_design` to between 10,000 - 60,000.

## Pipeline output
When the pipeline completes your output directory will have:
- `config/`, `steps.yaml`: configuration files.
- `intermediate_designs/`: output of design step
  - `/*.cif` and `/*.npz`: CIF and NPZ (metadata files) for the designed proteins and targets before inverse folding
- `intermediate_designs_inverse_folded/`: output of inverse folding, folding, and analysis steps
  - `/*.cif` and `/*.npz` : CIF and NPZ for designed proteins and targets after inverse folding.  *Note: For designed residues, only the backbone atoms will have coordinates (sidechain coordinates will be 0,0,0).*
  - `/refold_cif`: refolded complex structures (target and binder). This is the primary input to the analysis and filtering steps.
  - `/refold_design_cif`: refolded binder structures, without target.
  - `/aggregate_metrics_analyze.csv`, `/per_target_metrics_analyze.csv` — outputs of the analysis step.
- `final_ranked_designs/` : outputs of the filtering step
  - `/intermediate_ranked_<N>_designs/` — top-N quality designs. CIFs are copied from `refold_cif` above.
  - `/final_<budget>_designs/` — quality + diversity set. CIFs copied from `refold_cif/`.
  - `/all_designs_metrics.csv` — metrics for all designs considered by filtering.
  - `/final_designs_metrics_<budget>.csv` — metrics for the selected final set.
  - `/results_overview.pdf` — plots

# Protocols

| Protocol (design-target) | Appropriate for                                                           | Major config differences        |
|--------------------------|---------------------------------------------------------------------------|------------------------|
| protein-anything         | Design proteins to bind proteins or peptides                              | Includes `design folding` step. |
| peptide-anything         | Design (cyclic) peptides or others to bind proteins | No Cys are generated in inverse folding. No `design folding` step. Don't compute largest hydrophobic patch. |
| protein-small_molecule   | Design proteins to bind small molecules                                | Includes binding affinity prediction. Includes `design folding` step. |
| antibody-anything        | Design antibody CDRs      | No Cys are generated in inverse folding. No `design folding` step. Don't compute largest hydrophobic patch. |
| nanobody-anything        | Design nanobody CDRs      | Same settings as antibody-anything |
| protein-redesign         | Redesign or optimize existing proteins | No `design folding` step. Uses `design_mask` for target/template definition. |

All configuration parameters can be overridden using the `--config` option; see `boltzgen run --help` or the `Advanced Users` section below for details.



# How to make a design specification .yaml
A more detailed explanation of how our <code>.yaml</code> design specification files work is in <a href="example/README.md" target="_blank">example/README.md</a>. Below is an example based explanation, which is sufficient for most tasks.

**IMPORTANT:** ⚠️ All residue indices are specified **starting at 1** and we use the canonical mmcif residue index `label_asym_id`, and **not** the `auth_asym_id` author residue index! 
You can check the indexing in your mmcif file by opening it in https://molstar.org/viewer/, hovering over a residue, and checking the index on the bottom right. You will see something like this where **41 is the index we use, the auth id 22 is incorrect**:

![](assets/label_seq_id.png)

After you constructed your `.yaml` file we recommend that you run the `check` command on it:
1. Run `boltzgen check example/vanilla_peptide_with_target_binding_site/beetletert.yaml`.  
2. Visualize the resulting mmcif file in a protein structure viewer (e.g. PyMOL, Chimera, or online: https://molstar.org/viewer/).
3. Your viewer should show the binding site in a differnt color than the rest of the target. 


## Example based explanation:
We provide many example `.yaml` files in the `example/` directory, including:

- `example/design_spec_showcasing_all_functionalities.yaml`
- `example/vanilla_peptide_with_target_binding_site/beetletert.yaml`
- `example/peptide_against_specific_site_on_ragc/rragc.yaml`
- `example/nanobody/penguinpox.yaml`
- `example/fab_targets/pdl1.yaml`
- `example/denovo_zinc_finger_against_dna/zinc_finger.yaml`
- `example/protein_binding_small_molecule/chorismite.yaml`
- `example/small_molecule_from_file_and_smiles/4g37.yaml`

Small example of a protein design against a target protein without binding site specified:
```yaml
entities:
  # Designed protein with between 80 and 140 residues 
  # (The length is randomly sampled)
  - protein: 
      id: B
      sequence: 80..140

  # The target is extracted from a .cif file
  - file:
      # file references are relative to the location of the .yaml file
      path: 6m1u.cif # .pdb files also work

      # Which chain in the .cif file to use as target (uses all chains if unspecified)
      include: 
        - chain:
            id: A
```

**IMPORTANT:** ⚠️ File references inside a yaml file (e.g. to cif files) are interpreted relative to the directory of the yaml file.


Example highlighting many (not all) functionalities:
```yaml
entities:
  # Specification of the target which is extracted from a .cif file
  - file:
      path: 8r3a.cif # .pdb files also work
      
      # Which chain and residues in the .cif file to use as target (uses all chains if unspecified)
      include: 
        - chain:
            id: A
            res_index: 2..50,55.. # residues between 2 and 50 and anything larger than 55
        - chain:
            id: B

      # Wich regions of the target the design should or should NOT
      # bind to (this can be left unspecified, then we just bind anywhere)
      binding_types:
        - chain:
            id: A
            binding: 5..7,13
        - chain:
            id: B
            not_binding: "all" 
      
      # Which regions of the target should have their structure specified.
      # By default, everything is visibility 1 which means that the structure is specified.
      # If the visibility is 0, then the structure is not specified.
      structure_groups:
        - group:
            visibility: 1
            id: A
            res_index: 10..13
        - group:
            # The relative positioning of things in structure group 2
            # is not specified w.r.t to things in structure group 1
            visibility: 2 
            id: B
        # Overwrite the previous visibility setting and set it to 0 for res_index 13
        - group:
            visibility: 0
            id: A
            res_index: 13 

      # Optionally you can say that some residues in a loaded .cif file should also be redesigned.
      design:
        - chain:
            id: A
            res_index: 14..19

      # For designed regions you can say what secondary structure they should have
      secondary_structure:
        - chain:
            id: A
            loop: 14
            helix: 15..17
            sheet: 19

  # Specify a NON-designed protein chain
  - protein: 
      id: X
      sequence: AAVTTTTPPP

  # Specify a designed protein chain 
  # Numbers specify what is being designed
  - protein: 
      id: G
      # random number between 15 and 20 of designed residues (inclusive)
      sequence: 15..20AAAAAAVTTTT18PPP 

  # A designed helical peptides with WHL staple  
  # (see the constraints below that connect the peptide with the WHL ligand)
  - protein: 
      id: R
      # Random number of design residues between 3 and 5,
      # then a Cystein, then 6 design residues, then ...
      sequence: 3..5C6C3 
  - ligand:
      id: Q
      ccd: WHL
  
  # A designed peptide with 17 residues
  - protein:
      id: H
      sequence: 17

  # specification for a designed peptide with two Cys and a disulfide bond (see constraints)
  - protein:
      id: S
      sequence: 10..14C6C3

constraints:
    # specify connections as if the minimum possible number of residues was sampled
  - bond:
      atom1: [R, 4, SG] # connection for a helical peptides with WHL staple  between small molecule and designed peptide
      atom2: [Q, 1, CK]
  - bond:
      atom1: [R, 11, SG] # connection for a helical peptides with WHL staple  between small molecule and designed peptide
      atom2: [Q, 1, CH]
  - bond:
      atom1: [S, 11, SG] # connection for a disulfide bond between Cys and Cys in designed peptide
      atom2: [S, 18, SG]

```

## Symmetric complex design (inverse-folding only)

For symmetric complexes (e.g., homo-dimers), the you can tie sequence generation during inverse folding by specifying the `symmetric_group` for each symmetric chain. The `protein-redesign` protocol allows for scoring complexes without separate binders/targets.

```yaml
entities:
  - file:
      path: symmetric_dimer.cif
      include:
        - chain:
            id: A
            res_index: 100..300
            symmetric_group: 1  # Link chains A and B for symmetric sampling
        - chain:
            id: B
            res_index: 100..300
            symmetric_group: 1  # Same group = same sampled insertion length

      # Mark residues to redesign on both chains
      design:
        - chain:
            id: A
            res_index: 200..210
        - chain:
            id: B
            res_index: 200..210
```

# Running only specific pipeline steps

You can run only specific parts of the pipeline using the `--steps` flag:

**Run only the design and inverse_folding steps:**
```bash
boltzgen run example/cyclotide/3ivq.yaml \
  --output workbench/partial-run \
  --protocol peptide-anything \
  --steps design inverse_folding \
  --num_designs 2
```

If you want to run only the inverse folding and subsequent design evaluation steps (but not the backbone design step), you can also run:

**Run only inverse_folding step:**
```bash
boltzgen run example/inverse_folding/1brs.yaml \
  --output workbench/if-only \
  --only_inverse_fold \
  --inverse_fold_num_sequences 2
```

**Available steps:**
- `design` - Generate num_design candidates using the diffusion model based on your design specification
- `inverse_folding` - Redesign sequences from the previous step using our inverse folding model
- `folding` - Re-fold the designed binders with their targets using Boltz-2 model
- `design_folding` - Re-fold the designed binders alone without target (disabled for peptide and nanobody binders)
- `affinity` - Predict binding affinity between designed proteins and their target small molecules using Boltz-2 (for design of small molecule binders only)
- `analysis` - Analyze the folded structures using various metrics to assess design quality
- `filtering` - Filter and rank designs based on analysis results to select the best candidates


# Rerunning the filtering (recommended)
After you generate designs, you will probably want to rerun the filtering step (which runs very fast) several times to tune your criteria for selecting good ones.

You can run the filtering step either using the `boltzgen` command or 
using a [jupyter notebook](filter.ipynb) that we provide. In most cases the notebook is more convenient. If you'd prefer to use the command-line, here is an example of re-running the filters without the notebook.

First, suppose we initially generated some designs with default filtering options:

```bash
boltzgen run example/binding_disordered_peptides/tpp4.yaml \
  --output workbench/tpp4 \
  --protocol protein-anything \
  --num_designs 20
```

After this runs we see that only a few designs passed our filters. We might now adjust the filters by running:

```
boltzgen run example/binding_disordered_peptides/tpp4.yaml \
  --output workbench/tpp4 \
  --protocol protein-anything \
  --steps filtering \
  --refolding_rmsd_threshold 3.0 \
  --filter_biased=false \
  --additional_filters 'ALA_fraction<0.3' 'filter_rmsd_design<2.5' \
  --metrics_override plip_hbonds_refolded=4 \
  --alpha 0.2
```

# Running on SLURM
See [slurm-example](slurm-example).

# All command line arguments

## `boltzgen run`
The `boltzgen run` command executes the BoltzGen binder design pipeline. Here are all available options:

### Design Specification
- `design_spec` - Path(s) to design specification YAML file(s), or a directory containing prepared configs

### General Configuration
- `--protocol {protein-anything,peptide-anything,protein-small_molecule,nanobody-anything,antibody-anything}` - Protocol to use for the design. This determines default settings and in some cases what steps are run. Default: protein-anything. See [Protocols](#protocols) section for details.
- `--output OUTPUT` - Output directory for pipeline results
- `--config CONFIG [CONFIG ...]` - Override pipeline step configuration, in format `<step_name> <arg1>=<value1> <arg2>=<value2> ...` (example: `--config folding num_workers=4 trainer.devices=4`). Can be used multiple times.
- `--devices DEVICES` - Number of devices to use. Default is all devices available.
- `--num_workers NUM_WORKERS` - Number of DataLoader worker processes.
- `--config_dir CONFIG_DIR` - Path to the directory of default config files. Default: `src/boltzgen/resources/config`
- `--use_kernels {auto,true,false}` - Whether to use kernels. One of 'auto', 'true', or 'false'. Default: auto. If 'auto', will use kernels if the device capability is >= 8.
- `--moldir MOLDIR` - Path to the moldir. Default: `huggingface:boltzgen/inference-data:mols.zip`
- `--reuse` - Reuse existing results across all steps. Generate only as many new designs are needed to achieve the specified total number of designs.

### Design
- `--num_designs NUM_DESIGNS` - Number of total designs to generate. This commonly would be something like 10,000. After generating 10,000 designs we then filter down to `--budget` many designs in the filter step
- `--diffusion_batch_size DIFFUSION_BATCH_SIZE` - Number of diffusion samples to generate per trunk run. If not specified, this defaults to 1 if `--num-designs` is less than 100, and 10 otherwise. Note that for design tasks that randomly sample the binder length (or use randomness in other ways), all designs generated in the same batch will share the same length. Having a large diffusion batch size compared to the total number of designs to generate will therefore not evenly sample the possible lengths.
- `--design_checkpoints DESIGN_CHECKPOINTS [DESIGN_CHECKPOINTS ...]` - Path to the boltzgen checkpoint(s). One or more checkpoints are supported. Just specifying an individual path here will work. Each will be used for an equal fraction of the designs. By default, two checkpoints are used. Default: `['huggingface:boltzgen/boltzgen1_diverse:boltzgen1_diverse.ckpt', 'huggingface:boltzgen/boltzgen1_adherence:boltzgen1_adherence.ckpt']`
- `--step_scale STEP_SCALE` - Fixed step scale to use (e.g. 1.8). Default is to use a schedule
- `--noise_scale NOISE_SCALE` - Fixed noise scale to use (e.g. 0.98). Default is to use a schedule

### Inverse Folding
- `--skip_inverse_folding` - Skip inverse folding step
- `--inverse_fold_num_sequences INVERSE_FOLD_NUM_SEQUENCES` - Number of sequences per backbone to generate in the inverse fold step. Default: 1
- `--inverse_fold_checkpoint INVERSE_FOLD_CHECKPOINT` - Path or huggingface repo and filename for the inverse fold checkpoint. Default: `huggingface:boltzgen/boltzgen1_ifold:boltzgen1_ifold.ckpt`
- `--inverse_fold_avoid INVERSE_FOLD_AVOID` - Disallowed residues as a string of one letter amino acid codes, e.g. 'KEC'. This is implemented at the inverse fold step, so it only affects results if inverse folding is enabled. Default: none for protein design, 'C' for peptide and nanobody design. Pass an empty list if you want Cysteins to be generated if you are using a nanobody or peptide protocol
- `--only_inverse_fold` - Skip design step and only run inverse folding. Requires a fully specified structure.

### Folding and Affinity Prediction
- `--folding_checkpoint FOLDING_CHECKPOINT` - Path to the folding checkpoint. Default: `huggingface:boltzgen/boltz2_conf_final:boltz2_conf_final.ckpt`
- `--affinity_checkpoint AFFINITY_CHECKPOINT` - Path to the affinity predictor checkpoint. Default: `huggingface:boltzgen/boltz2_affinity:boltz2_aff.ckpt`

### Filtering
- `--budget BUDGET` - How many designs should be in the final diversity optimized set. This is used in the filtering step.
- `--alpha ALPHA` - Trade-off for sequence diversity selection: 0.0=quality-only, 1.0=diversity-only. Default is 0.01 (peptide-anything protocol) or 0.001 (other protocols).
- `--filter_biased {true,false}` - Remove amino-acid composition outliers (default caps on ALA/GLY/GLU/LEU/VAL). Default: true.
- `--metrics_override METRICS_OVERRIDE [METRICS_OVERRIDE ...]` - Per-metric inverse-importance weights for ranking. Format: `metric_name=weight` (e.g., `plip_hbonds_refolded=4 delta_sasa_refolded=2`). A larger value down-weights that metric's rank. Use `metric_name=none` to remove a metric.
- `--additional_filters ADDITIONAL_FILTERS [ADDITIONAL_FILTERS ...]` - Extra hard filters. Format: `feature>threshold` or `feature<threshold` (e.g., `'design_ALA>0.3' 'design_GLY<0.2'`). Use '>' if higher is better, '<' if lower is better. Make sure to single-quote the strings so your shell doesn't get confused by < and > characters.
- `--size_buckets SIZE_BUCKETS [SIZE_BUCKETS ...]` - Optional constraint for maximum number of designs in size ranges. Format: `min-max:count` (e.g., `10-20:5 20-30:10 30-40:5`).
- `--refolding_rmsd_threshold REFOLDING_RMSD_THRESHOLD` - Threshold used for RMSD-based filters (lower is better).

### Execution Options
- `--no_subprocess` - Run each step in the main process. Will cause issues when devices >1.
- `--steps {design,inverse_folding,design_folding,folding,affinity,analysis,filtering} [{design,inverse_folding,design_folding,folding,affinity,analysis,filtering} ...]` - Run only the specified pipeline steps (default: run all steps). See [The individual pipeline steps](#the-individual-pipeline-steps) section for details.

### Model and Data Download Options
- `--force_download` - Force a (re)-download of models and data.
- `--models_token MODELS_TOKEN` - Secret token to use for our models hosting service (Hugging Face). Default: `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`
- `--cache CACHE` - Directory where downloaded models will be stored. Default: `~/.cache`

## `boltzgen download`

The `boltzgen download` command downloads model weights and data artifacts needed for BoltzGen. In most cases you don't need to use `boltzgen download`, since `boltzgen run` will download what is needed automatically.

Downloaded weights and datasets are stored in `~/.cache` by default but this can be changed by specifying `--cache`.

### Example

```bash
boltzgen download all # downloads all models
boltzgen download inverse-fold # downloads only the inverse folding model
```

### Usage
```bash
boltzgen download [-h] [--force_download] [--models_token MODELS_TOKEN] [--cache CACHE] {affinity,design-adherence,design-diverse,folding,inverse-fold,moldir,all} [{affinity,design-adherence,design-diverse,folding,inverse-fold,moldir,all} ...]
```

### Positional arguments
- `{affinity,design-adherence,design-diverse,folding,inverse-fold,moldir,all}` - Subset of artifacts to download, or 'all' to download all artifacts.

### Options
- `--force_download` - Force a (re)-download of models and data.
- `--models_token MODELS_TOKEN` - Secret token to use for our models hosting service. Not usually required.
- `--cache CACHE` - Directory where downloaded models will be stored. Default: `~/.cache`

## `boltzgen configure`
For more control over your design process, you can separate the configuration generation from execution:

### Example
```bash
boltzgen configure example/cyclotide/3ivq.yaml \
  --output workbench/test-peptide-protein \
  --protocol peptide-anything \
  --num_designs 2 \
  --reuse
```

This creates configuration files in `workbench/test-peptide-protein/` without running the actual design pipeline. You can edit these files if needed and then run `boltzgen execute workbench/test-peptide-protein` to run the workflow.

The options that `boltzgen configure` takes are a subset of the `boltzgen run` options so we don't list them all again here. Try `boltzgen configure --help` if you need help.

## `boltzgen execute`

The `boltzgen execute` command executes a pre-configured pipeline from a directory of config files generated by the `boltzgen configure` command.

### Usage
```bash
boltzgen execute [-h] [--no_subprocess] [--steps {design,inverse_folding,design_folding,folding,affinity,analysis,filtering} [{design,inverse_folding,design_folding,folding,affinity,analysis,filtering} ...]] output
```

### Positional Arguments
- `output` - Directory containing pre-configured pipeline files (generated by 'configure' command)

### Execution Options
- `--no_subprocess` - Run each step in the main process. Will cause issues when devices >1.
- `--steps {design,inverse_folding,design_folding,folding,affinity,analysis,filtering} [{design,inverse_folding,design_folding,folding,affinity,analysis,filtering} ...]` - Run only the specified pipeline steps (default: run all steps)

## `boltzgen merge`

If you produced designs across multiple pipeline runs (e.g. for parallelization) you can merge the finished outputs into one directory and then rerun the fast filtering step on the combined set.

### Example
```bash
boltzgen merge workbench/run_a workbench/run_b workbench/run_c \
  --output workbench/merged_run

# Now rerun filtering (with any tweaked parameters you like)
boltzgen run example/vanilla_protein/1g13prot.yaml \
  --steps filtering \
  --output workbench/merged_run \
  --protocol protein-anything \
  --budget 60 \
  --alpha 0.05
```

### Usage
```bash
boltzgen merge [-h] --output OUTPUT source [source ...]
```

### Arguments
- `source` (positional) – One or more BoltzGen output directories that already contain folded/analyzed results (i.e., the directories you previously passed to `--output` when running the pipeline).
- `--output OUTPUT` – Destination directory for the merged data. The command creates (or replaces) the design artifacts inside this folder so that `boltzgen run --steps filtering --output OUTPUT ...` can be executed afterwards.

# Training BoltzGen models
Install in dev mode which will install additional packages like `wandb`.
```bash
git clone https://github.com/HannesStark/boltzgen
pip install -e .[dev]
```
### 1 – Download training data and checkpoints
```bash
# Choose any location; this is default in yaml files
mkdir -p training_data
cd training_data

# ─ Targets ─
wget -O targets.zip "https://huggingface.co/datasets/boltzgen/boltzgen1_train/resolve/main/targets.zip?download=true"
unzip targets.zip      # → training_data/targets/

# ─ MSAs  ─
wget -O msa.zip "https://huggingface.co/datasets/boltzgen/boltzgen1_train/resolve/main/msa.zip?download=true"
unzip msa.zip          # → training_data/msa/

# ─ Small-molecule dictionary ─
wget -O mols.zip "https://huggingface.co/datasets/boltzgen/inference-data/resolve/main/mols.zip?download=true"
mkdir mols && cd mols && unzip ../mols.zip && cd ..

# ─ Folding checkpoint  ─
wget -O boltz2_fold.ckpt "https://huggingface.co/boltzgen/boltzgen-1/resolve/main/boltz2_conf_final.ckpt?download=true"

# ─────────── (optional) pretrained structure-only ckpt ───────────
# Needed ONLY if you want to resume from a structure-trained model.
wget -O boltzgen1_structuretrained_small.ckpt \
  "https://huggingface.co/boltzgen/boltzgen-1/resolve/main/boltzgen1_structuretrained_small.ckpt?download=true"
```
Resulting layout
```
training_data/
 ├─ targets/     (used for target_dir in yaml)     
 ├─ msa/         (used for msa_dir in yaml)     
 ├─ mols/        (used for mol_dir in yaml) 
 ├─ boltz2_fold.ckpt    (used for folding_checkpoint in yaml) 
 └─ boltzgen1_structuretrained_small.ckpt   (used for pretrained in yaml)
```

The directory `training_data` is the default location referenced in the 
example YAML configuration files.  If you place the data elsewhere, be sure to update those paths accordingly.
### 2 – Training YAML files

Below is a quick reference for the three training configurations and how to launch them once paths are set:

| Config file | Purpose | Example command |
|-------------|---------|-----------------|
| `src/boltzgen/resources/config/train/boltzgen_small.yaml` | Train the **small** Boltzgen model (recommended for development, 8 GPUs, gradient accumulation 16) | `python src/boltzgen/resources/main.py src/boltzgen/resources/config/train/boltzgen_small.yaml  name=boltzgen_small` |
| `src/boltzgen/resources/config/train/boltzgen.yaml` | Train the **large** BoltzGen model | `python src/boltzgen/resources/main.py src/boltzgen/resources/config/train/boltzgen.yaml        name=boltzgen_large` |
| `src/boltzgen/resources/config/train/inverse_folding.yaml` | Train the **inverse-folding** model only | `python src/boltzgen/resources/main.py src/boltzgen/resources/config/train/inverse_folding.yaml name=boltzgen_if` |

If you store the data somewhere other than `./training_data`, search and replace that path **in all three YAML files**.  Typical keys you may need to update are `target_dir`, `msa_dir`, `moldir`, `pretrained`, `folding_checkpoint`, `monomer_target_dir`, and `ligand_target_dir`.

Example places:

```yaml
data:
  datasets:
    - target_dir: ./training_data/targets
      msa_dir:    ./training_data/msa
  moldir: ./training_data/mols

pretrained: ./training_data/boltzgen1_structuretrained_small.ckpt

folding_checkpoint: ./training_data/boltz2_fold.ckpt
```

### 3 – Train the models (example commands)

Small model on 8 GPUs gradient accumulation 16 (recommended dev setup):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/boltzgen/resources/main.py src/boltzgen/resources/config/train/boltzgen_small.yaml \
       name=boltzgen_small
```

Large model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/boltzgen/resources/main.py src/boltzgen/resources/config/train/boltzgen.yaml \
       name=boltzgen_large
```

Inverse-folding model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/boltzgen/resources/main.py src/boltzgen/resources/config/train/inverse_folding.yaml \
       name=boltzgen_if
```

Note: the large model currently expects additional distillation datasets (to be released). You can still explore its hyper-parameters and train solely on PDB data by adjusting the paths.

**Optionally resuming from a checkpoint**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/boltzgen/resources/main.py src/boltzgen/resources/config/train/boltzgen_small.yaml \
       pretrained=./training_data/boltzgen1_structuretrained_small.ckpt \
       name=boltzgen_small_pretrained
```

# Cite
```
@article{stark2025boltzgen,
	author = {Stark, Hannes and Faltings, Felix and Choi, MinGyu and Xie, Yuxin and Hur, Eunsu and O{\textquoteright}Donnell, Timothy John and Bushuiev, Anton and U{\c c}ar, Talip and Passaro, Saro and Mao, Weian and Reveiz, Mateo and Bushuiev, Roman and Pluskal, Tom{\'a}{\v s} and Sivic, Josef and Kreis, Karsten and Vahdat, Arash and Ray, Shamayeeta and Goldstein, Jonathan T. and Savinov, Andrew and Hambalek, Jacob A. and Gupta, Anshika and Taquiri-Diaz, Diego A. and Zhang, Yaotian and Hatstat, A. Katherine and Arada, Angelika and Kim, Nam Hyeong and Tackie-Yarboi, Ethel and Boselli, Dylan and Schnaider, Lee and Liu, Chang C. and Li, Gene-Wei and Hnisz, Denes and Sabatini, David M. and DeGrado, William F. and Wohlwend, Jeremy and Corso, Gabriele and Barzilay, Regina and Jaakkola, Tommi},
	title = {BoltzGen: Toward Universal Binder Design},
	year = {2025},
	doi = {10.1101/2025.11.20.689494},
	journal = {bioRxiv}
}
```
