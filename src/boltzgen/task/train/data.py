from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from rdkit.Chem import Mol
from torch import Tensor
from torch.utils.data import DataLoader

from boltzgen.data import const
from boltzgen.data.crop.cropper import Cropper
from boltzgen.data.select.selector import Selector
from boltzgen.data.data import (
    MSA,
    Input,
    Manifest,
    Record,
    Structure,
)
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.filter.dynamic.filter import DynamicFilter
from boltzgen.data.mol import load_canonicals, load_molecules
from boltzgen.data.pad import pad_to_max
from boltzgen.data.sample.sampler import Sample, Sampler
from boltzgen.data.template.features import load_dummy_templates
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.task.predict import data_ligands, data_protein_binder


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    target_dir: str
    msa_dir: str
    prob: Optional[float]
    sampler: Sampler
    cropper: Cropper
    selector: Optional[Selector] = None
    manifest_path: Optional[str] = None
    filters: Optional[list[DynamicFilter]] = None
    split: Optional[str] = None
    symmetry_correction: bool = True
    val_group: Optional[str] = "RCSB"
    use_train_subset: Optional[float] = None
    moldir: Optional[str] = None
    override_bfactor: Optional[bool] = False
    override_method: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration."""

    datasets: List[DatasetConfig]
    featurizer: Featurizer
    tokenizer: Tokenizer
    selector: Selector
    max_atoms: int
    max_tokens: int
    max_seqs: int
    samples_per_epoch: int
    batch_size: int
    num_workers: int
    random_seed: int
    pin_memory: bool
    atoms_per_window_queries: int
    min_dist: float
    max_dist: float
    num_bins: int
    overfit: Optional[int] = None
    pad_to_max_tokens: bool = False
    pad_to_max_atoms: bool = False
    pad_to_max_seqs: bool = False
    return_train_symmetries: bool = False
    return_val_symmetries: bool = True
    val_batch_size: int = 1
    single_sequence_prop_training: float = 0.0
    msa_sampling_training: bool = False
    moldir: Optional[str] = None
    compute_frames: bool = True
    backbone_only: bool = False
    atom14: bool = False
    atom37: bool = False
    design: bool = False
    monomer_split: str = None
    monomer_target_dir: str = None
    monomer_seq_len: int = 100
    monomer_target_structure_condition: bool = True
    inverse_fold: bool = False
    ligand_split: str = None
    ligand_target_dir: str = None
    ligand_seq_len: int = 100
    use_msa: bool = True
    disulfide_prob: float = 1.0
    disulfide_on: bool = False


@dataclass
class Dataset:
    """Data holder."""

    samples: pd.DataFrame
    struct_dir: Path
    msa_dir: Path
    record_dir: Path
    prob: float
    cropper: Cropper
    tokenizer: Tokenizer
    featurizer: Featurizer
    val_group: str
    selector: Selector
    symmetry_correction: bool = True
    moldir: Optional[str] = None
    override_bfactor: Optional[bool] = False
    override_method: Optional[str] = None


def load_record(record_id: str, record_dir: Path) -> Record:
    """Load the given record.

    Parameters
    ----------
    record_id : str
        The record id to load.
    record_dir : Path
        The path to the record directory.

    Returns
    -------
    Record
        The loaded record.
    """
    return Record.load(record_dir / f"{record_id}.json")


def load_structure(record: Record, struct_dir: Path) -> Structure:
    """Load the given input data.

    Parameters
    ----------
    record : str
        The record to load.
    target_dir : Path
        The path to the data directory.

    Returns
    -------
    Input
        The loaded input.

    """
    if (struct_dir / f"{record.id}.npz").exists():
        structure_path = struct_dir / f"{record.id}.npz"
    else:
        structure_path = struct_dir / f"{record.id}" / f"{record.id}_model_0.npz"
    return Structure.load(structure_path)


def load_msas(chain_ids: set[int], record: Record, msa_dir: Path) -> Input:
    """Load the given input data.

    Parameters
    ----------
    chain_ids : set[int]
        The chain ids to load.
    record : Record
        The record to load.
    msa_dir : Path
        The path to the MSA directory.

    Returns
    -------
    Input
        The loaded input.

    """
    msas = {}
    for chain in record.chains:
        if chain.chain_id not in chain_ids:
            continue

        msa_id = chain.msa_id
        if msa_id != -1:
            msa_path = msa_dir / f"{msa_id}.npz"
            msa = MSA.load(msa_path)
            msas[chain.chain_id] = msa

    return msas


def collate(data: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "chain_swaps",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "activity_name",
            "activity_qualifier",
            "sid",
            "cid",
            "normalized_protein_accession",
            "pair_id",
            "ligand_edge_index",
            "ligand_edge_lower_bounds",
            "ligand_edge_upper_bounds",
            "ligand_edge_bond_mask",
            "ligand_edge_angle_mask",
            "connections_edge_index",
            "ligand_chiral_atom_index",
            "ligand_chiral_check_mask",
            "ligand_chiral_atom_orientations",
            "ligand_stereo_bond_index",
            "ligand_stereo_check_mask",
            "ligand_stereo_bond_orientations",
            "ligand_aromatic_5_ring_index",
            "ligand_aromatic_6_ring_index",
            "ligand_planar_double_bond_index",
            "pdb_id",
            "id",
            "structure_bonds",
            "extra_mols",
        ]:
            if values[0] is not None:
                # Check if all have the same shape
                shape = values[0].shape
                if not all(v.shape == shape for v in values):
                    values = pad_to_max(values, 0)
                else:
                    values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class TrainingDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: List[Dataset],
        canonicals: dict[str, Mol],
        moldir: str,
        samples_per_epoch: int,
        max_atoms: int,
        max_tokens: int,
        max_seqs: int,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        overfit: Optional[int] = None,
        return_symmetries: Optional[bool] = False,
        single_sequence_prop: Optional[float] = 0.0,
        msa_sampling: bool = False,
        compute_frames: bool = True,
        backbone_only: bool = False,
        atom14: bool = False,
        atom37: bool = False,
        design: bool = False,
        disulfide_prob: float = 1.0,
        disulfide_on: bool = False,
        use_msa: bool = True,
        inverse_fold: bool = False,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        datasets : List[Dataset]
            The datasets to sample from.
        samplers : List[Sampler]
            The samplers to sample from each dataset.
        probs : List[float]
            The probabilities to sample from each dataset.
        samples_per_epoch : int
            The number of samples per epoch.
        max_tokens : int
            The maximum number of tokens.

        """
        super().__init__()
        self.datasets = datasets
        self.canonicals = canonicals
        self.moldir = moldir
        self.probs = [d.prob for d in datasets]
        self.samples_per_epoch = samples_per_epoch
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.max_atoms = max_atoms
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.return_symmetries = return_symmetries
        self.backbone_only = backbone_only
        self.atom14 = atom14
        self.atom37 = atom37
        self.design = design
        self.disulfide_prob = disulfide_prob
        self.disulfide_on = disulfide_on
        self.single_sequence_prop = single_sequence_prop
        self.msa_sampling = msa_sampling
        self.use_msa = use_msa
        self.overfit = overfit
        self.compute_frames = compute_frames
        self.inverse_fold = inverse_fold

        self.samples: list[list[Dict]] = []
        self.samples_weight: list[list[float]] = []
        for d in self.datasets:
            if self.overfit:
                samples = d.samples[: self.overfit]
            else:
                samples = d.samples
            self.samples.append(
                [
                    samples.iloc[sample_idx].to_dict()
                    for sample_idx in range(len(samples))
                ]
            )
            self.samples_weight.append(samples["weight"].tolist())

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Set a random state
        random = np.random.default_rng()

        # Pick a random dataset
        dataset_idx = random.choice(len(self.datasets), p=self.probs)

        dataset = self.datasets[dataset_idx]

        # Get a sample from the dataset
        samples = self.samples[dataset_idx]
        sample_idx = random.choice(
            len(samples),
            p=(
                self.samples_weight[dataset_idx]
                / np.sum(self.samples_weight[dataset_idx])
                if self.overfit
                else self.samples_weight[dataset_idx]
            ),
        )

        sample = samples[sample_idx]
        sample: Sample = Sample(
            record_id=str(sample["record_id"]),
            chain_id=(
                int(sample["chain_id"]) if sample["chain_id"] is not None else None
            ),
            interface_id=(
                int(sample["interface_id"])
                if sample["interface_id"] is not None
                else None
            ),
            weight=float(sample["weight"]),
        )

        # Load record
        record = load_record(sample.record_id, dataset.record_dir)

        # Get the structure
        try:
            structure = load_structure(record, dataset.struct_dir)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")
            return self.__getitem__(random.integers(0, len(self)))

        # Tokenize structure
        try:
            tokenized = dataset.tokenizer.tokenize(
                structure, inverse_fold=self.inverse_fold
            )
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")
            traceback.print_exc()  # noqa: T201
            return self.__getitem__(random.integers(0, len(self)))

        # Compute crop
        try:
            if self.max_tokens is not None and len(tokenized.tokens) > self.max_tokens:
                tokenized = dataset.cropper.crop(
                    tokenized,
                    max_atoms=self.max_atoms,
                    max_tokens=self.max_tokens,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                    random=random,
                    prefer_protein_queries=self.inverse_fold,
                )
                if len(tokenized.tokens) == 0:
                    msg = "No tokens in cropped structure."
                    raise ValueError(msg)  # noqa: TRY301
        except Exception as e:  # noqa: BLE001
            print(f"Cropper failed on {record.id} with error {e}. Skipping.")
            traceback.print_exc()  # noqa: T201
            return self.__getitem__(random.integers(0, len(self)))

        # Select which tokens to design
        try:
            tokenized, design_task = dataset.selector.select(
                tokenized,
                random=random,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Selector failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            traceback.print_exc()  # noqa: T201
            return self.__getitem__(random.integers(0, len(self)))
        structure = tokenized.structure

        # Get unique chain ids
        chain_ids = set(tokenized.tokens["asym_id"])

        # Load msas and templates
        try:
            if self.use_msa:
                msas = load_msas(
                    chain_ids=chain_ids,
                    record=record,
                    msa_dir=dataset.msa_dir,
                )
            else:
                msas = {}
        except Exception as e:  # noqa: BLE001
            print(f"MSA loading failed for {record.id} with error {e}. Skipping.")
            return self.__getitem__(random.integers(0, len(self)))

        # Load molecules
        try:
            # Try to find molecules in the dataset moldir if provided
            # Find missing ones in global moldir and check if all found
            molecules = {}
            molecules.update(self.canonicals)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(self.canonicals.keys())
            if dataset.moldir is not None:
                molecules.update(load_molecules(dataset.moldir, mol_names))

            mol_names = mol_names - set(molecules.keys())
            molecules.update(load_molecules(self.moldir, mol_names))
        except Exception as e:  # noqa: BLE001
            print(f"Molecule loading failed for {record.id} with error {e}. Skipping.")
            return self.__getitem__(random.integers(0, len(self)))

        # Finalize input data
        input_data = Input(
            tokens=tokenized.tokens,
            bonds=tokenized.bonds,
            token_to_res=tokenized.token_to_res,
            structure=tokenized.structure,
            msa=msas,
            templates=None,
            record=record,
        )

        # Compute features
        try:
            features: dict = dataset.featurizer.process(
                input_data,
                molecules=molecules,
                random=random,
                training=True,
                max_atoms=self.max_atoms if self.pad_to_max_atoms else None,
                max_tokens=self.max_tokens if self.pad_to_max_tokens else None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                single_sequence_prop=self.single_sequence_prop,
                msa_sampling=self.msa_sampling,
                override_bfactor=dataset.override_bfactor,
                override_method=dataset.override_method,
                compute_frames=self.compute_frames,
                backbone_only=self.backbone_only,
                atom14=self.atom14,
                atom37=self.atom37,
                design=self.design,
                disulfide_prob=self.disulfide_prob,
                inverse_fold=self.inverse_fold,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")
            traceback.print_exc()
            return self.__getitem__(random.integers(0, len(self)))

        # Check that there is enough stuff to design in the inverse folding case so we have no nan losses
        if self.inverse_fold and features["design_mask"].sum() < 3:
            print(f"Skipping {record.id}. Fewer than 3 design residues.")
            return self.__getitem__(random.integers(0, len(self)))

        # Set template features
        template_features = load_dummy_templates(
            tdim=1, num_tokens=len(features["res_type"])
        )
        features.update(template_features)

        features.update({"id": sample.record_id})
        features["pdb_id"] = record.id

        # Assert that all design tokens make sense
        bad_protein_mask = (
            (~features["is_standard"].bool())
            & features["design_mask"].bool()
            & (features["mol_type"] == const.chain_type_ids["PROTEIN"])
        )
        assert not bad_protein_mask.any()

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.samples_per_epoch


class ValidationDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: List[Dataset],
        canonicals: dict[str, Mol],
        moldir: str,
        seed: int,
        max_atoms: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_seqs: Optional[int] = None,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        overfit: Optional[int] = None,
        return_symmetries: Optional[bool] = False,
        compute_frames: bool = True,
        backbone_only: bool = False,
        atom14: bool = False,
        atom37: bool = False,
        design: bool = False,
        inverse_fold: bool = False,
        disulfide_prob: float = 1.0,
        disulfide_on: bool = False,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        datasets : List[Dataset]
            The datasets to sample from.
        seed : int
            The random seed.
        max_tokens : int
            The maximum number of tokens.
        overfit : bool
            Whether to overfit the dataset

        """
        super().__init__()
        self.datasets = datasets
        self.canonicals = canonicals
        self.moldir = moldir
        self.max_atoms = max_atoms
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.seed = seed
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.overfit = overfit
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.return_symmetries = return_symmetries
        self.compute_frames = compute_frames
        self.backbone_only = backbone_only
        self.atom14 = atom14
        self.atom37 = atom37
        self.design = design
        self.inverse_fold = inverse_fold
        self.disulfide_prob = disulfide_prob
        self.disulfide_on = disulfide_on

    def __getitem__(self, idx: int) -> Structure:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Set random state
        seed = self.seed if self.overfit is None else None
        random = np.random.default_rng(seed)

        # Pick dataset based on idx
        for idx_dataset, dataset in enumerate(self.datasets):  # noqa: B007
            size = len(dataset.samples)
            if self.overfit is not None:
                size = min(size, self.overfit)
            if idx < size:
                break
            idx -= size

        # Get a sample from the dataset
        sample = Sample(**dataset.samples.iloc[idx].to_dict())
        record = load_record(sample.record_id, dataset.record_dir)

        # Get the structure
        try:
            structure = load_structure(record, dataset.struct_dir)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized = dataset.tokenizer.tokenize(structure)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Compute crop
        try:
            if self.max_tokens is not None:
                tokenized = dataset.cropper.crop(
                    tokenized,
                    max_atoms=self.max_atoms,
                    max_tokens=self.max_tokens,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                    random=random,
                    prefer_protein_queries=self.inverse_fold,
                )
                if len(tokenized.tokens) == 0:
                    msg = "No tokens in cropped structure."
                    raise ValueError(msg)  # noqa: TRY301
        except Exception as e:  # noqa: BLE001
            print(f"Cropper failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Get unique chains
        chain_ids = set(np.unique(tokenized.tokens["asym_id"]).tolist())

        # Load msas and templates
        try:
            msas = load_msas(chain_ids, record, dataset.msa_dir)
        except Exception as e:  # noqa: BLE001
            print(f"MSA loading failed for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Select which tokens to design
        try:
            tokenized, design_task = dataset.selector.select(
                tokenized,
                random=random,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Selector failed on {sample.record_id} with error {e}. Skipping.")  # noqa: T201
            traceback.print_exc()  # noqa: T201
            return self.__getitem__(0)
        structure = tokenized.structure

        try:
            # Try to find molecules in the dataset moldir if provided
            # Find missing ones in global moldir and check if all found
            molecules = {}
            molecules.update(self.canonicals)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(self.canonicals.keys())
            if dataset.moldir is not None:
                molecules.update(load_molecules(dataset.moldir, mol_names))

            mol_names = mol_names - set(molecules.keys())
            molecules.update(load_molecules(self.moldir, mol_names))
        except Exception as e:  # noqa: BLE001
            print(f"Molecule loading failed for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Finalize input data
        input_data = Input(
            tokens=tokenized.tokens,
            bonds=tokenized.bonds,
            token_to_res=tokenized.token_to_res,
            structure=tokenized.structure,
            msa=msas,
            templates=None,
            record=record,
        )

        # Compute features
        try:
            features: dict = dataset.featurizer.process(
                input_data,
                molecules=molecules,
                random=random,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                single_sequence_prop=0.0,
                override_method=dataset.override_method,
                compute_frames=self.compute_frames,
                backbone_only=self.backbone_only,
                atom14=self.atom14,
                atom37=self.atom37,
                design=self.design,
                inverse_fold=self.inverse_fold,
                disulfide_prob=self.disulfide_prob,
                disulfide_on=self.disulfide_on,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Check that there is enough stuff to design in the inverse folding case so we have no nan losses
        if self.inverse_fold and features["design_mask"].sum() < 3:
            print(f"Skipping {record.id}. Fewer than 3 design residues.")
            return self.__getitem__(0)

        # Set template features
        template_features = load_dummy_templates(
            tdim=1, num_tokens=len(features["res_type"])
        )
        features.update(template_features)

        # Add dataset idx
        idx_dataset = torch.tensor([idx_dataset], dtype=torch.long)
        features.update({"idx_dataset": idx_dataset})
        features.update({"id": record.id})
        bad_protein_mask = (
            (~features["is_standard"].bool())
            & features["design_mask"].bool()
            & (features["mol_type"] == const.chain_type_ids["PROTEIN"])
        )
        assert not bad_protein_mask.any()
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataaset.

        """
        if self.overfit is not None:
            length = sum(len(d.samples[: self.overfit]) for d in self.datasets)
        else:
            length = sum(len(d.samples) for d in self.datasets)

        return length


class TrainingDataModule(pl.LightningDataModule):
    """DataModule for BoltzGen training."""

    def __init__(
        self,
        cfg: DataConfig,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.cfg = cfg
        self.inverse_fold = cfg.inverse_fold

        assert self.cfg.val_batch_size == 1, "Validation only works with batch size=1."

        # Load datasets
        train: List[Dataset] = []
        val: List[Dataset] = []

        for data_config in cfg.datasets:
            # Get relevant directories
            if data_config.manifest_path is not None:
                manifest_path = Path(data_config.manifest_path)
            else:
                manifest_path = Path(data_config.target_dir) / "manifest.json"
            struct_dir = Path(data_config.target_dir) / "structures"
            record_dir = Path(data_config.target_dir) / "records"
            msa_dir = Path(data_config.msa_dir)

            # Get moldir, if any
            moldir = data_config.moldir
            moldir = Path(moldir) if moldir is not None else None

            # Load all records
            manifest: Manifest = Manifest.load(manifest_path)

            # Split records if givens
            if data_config.split is not None:
                with Path(data_config.split).open("r") as f:
                    split = {line.strip().lower() for line in f}

                train_records = []
                val_records = []
                for record in manifest.records:
                    if record.id.lower() in split:
                        val_records.append(record)
                    else:
                        train_records.append(record)
            else:
                train_records = manifest.records
                if cfg.overfit is None:
                    val_records = []
                else:
                    print("Warning: modified overfit val behavior.")
                    val_records = manifest.records[: cfg.overfit]

            print("train_records before filter", len(train_records))

            # Apply dataset-specific filters
            if data_config.filters is not None:
                train_records = [
                    record
                    for record in train_records
                    if all(f.filter(record) for f in data_config.filters)
                ]

            # Train with subset of data
            if data_config.use_train_subset is not None:
                # Shuffle train_records list
                assert 0 < data_config.use_train_subset < 1.0
                rng = np.random.default_rng(cfg.random_seed)
                rng.shuffle(train_records)
                train_records = train_records[
                    0 : int(len(train_records) * data_config.use_train_subset)
                ]
            print("train_records after filter", len(train_records))
            print("val_records after filter", len(val_records))

            # Get samples
            train_samples: list[Sample] = data_config.sampler.sample(train_records)
            val_samples: list[Sample] = [Sample(r.id) for r in val_records]

            # Convert samples to pandas dataframe to avoid copy-on-write behavior
            train_samples = pd.DataFrame(
                [
                    (
                        r.record_id,
                        r.chain_id,
                        r.interface_id,
                        r.weight,
                    )
                    for r in train_samples
                ],
                columns=["record_id", "chain_id", "interface_id", "weight"],
            )
            val_samples = pd.DataFrame(
                [s.record_id for s in val_samples], columns=["record_id"]
            )

            # Use appropriate string type
            train_samples = train_samples.replace({np.nan: None})
            val_samples = val_samples.replace({np.nan: None})
            train_samples["record_id"] = train_samples["record_id"].astype("string")
            val_samples["record_id"] = val_samples["record_id"].astype("string")

            del manifest, train_records, val_records
            # Create train dataset
            if data_config.prob > 0:
                train.append(
                    Dataset(
                        samples=train_samples,
                        record_dir=record_dir,
                        struct_dir=struct_dir,
                        msa_dir=msa_dir,
                        moldir=moldir,
                        prob=data_config.prob,
                        cropper=data_config.cropper,
                        tokenizer=cfg.tokenizer,
                        featurizer=cfg.featurizer,
                        val_group=data_config.val_group,
                        symmetry_correction=data_config.symmetry_correction,
                        override_bfactor=data_config.override_bfactor,
                        override_method=data_config.override_method,
                        selector=cfg.selector,
                    )
                )

            # Create validation dataset
            if len(val_samples) > 0:
                val.append(
                    Dataset(
                        samples=val_samples,
                        record_dir=record_dir,
                        struct_dir=struct_dir,
                        msa_dir=msa_dir,
                        moldir=moldir,
                        prob=data_config.prob,
                        cropper=data_config.cropper,
                        tokenizer=cfg.tokenizer,
                        featurizer=cfg.featurizer,
                        val_group=data_config.val_group,
                        symmetry_correction=data_config.symmetry_correction,
                        selector=cfg.selector,
                    )
                )

        # Print dataset sizes
        for dataset in train:
            dataset: Dataset
            print(f"Training dataset size: {len(dataset.samples)}")

        self.val_group_mapper = defaultdict(dict)

        for i, dataset in enumerate(train if cfg.overfit is not None else val):
            dataset: Dataset
            print(f"Validation dataset size: {len(dataset.samples)}")
            self.val_group_mapper[i]["label"] = dataset.val_group
            self.val_group_mapper[i]["symmetry_correction"] = (
                # If overfit, use symmetry_correction from val dataset instead of training dataset
                dataset.symmetry_correction
                if cfg.overfit is None
                else data_config.symmetry_correction
            )

        # Load canonical molecules
        canonicals = load_canonicals(cfg.moldir)

        # Create wrapper datasets
        self._train_set = TrainingDataset(
            datasets=train,
            canonicals=canonicals,
            moldir=cfg.moldir,
            samples_per_epoch=cfg.samples_per_epoch,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            overfit=cfg.overfit,
            return_symmetries=cfg.return_train_symmetries,
            single_sequence_prop=cfg.single_sequence_prop_training,
            msa_sampling=cfg.msa_sampling_training,
            use_msa=cfg.use_msa,
            compute_frames=cfg.compute_frames,
            backbone_only=cfg.backbone_only,
            atom14=cfg.atom14,
            atom37=cfg.atom37,
            design=cfg.design,
            inverse_fold=cfg.inverse_fold,
            disulfide_prob=cfg.disulfide_prob,
            disulfide_on=cfg.disulfide_on,
        )
        self._val_set = ValidationDataset(
            datasets=train if cfg.overfit is not None else val,
            canonicals=canonicals,
            moldir=cfg.moldir,
            seed=cfg.random_seed,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            overfit=cfg.overfit,
            return_symmetries=cfg.return_val_symmetries,
            compute_frames=cfg.compute_frames,
            backbone_only=cfg.backbone_only,
            atom14=cfg.atom14,
            atom37=cfg.atom37,
            design=cfg.design,
            inverse_fold=cfg.inverse_fold,
            disulfide_prob=cfg.disulfide_prob,
            disulfide_on=cfg.disulfide_on,
        )

        self.monomer_split = cfg.monomer_split
        print("monomer_split", self.monomer_split)
        if self.monomer_split is not None:
            with Path(self.monomer_split).open("r") as f:
                monomer_ids = [line.strip().lower() for line in f]
                print("monomer_split", monomer_ids)

            dataset = data_protein_binder.Dataset(
                struct_dir=Path(cfg.monomer_target_dir) / "structures",
                record_dir=Path(cfg.monomer_target_dir) / "records",
                target_ids=monomer_ids,
                seq_len=cfg.monomer_seq_len,
                tokenizer=cfg.tokenizer,
                featurizer=cfg.featurizer,
            )

            # Load canonical molecules
            canonicals = load_canonicals(cfg.moldir)

            self.monomer_val_set = data_protein_binder.PredictionDataset(
                dataset=dataset,
                canonicals=canonicals,
                moldir=Path(cfg.moldir),
                backbone_only=cfg.backbone_only,
                atom14=cfg.atom14,
                atom37=cfg.atom37,
                design=cfg.design,
                target_structure_condition=cfg.monomer_target_structure_condition,
                inverse_fold=cfg.inverse_fold,
                disulfide_prob=cfg.disulfide_prob,
                disulfide_on=cfg.disulfide_on,
            )

        self.ligand_split = cfg.ligand_split
        print("ligand_split", self.ligand_split)
        if self.ligand_split is not None:
            ligand_ids = [
                x.lower() for x in Path(self.ligand_split).read_text().splitlines()
            ]
            print("ligand_split", ligand_ids)

            dataset = data_ligands.Dataset(
                struct_dir=Path(cfg.ligand_target_dir) / "structures",
                record_dir=Path(cfg.ligand_target_dir) / "records",
                target_ids=ligand_ids,
                min_len=cfg.ligand_seq_len,
                max_len=cfg.ligand_seq_len,
                tokenizer=cfg.tokenizer,
                featurizer=cfg.featurizer,
            )

            # Load canonical molecules
            canonicals = load_canonicals(cfg.moldir)

            self.ligand_val_set = data_ligands.PredictionDataset(
                dataset=dataset,
                canonicals=canonicals,
                moldir=Path(cfg.moldir),
                backbone_only=cfg.backbone_only,
                atom14=cfg.atom14,
                atom37=cfg.atom37,
                design=cfg.design,
                disulfide_prob=cfg.disulfide_prob,
                disulfide_on=cfg.disulfide_on,
            )

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002 (unused)
        """Run the setup for the DataModule.

        Parameters
        ----------
        stage : str, optional
            The stage, one of 'fit', 'validate', 'test'.

        """
        return

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        return DataLoader(
            self._train_set,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.s

        """
        val_loaders = []
        val_loaders.append(
            DataLoader(
                self._val_set,
                batch_size=self.cfg.val_batch_size,
                num_workers=self.cfg.num_workers if not self.inverse_fold else 1,
                pin_memory=self.cfg.num_workers if not self.inverse_fold else False,
                shuffle=False,
                collate_fn=collate,
            )
        )
        if self.monomer_split is not None:
            val_loaders.append(
                DataLoader(
                    self.monomer_val_set,
                    batch_size=self.cfg.val_batch_size,
                    num_workers=self.cfg.num_workers if not self.inverse_fold else 1,
                    pin_memory=self.cfg.pin_memory if not self.inverse_fold else False,
                    shuffle=False,
                    collate_fn=data_protein_binder.collate,
                )
            )
        if self.ligand_split is not None:
            val_loaders.append(
                DataLoader(
                    self.ligand_val_set,
                    batch_size=self.cfg.val_batch_size,
                    num_workers=self.cfg.num_workers if not self.inverse_fold else 1,
                    pin_memory=self.cfg.pin_memory if not self.inverse_fold else False,
                    shuffle=False,
                    collate_fn=data_ligands.collate,
                )
            )
        return val_loaders

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_set,
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

    def transfer_batch_to_device(
        self,
        batch: Dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> Dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        np.Any
            The transferred batch.

        """
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "chain_swaps",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "activity_name",
                "activity_qualifier",
                "sid",
                "cid",
                "normalized_protein_accession",
                "pair_id",
                "ligand_edge_index",
                "ligand_edge_lower_bounds",
                "ligand_edge_upper_bounds",
                "ligand_edge_bond_mask",
                "ligand_edge_angle_mask",
                "connections_edge_index",
                "ligand_chiral_atom_index",
                "ligand_chiral_check_mask",
                "ligand_chiral_atom_orientations",
                "ligand_stereo_bond_index",
                "ligand_stereo_check_mask",
                "ligand_stereo_bond_orientations",
                "ligand_aromatic_5_ring_index",
                "ligand_aromatic_6_ring_index",
                "ligand_planar_double_bond_index",
                "pdb_id",
                "id",
                "tokenized",
                "structure",
                "structure_bonds",
                "extra_mols",
            ]:
                if hasattr(batch[key], "to"):
                    batch[key] = batch[key].to(device)
        return batch
