from dataclasses import dataclass
from boltzgen.data.data import StructureInfo
from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from rdkit.Chem import Mol
from boltzgen.data import const
from boltzgen.data.data import Input, Structure, Record, MSA
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.pad import pad_to_max
from boltzgen.data.mol import load_canonicals, load_molecules
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.template.features import (
    load_dummy_templates,
)
from boltzgen.task.predict.loading_utils import load_record, load_structure


@dataclass
class DataConfig:
    """Data configuration."""

    target_dir: str
    msa_dir: str
    moldir: str
    seq_len: int
    target_ids: str
    tokenizer: Tokenizer
    featurizer: Featurizer
    backbone_only: bool = False
    atom14: bool = False
    atom37: bool = False
    design: bool = False
    target_structure_condition: bool = False
    msa_condition: bool = False
    max_seq: int = 1024
    ss_single_condition: bool = False
    ss_double_condition: bool = False
    ss_short_motif: bool = False
    multiplicity: int = 1
    inverse_fold: bool = False
    num_targets: int = 100000
    disulfide_prob: float = 1.0
    disulfide_on: bool = False


@dataclass
class Dataset:
    """Data holder."""

    struct_dir: Path
    record_dir: Path
    target_ids: List[str]
    seq_len: int
    tokenizer: Tokenizer
    featurizer: Featurizer
    chain_ids: List[str] = None


def ss_single(feats, random):
    feats["design_ss_mask"].zero_()
    feats["ss_type"].zero_()
    design_mask = feats["design_mask"].bool()
    seq_len = design_mask.sum().item()

    feats["design_ss_mask"][design_mask] = 0
    design_indices = torch.where(design_mask)[0]

    motif_length = random.integers(5, min(30, seq_len) + 1)
    start_pos = random.choice(design_indices[: seq_len - motif_length + 1].cpu())

    motif_type = random.choice([2, 3])  # helix or sheet
    feats["design_ss_mask"][start_pos : start_pos + motif_length] = 1
    feats["ss_type"][start_pos : start_pos + motif_length] = motif_type


def ss_double(feats, random):
    feats["design_ss_mask"].zero_()
    feats["ss_type"].zero_()
    design_mask = feats["design_mask"].bool()
    seq_len = design_mask.sum().item()

    feats["design_ss_mask"][design_mask] = 0
    design_indices = torch.where(design_mask)[0]

    motif_len1 = random.integers(5, min(30, seq_len - 10) + 1)
    motif_len2 = random.integers(5, min(30, seq_len - motif_len1 - 5) + 1)
    total_required = motif_len1 + 5 + motif_len2
    start_pos = random.choice(design_indices[: seq_len - total_required + 1].cpu())

    motif_type1 = random.choice([2, 3])
    motif_type2 = random.choice([2, 3])

    # first motif
    feats["design_ss_mask"][start_pos : start_pos + motif_len1] = 1
    feats["ss_type"][start_pos : start_pos + motif_len1] = motif_type1

    # loop
    loop_start = start_pos + motif_len1
    feats["design_ss_mask"][loop_start : loop_start + 5] = 1
    feats["ss_type"][loop_start : loop_start + 5] = 1  # loop

    # second motif
    second_start = loop_start + 5
    feats["design_ss_mask"][second_start : second_start + motif_len2] = 1
    feats["ss_type"][second_start : second_start + motif_len2] = motif_type2


def ss_short_motif(feats, random, helix_prob=0.5):
    feats["design_ss_mask"].zero_()
    feats["ss_type"].zero_()
    design_mask = feats["design_mask"].bool()
    seq_len = design_mask.sum().item()
    design_indices = torch.where(design_mask)[0]

    start_pos = random.choice(design_indices[: seq_len - 4 + 1].cpu())
    motif_type = 2 if random.random() < helix_prob else 3

    feats["design_ss_mask"][start_pos : start_pos + 4] = 1
    feats["ss_type"][start_pos : start_pos + 4] = motif_type


def load_msas(record: Record, chain_ids: set[int], msa_dir: Path) -> Dict[int, MSA]:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    chain_ids : set[int]
        The chain ids to load.
    msa_dir : Path
        The path to the MSA directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the relevant MSAs
    msas = {}
    for chain in record.chains:
        if chain.chain_id in chain_ids:
            msa_id = chain.msa_id
            if msa_id != -1:
                msa = np.load(msa_dir / f"{msa_id}.npz")
                msas[chain.chain_id] = MSA(**msa)
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
            "amino_acids_symmetries",
            "ligand_symmetries",
            "activity_name",
            "activity_qualifier",
            "sid",
            "cid",
            "aid",
            "normalized_protein_accession",
            "pair_id",
            "record",
            "id",
            "structure_bonds",
            "extra_mols",
            "structure",
            "tokenized",
            "data_sample_idx",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class PredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        dataset: Dataset,
        canonicals: dict[str, Mol],
        moldir: str,
        backbone_only: bool = False,
        atom14: bool = False,
        atom37: bool = False,
        design: bool = False,
        target_structure_condition: bool = False,
        msa_condition: bool = False,
        ss_single_condition: bool = False,
        ss_double_condition: bool = False,
        ss_short_motif: bool = False,
        max_seq: int = 1024,
        msa_dir: str = "/data/rbg/shared/projects/foldeverything/rcsb/msa",
        inverse_fold: bool = False,
        multiplicity: int = 1,
        disulfide_prob: float = 1.0,
        disulfide_on: bool = False,
        extra_features: Optional[List[str]] = None,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        datasets : List[Dataset]
            The datasets to sample from.

        """
        super().__init__()
        self.dataset = dataset
        self.moldir = moldir
        self.canonicals = canonicals
        self.backbone_only = backbone_only
        self.atom14 = atom14
        self.atom37 = atom37
        self.design = design
        self.target_structure_condition = target_structure_condition
        self.msa_condition = msa_condition
        self.max_seq = max_seq
        self.msa_dir = msa_dir
        self.inverse_fold = inverse_fold
        self.ss_single_condition = ss_single_condition
        self.ss_double_condition = ss_double_condition
        self.ss_short_motif = ss_short_motif
        self.multiplicity = multiplicity
        self.disulfide_prob = disulfide_prob
        self.disulfide_on = disulfide_on
        self.extra_features = (
            set(extra_features) if extra_features is not None else set()
        )

    def __getitem__(self, idx: int) -> Dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get a sample from the dataset
        data_sample_idx = idx // len(self.dataset.target_ids)
        pdb_id = self.dataset.target_ids[idx % len(self.dataset.target_ids)]
        if not self.dataset.chain_ids is None:
            chain_id = self.dataset.chain_ids[idx % len(self.dataset.target_ids)]
        else:
            chain_id = None

        # Load record
        record = load_record(pdb_id, self.dataset.record_dir)
        
        # Get the structure
        try:
            if self.inverse_fold:
                structure = load_structure(record, self.dataset.struct_dir)
            else:
                str_native = load_structure(record, self.dataset.struct_dir)

                # If chain id is specified, extract chain
                if not chain_id is None:
                    chain_id = chain_id.lower()
                    chain = None
                    chain_names = []
                    for _chain in str_native.chains:
                        chain_names.append(_chain[0].lower())
                        if _chain[0].lower() == chain_id:
                            chain = _chain
                    if chain is None:
                        chain_names = "\n".join(chain_names)
                        msg = f"Could not find chain {chain_id}. Structure contains chains:\n{chain_names}"
                        e = ValueError(msg)
                    res_idx, res_num = chain[7], chain[8]
                    res_idxs = np.arange(res_idx, res_idx + res_num)
                    str_native = str_native.extract_residues(str_native, res_idxs)

                str_prot = Structure.empty_protein(seq_len=self.dataset.seq_len)
                structure = Structure.concatenate(str_native, str_prot)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {pdb_id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized = self.dataset.tokenizer.tokenize(structure)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {pdb_id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        if self.inverse_fold:
            tokenized.tokens["design_mask"] = (
                tokenized.tokens["is_standard"]
                & (tokenized.tokens["mol_type"] == const.chain_type_ids["PROTEIN"])
                & tokenized.tokens["resolved_mask"]
            )
            tokenized.tokens["structure_group"] = 1
        else:
            # Set the last chain to be designed
            tokenized.tokens["design_mask"] = (
                tokenized.tokens["asym_id"] == tokenized.tokens["asym_id"][-1]
            ).astype(tokenized.tokens["asym_id"].dtype)

            if self.target_structure_condition:
                tokenized.tokens["structure_group"][
                    ~tokenized.tokens["design_mask"].astype(bool)
                ] = 1

        # Propagate design mask to obtain chain_design_mask (True whenever something is covalently bound to any residue that is in a chain that contains a design residue).
        chain_design_mask = tokenized.tokens["design_mask"].astype(bool)
        asym_id = tokenized.tokens["asym_id"]
        while True:
            design_chains = np.unique(asym_id[chain_design_mask])
            chain_propagated = np.isin(asym_id, design_chains)
            for i, j, _ in tokenized.bonds:
                if any([chain_propagated[i], chain_propagated[j]]):
                    chain_propagated[i] = True
                    chain_propagated[j] = True
            if np.equal(chain_propagated, chain_design_mask).all():
                break
            chain_design_mask = chain_propagated.astype(bool)

        # Find the record with the matching pdb_id
        msas = {}
        if self.msa_condition:
            chain_ids = set(tokenized.tokens["asym_id"])
            msas = load_msas(record=record, chain_ids=chain_ids, msa_dir=self.msa_dir)

        try:
            # Try to find molecules in the dataset moldir if provided
            # Find missing ones in global moldir and check if all found
            molecules = {}
            molecules.update(self.canonicals)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(self.canonicals.keys())
            if self.moldir is not None:
                molecules.update(load_molecules(self.moldir, mol_names))

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
            structure=structure,
            msa=msas,
            templates=None,
            record=record,
        )
        # Compute features
        try:
            features = self.dataset.featurizer.process(
                input_data,
                molecules=molecules,
                random=np.random.default_rng(None),
                training=False,
                max_seqs=self.max_seq,  # if self.msa_condition else 1,
                backbone_only=self.backbone_only,
                atom14=self.atom14,
                atom37=self.atom37,
                design=self.design,
                pad_to_max_seqs=True,
                override_method="X-RAY DIFFRACTION",
                disulfide_prob=self.disulfide_prob,
                disulfide_on=self.disulfide_on,
            )

        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {pdb_id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # set chain_design_mask
        features["chain_design_mask"] = torch.from_numpy(chain_design_mask)

        # Compute template features
        templates_features = load_dummy_templates(
            tdim=1, num_tokens=len(features["res_type"])
        )
        features.update(templates_features)

        rng = np.random.default_rng(None)
        if self.ss_single_condition:
            ss_single(features, rng)
        if self.ss_double_condition:
            ss_double(features, rng)
        if self.ss_short_motif:
            ss_short_motif(features, rng, helix_prob=0.8)

        features["idx_dataset"] = torch.tensor(1)
        features["id"] = pdb_id
        if "structure" in self.extra_features:
            features["structure"] = structure
        if "tokenized" in self.extra_features:
            features["tokenized"] = tokenized
        if self.multiplicity > 1:
            features["data_sample_idx"] = data_sample_idx
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.dataset.target_ids) * self.multiplicity


class ProteinBinderDataModule(pl.LightningDataModule):
    """DataModule for BoltzGen."""

    def __init__(
        self,
        cfg: DataConfig,
        batch_size,
        num_workers,
        pin_memory,
        extra_features: Optional[List[str]] = None,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        with Path(cfg.target_ids).open("r") as f:
            regex = re.compile(r"([^_]+)(?:_(.*))?")
            target_ids = [
                regex.match(x.lower()).groups() for x in f.read().splitlines()
            ]
            chain_ids = [tup[1] for tup in target_ids]
            target_ids = [tup[0] for tup in target_ids]
            if cfg.num_targets is not None:
                target_ids = target_ids[: cfg.num_targets]
            print("split", target_ids)

        dataset = Dataset(
            struct_dir=Path(cfg.target_dir) / "structures",
            record_dir=Path(cfg.target_dir) / "records",
            target_ids=target_ids,
            seq_len=cfg.seq_len,
            tokenizer=cfg.tokenizer,
            featurizer=cfg.featurizer,
            chain_ids=chain_ids,
        )

        # Load canonical molecules
        canonicals = load_canonicals(cfg.moldir)

        self.predict_set = PredictionDataset(
            dataset=dataset,
            canonicals=canonicals,
            moldir=Path(cfg.moldir),
            backbone_only=cfg.backbone_only,
            atom14=cfg.atom14,
            atom37=cfg.atom37,
            design=cfg.design,
            target_structure_condition=cfg.target_structure_condition,
            msa_condition=cfg.msa_condition,
            ss_single_condition=cfg.ss_single_condition,
            ss_double_condition=cfg.ss_double_condition,
            ss_short_motif=cfg.ss_short_motif,
            max_seq=cfg.seq_len,
            msa_dir=Path(cfg.msa_dir),
            multiplicity=cfg.multiplicity,
            inverse_fold=cfg.inverse_fold,
            disulfide_prob=cfg.disulfide_prob,
            disulfide_on=cfg.disulfide_on,
            extra_features=extra_features,
        )

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
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
                "amino_acids_symmetries",
                "ligand_symmetries",
                "activity_name",
                "activity_qualifier",
                "sid",
                "cid",
                "normalized_protein_accession",
                "pair_id",
                "record",
                "id",
                "structure",
                "tokenized",
                "structure_bonds",
                "extra_mols",
                "data_sample_idx",
            ]:
                batch[key] = batch[key].to(device)
        return batch


if __name__ == "__main__":
    # debugging code
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import tqdm

    cfg = OmegaConf.load("configs/predict/prot.yaml")
    # cfg = OmegaConf.load("configs/predict/lig.yaml")
    # datamodule = instantiate(cfg.data)

    cfg.data.cfg.target_ids = "/data/scratch/faltings/data/bgen/hard_test_set_ids.txt"
    datamodule = instantiate(cfg.data)
    for i in tqdm.tqdm(range(len(datamodule.predict_set))):
        try:
            entry = datamodule.predict_set[i]
        except Exception as e:
            print(e)
            continue
