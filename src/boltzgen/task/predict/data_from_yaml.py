from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from rdkit.Chem import Mol
from boltzgen.data import const
from boltzgen.data.data import Input
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.pad import pad_to_max
from boltzgen.data.mol import load_canonicals, load_molecules
from boltzgen.data.parse.schema import YamlDesignParser
from boltzgen.data.template.features import load_dummy_templates
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.data import Input
from boltzgen.data.select.protein import ProteinSelector


@dataclass
class DataConfig:
    """Data configuration."""

    moldir: str
    multiplicity: int
    yaml_path: Union[List[str], str]
    tokenizer: Tokenizer
    featurizer: Featurizer
    backbone_only: bool = False
    atom14: bool = False
    atom37: bool = False
    design: bool = True
    compute_affinity: bool = False
    disulfide_prob: float = 1.0
    disulfide_on: bool = False
    skip_existing: bool = False
    skip_offset: int = 0
    diffusion_samples: int = 1
    output_dir: Optional[str] = None
  
   


@dataclass
class Dataset:
    yaml_path: Union[List[str], str]
    tokenizer: Tokenizer
    featurizer: Featurizer
    multiplicity: int = 1


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
            "structure",
            "tokenized",
            "structure_bonds",
            "extra_mols",
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
        extra_features: Optional[List[str]] = None,
        design: bool = True,
        compute_affinity: bool = False,
        disulfide_prob: float = 1.0,
        disulfide_on: bool = False,
        skip_offset: int = 0,
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
        self.skip_offset = skip_offset
        path = dataset.yaml_path
        self.yaml_paths = [path] if isinstance(path, str) else path

        for path in self.yaml_paths:
            filename = Path(path).name
            if re.search(r"_\d+\.yaml$", filename):
                raise ValueError(
                    f"Illegal YAML filename for '{str(path)}': names must not end with the pattern _\\d+\\.yaml so, e.g., the ends '_001.yaml' or '_4.yaml' are not allowed."
                    "This pattern is reserved for internal file indexing. Sorry :)"
                )
            if "_native" in filename:
                raise ValueError(
                    f"Illegal YAML filename for '{str(path)}': names must not contain '_native' because this substring is reserved for native structure companions. Sorry :)"
                )
        self.extra_features = (
            set(extra_features) if extra_features is not None else set()
        )
        self.selector = (
            ProteinSelector(  
                design_neighborhood_sizes=[2, 4, 6, 8, 10, 12, 14, 16, 18],
                substructure_neighborhood_sizes=[2, 4, 6, 8, 10, 12, 24],
                structure_condition_prob=1.0,
                distance_noise_std=1,
                run_selection=True,
                specify_binding_sites=True,
                ss_condition_prob=0.1,
                select_all=False,
                chain_reindexing=False,
            )
        )
        self.design = design
        self.compute_affinity = compute_affinity
        self.disulfide_prob = disulfide_prob
        self.disulfide_on = disulfide_on

        self.mols = {}
        self.parser = YamlDesignParser(mol_dir=self.moldir)

    def __getitem__(self, idx: int) -> Dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        path = Path(self.yaml_paths[idx % len(self.yaml_paths)])
        feat = self.get_sample(path)
        data_sample_idx = idx // len(self.yaml_paths) + self.skip_offset
        if self.dataset.multiplicity > 1:
            feat["data_sample_idx"] = data_sample_idx
        return feat

    def get_sample(self, path: Path, sample_id: Optional[str] = None) -> Dict:
        # Get itemn also needs to take a smaple id as input
        parsed = self.parser.parse_yaml(
            path, mol_dir=self.moldir, mols=self.mols
        )
        structure = parsed.structure
        design_info = parsed.design_info

        # Tokenize structure
        tokenized = self.dataset.tokenizer.tokenize(structure)

        # Transfer conditioning information that is stored in tokens
        token_to_res = tokenized.token_to_res
        tokenized.tokens["design_mask"] = design_info.res_design_mask[token_to_res]
        tokenized.tokens["binding_type"] = design_info.res_binding_type[token_to_res]
        tokenized.tokens["structure_group"] = design_info.res_structure_groups[
            token_to_res
        ]
        # Transfer per-residue amino acid constraints (shape: num_tokens x 20)
        tokenized.tokens["aa_constraint_mask"] = design_info.res_aa_constraint_mask[
            token_to_res
        ]

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

        # Try to find molecules in the dataset moldir if provided
        # Find missing ones in global moldir and check if all found
        molecules = {}
        molecules.update(self.canonicals)
        mol_names = set(tokenized.tokens["res_name"].tolist())
        mol_names = mol_names - set(self.canonicals.keys())
        mol_names = mol_names - set(parsed.extra_mols.keys())
        if self.moldir is not None:
            molecules.update(load_molecules(self.moldir, mol_names))

        mol_names = mol_names - set(molecules.keys())
        molecules.update(load_molecules(self.moldir, mol_names))
        molecules.update(parsed.extra_mols)

        # Finalize input data
        input_data = Input(
            tokens=tokenized.tokens,
            bonds=tokenized.bonds,
            token_to_res=token_to_res,
            structure=structure,
            msa={},
            templates=None,
        )

        # Compute features
        features = self.dataset.featurizer.process(
            input_data,
            molecules=molecules,
            random=np.random.default_rng(None),
            training=False,
            max_seqs=1,
            backbone_only=self.backbone_only,
            atom14=self.atom14,
            atom37=self.atom37,
            design=self.design,
            override_method="X-RAY DIFFRACTION",
            compute_affinity=self.compute_affinity,
            disulfide_prob=self.disulfide_prob,
            disulfide_on=self.disulfide_on,
        )

        # transfer secondary structure conditioning
        ss_type = design_info.res_ss_types[token_to_res]
        features["ss_type"] = torch.from_numpy(ss_type).to(features["ss_type"])
        features["design_ss_mask"][ss_type != const.ss_type_ids["UNSPECIFIED"]] = 1

        # set chain_design_mask
        features["chain_design_mask"] = torch.from_numpy(chain_design_mask)

        # Compute template features
        templates_features = load_dummy_templates(
            tdim=1, num_tokens=len(features["res_type"])
        )
        features.update(templates_features)

        # set last necessary features
        features["idx_dataset"] = torch.tensor(1)

        # If a smaple id is provided then this should be the sample id instead of path.stem
        if sample_id is not None:
            features["id"] = sample_id
        else:
            features["id"] = path.stem
        if "structure" in self.extra_features:
            features["structure"] = structure
        if "tokenized" in self.extra_features:
            features["tokenized"] = tokenized

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        total = len(self.yaml_paths) * (self.dataset.multiplicity - self.skip_offset)
        return max(total, 0)


class FromYamlDataModule(pl.LightningDataModule):
    """DataModule for BoltzGen."""

    def __init__(
        self, cfg: DataConfig, batch_size, num_workers, pin_memory, extra_features=None
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()

        if cfg.skip_existing and cfg.output_dir is not None:
            design_dir = Path(cfg.output_dir)
            max_idx: int = -1
            if design_dir.exists():
                pattern = re.compile(r"_(\d+)(?:\.[^.]+)$")
                max_idx = max(
                    (
                        int(m.group(1))
                        for fp in design_dir.iterdir()
                        if fp.suffix in {".cif", ".pdb"}
                        and not any(s in fp.name for s in ("_native.cif", "_metadata.npz"))
                        for m in [pattern.search(fp.name)]
                        if m
                    ),
                    default=-1,
                )
            n_samples = getattr(cfg, "diffusion_samples", 1)
            cfg.skip_offset = (max_idx // max(n_samples, 1)) + 1 if max_idx >= 0 else 0
            
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate = collate

        dataset = Dataset(
            yaml_path=cfg.yaml_path,
            multiplicity=cfg.multiplicity,
            tokenizer=cfg.tokenizer,
            featurizer=cfg.featurizer,
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
            extra_features=extra_features,
            design=cfg.design,
            compute_affinity=cfg.compute_affinity,
            disulfide_prob=cfg.disulfide_prob,
            disulfide_on=cfg.disulfide_on,
            skip_offset=cfg.skip_offset,
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
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> Dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.

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
