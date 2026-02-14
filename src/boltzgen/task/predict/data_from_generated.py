from dataclasses import dataclass
from pathlib import Path
import random
import re
import warnings
from typing import Dict, List, Optional
from collections import defaultdict
from rdkit.Chem import Mol
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from boltzgen.data import const
from boltzgen.data.data import Input, Structure, Tokenized
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.mol import load_canonicals, load_molecules
from boltzgen.data.pad import pad_to_max
from boltzgen.data.parse import mmcif
from boltzgen.data.parse.pdb_parser import parse_pdb
from boltzgen.data.template.features import (
    load_dummy_templates,
)
from boltzgen.data.parse.schema import parse_redesign_yaml
from boltzgen.data.tokenize.tokenizer import Tokenizer


class DataFetchException(Exception):
    pass


@dataclass
class DataConfig:
    """Data configuration."""

    num_targets: int
    samples_per_target: int
    moldir: str
    tokenizer: Tokenizer
    featurizer: Featurizer
    batch_size: int
    num_workers: int
    pin_memory: bool
    suffix: str = ".cif"
    suffix_native: str = "_native.cif"
    suffix_metadata: str = ".npz"
    target_id_regex: str = (
        r"^(?:(?:sample\d+_|batch\d+_|rank\d+_)+)?([^_]+)(?:_[^_]+)*?(?:_(?:gen))*$"
    )
    design: bool = False
    # Featurizer args (if design is True these should match with training config):
    backbone_only: bool = False
    atom14: bool = True
    max_seqs: int = 1
    inverse_fold: bool = False
    extra_mol_dir: Optional[str] = None
    disulfide_prob: float = 1.0
    disulfide_on: bool = False
    design_mask_override: Optional[str] = None
    multiplicity: int = 1
    return_designfolding: bool = False


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
            "metadata",
            "str_gen",
            "id",
            "path",
            "native_metadata",
            "native_str_gen",
            "native_id",
            "native_path",
            "exception",
            "native_exception",
            "skip",
            "native_skip",
            "structure_bonds",
            "native_structure_bonds",
            "extra_mols",
            "native_extra_mols",
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


@dataclass(frozen=True)
class TemplateInfo:
    """TemplateInfo datatype."""

    name: str
    query_chain: str
    query_st: int
    query_en: int
    template_chain: str
    template_st: int
    template_en: int


def template_from_tokens(
    tokenized: Tokenized,
    token_mask: np.ndarray[bool],
    tdim: int = 1,
) -> dict[str, torch.Tensor]:
    """Get template features where the tokens specified in token_mask have their structure specified."""
    # Get num token
    num_tokens = len(tokenized.tokens)

    # Allocate features
    res_type = np.zeros((tdim, num_tokens), dtype=np.int64)
    frame_rot = np.zeros((tdim, num_tokens, 3, 3), dtype=np.float32)
    frame_t = np.zeros((tdim, num_tokens, 3), dtype=np.float32)
    cb_coords = np.zeros((tdim, num_tokens, 3), dtype=np.float32)
    ca_coords = np.zeros((tdim, num_tokens, 3), dtype=np.float32)
    frame_mask = np.zeros((tdim, num_tokens), dtype=np.float32)
    cb_mask = np.zeros((tdim, num_tokens), dtype=np.float32)
    template_mask = np.zeros((tdim, num_tokens), dtype=np.float32)
    query_to_template = np.zeros((tdim, num_tokens), dtype=np.int64)
    visibility_ids = np.zeros((tdim, num_tokens), dtype=np.float32)

    # Now create features per token
    template_indices = np.where(token_mask)[0]
    for token_idx in template_indices:
        token = tokenized.tokens[token_idx]
        res_type[:, token_idx] = token["res_type"]
        frame_rot[:, token_idx] = token["frame_rot"].reshape(3, 3)
        frame_t[:, token_idx] = token["frame_t"]
        cb_coords[:, token_idx] = token["disto_coords"]
        ca_coords[:, token_idx] = token["center_coords"]
        cb_mask[:, token_idx] = token["disto_mask"]
        frame_mask[:, token_idx] = token["frame_mask"]
        template_mask[:, token_idx] = 1.0
        visibility_ids[:, token_idx] = 1

    # Convert to one-hot
    res_type = torch.from_numpy(res_type)
    res_type = torch.nn.functional.one_hot(res_type, num_classes=const.num_tokens)

    return {
        "template_restype": res_type,
        "template_frame_rot": torch.from_numpy(frame_rot),
        "template_frame_t": torch.from_numpy(frame_t),
        "template_cb": torch.from_numpy(cb_coords),
        "template_ca": torch.from_numpy(ca_coords),
        "template_mask_cb": torch.from_numpy(cb_mask),
        "template_mask_frame": torch.from_numpy(frame_mask),
        "template_mask": torch.from_numpy(template_mask),
        "query_to_template": torch.from_numpy(query_to_template),
        "visibility_ids": torch.from_numpy(visibility_ids),
    }


class FromGeneratedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        generated_paths: List[Path],
        metadata_paths: List[Path],
        native_paths: List[Path],
        moldir: Path,
        canonicals: dict[str, Mol],
        tokenizer: Tokenizer,
        featurizer: Featurizer,
        return_native: bool = False,
        reference_metadata_dir: Optional[Path] = None,
        target_templates: bool = False,
        design_mask_templates: bool = False,
        compute_affinity: bool = False,
        design: bool = False,
        backbone_only: bool = False,
        atom14: bool = True,
        max_seqs: int = 1,
        inverse_fold: bool = False,
        extra_mol_dir: Optional[Path] = None,
        extra_features: Optional[List[str]] = None,
        disulfide_prob: float = 1.0,
        disulfide_on: bool = False,
        design_mask_override: Optional[str] = None,
        use_new_design_mask: bool = False,
        multiplicity: int = 1,
        return_designfolding=False,
    ) -> None:
        """
        Parameters
        ----------
        design : bool
            Set to True if this dataset is used to make predictions over (i.e. design some parts
            of the structure). Set to False if this dataset is used to only evaluate the predictions
            under the paths (i.e. no design is done).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.moldir = moldir
        self.canonicals = canonicals
        self.featurizer = featurizer
        self.metadata_paths = metadata_paths
        self.generated_paths = generated_paths
        self.native_paths = native_paths
        self.return_native = return_native
        self.reference_metadata_dir = reference_metadata_dir
        self.target_templates = target_templates
        self.design_mask_templates = design_mask_templates
        self.compute_affinity = compute_affinity
        self.design = design
        self.backbone_only = backbone_only
        self.atom14 = atom14
        self.max_seqs = max_seqs
        self.inverse_fold = inverse_fold
        self.extra_mol_dir = extra_mol_dir
        self.extra_features = (
            set(extra_features) if extra_features is not None else set()
        )
        self.disulfide_prob = disulfide_prob
        self.disulfide_on = disulfide_on
        self.design_mask_override = design_mask_override
        self.use_new_design_mask = use_new_design_mask
        self.multiplicity = multiplicity
        self.return_designfolding = return_designfolding

    def __getitem__(self, idx: int) -> Dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]

        """
        data_sample_idx = idx // len(self.generated_paths)
        idx = idx % len(self.generated_paths)

        try:
            feat = self.getitem_from_paths(
                self.metadata_paths[idx],
                self.generated_paths[idx],
                self.native_paths[idx],
            )
            if self.multiplicity > 1:
                feat["data_sample_idx"] = data_sample_idx
            return feat
        except DataFetchException:
            idx = random.randint(0, len(self) - 1)
            feat = self.getitem_from_paths(
                self.metadata_paths[idx],
                self.generated_paths[idx],
                self.native_paths[idx],
            )
            if self.multiplicity > 1:
                feat["data_sample_idx"] = data_sample_idx
            return feat

    def get_sample(self, design_dir: Path, sample_id: Optional[str] = None) -> Dict:
        metadata_path = design_dir / f"{sample_id}.npz"
        generated_path = design_dir / f"{sample_id}.cif"
        native_path = design_dir / f"{sample_id}_native.cif"
        return self.getitem_from_paths(metadata_path, generated_path, native_path)

    def getitem_from_paths(self, metadata_path, generated_path, native_path) -> Dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]

        """
        # Get metadata

        if self.reference_metadata_dir:
            reference_metadata_path = self.reference_metadata_dir / metadata_path.name
            metadata = np.load(reference_metadata_path)
        else:
            metadata = np.load(metadata_path)

        # get conditioning information from metadata
        metadata_design_mask = metadata["design_mask"]
        if self.use_new_design_mask:
            design_mask = metadata["inverse_fold_design_mask"].astype(np.float32)
        else:
            design_mask = metadata_design_mask

        ss_type = None
        if "ss_type" in metadata:
            ss_type = metadata["ss_type"]

        binding_type = None
        if "binding_type" in metadata:
            binding_type = metadata["binding_type"]

        # Per-residue amino acid constraints for inverse folding
        aa_constraint_mask = None
        if "aa_constraint_mask" in metadata:
            loaded_mask = metadata["aa_constraint_mask"]
            # Validate the loaded mask is a proper array with expected shape
            if (
                isinstance(loaded_mask, np.ndarray)
                and loaded_mask.ndim == 2
                and loaded_mask.shape[1] == 20  # 20 canonical amino acids
            ):
                aa_constraint_mask = loaded_mask
            else:
                warnings.warn(
                    f"Invalid aa_constraint_mask in NPZ: "
                    f"type={type(loaded_mask)}, shape={getattr(loaded_mask, 'shape', 'N/A')}. "
                    f"Expected ndarray with shape (N, 20). Ignoring constraints.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Get features
        feat = self.get_feat(generated_path, design_mask, ss_type, binding_type, aa_constraint_mask)

        # Get native features
        if self.return_native:
            if "native_design_mask" in metadata.keys():
                feat_native = self.get_feat(native_path, metadata["native_design_mask"])
            else:
                feat_native = self.get_feat(native_path, metadata_design_mask)

            for k, v in feat_native.items():
                feat[f"native_{k}"] = v

        return feat

    def get_feat(self, path, design_mask, ss_type=None, binding_type=None, aa_constraint_mask=None):
        # Load design
        if self.extra_mol_dir is not None:
            mols = {
                path.stem: pickle.load(path.open("rb"))
                for path in self.extra_mol_dir.glob("*.pkl")
            }
            for mol_name, mol in mols.items():
                element_counts = defaultdict(int)
                for i, atom in enumerate(mol.GetAtoms()):
                    symbol = atom.GetSymbol()
                    element_counts[symbol] += 1
                    atom_name = f"{symbol}{element_counts[symbol]}"
                    atom.SetProp("name", atom_name)

        try:
            if path.suffix == ".cif":
                structure = mmcif.parse_mmcif(
                    path, mols, moldir=self.moldir, use_original_res_idx=False
                ).data
            elif path.suffix == ".pdb":
                structure = parse_pdb(
                    path, moldir=self.moldir, use_original_res_idx=False
                ).data
            else:
                raise ValueError(f"Invalid path:{path}")  # noqa: T201
        except Exception as e:  # noqa: BLE001
            print(f"Failed to parse {path} with error {e}. Skipping.")  # noqa: T201
            raise DataFetchException() from e

        # Tokenize structure

        try:
            tokenized = self.tokenizer.tokenize(
                structure, inverse_fold=self.inverse_fold
            )
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {path} with error {e}. Skipping.")  # noqa: T201
            raise DataFetchException() from e

        # Propagate design mask to obtain chain_design_mask (True whenever something is covalently bound to any residue that is in a chain that contains a design residue).
        chain_design_mask = design_mask.astype(bool)
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

        # Extract design for refolding the design only
        if self.return_designfolding:
            residue_design_mask = np.zeros(tokenized.token_to_res.max() + 1, dtype=bool)
            np.put_along_axis(
                residue_design_mask, tokenized.token_to_res, chain_design_mask, axis=0
            )
            structure = Structure.extract_residues(structure, residue_design_mask)
            tokenized = self.tokenizer.tokenize(structure)
            design_mask = design_mask[chain_design_mask]
            chain_design_mask = chain_design_mask[chain_design_mask]

        # For inverse folding, condition even on structure selected for design
        if self.inverse_fold:
            tokenized.tokens["structure_group"] = 1

        try:
            # Try to find molecules in the dataset moldir if provided
            # Find missing ones in global moldir and check if all found
            molecules = {}
            molecules.update(self.canonicals)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(self.canonicals.keys())
            if mols is not None:
                molecules.update(mols)
            mol_names = mol_names - set(molecules.keys())
            if self.moldir is not None:
                molecules.update(load_molecules(self.moldir, mol_names))
            molecules.update(load_molecules(self.moldir, mol_names))
        except Exception as e:  # noqa: BLE001
            print(f"Molecule loading failed for {path} with error {e}. Skipping.")
            raise DataFetchException() from e

        # Set design mask for tokens. This will impact the featurization and add the atom14 features
        if self.design:
            tokenized.tokens["design_mask"] = torch.from_numpy(design_mask).bool()

        # Finalize input data
        input_data = Input(
            tokens=tokenized.tokens,
            bonds=tokenized.bonds,
            token_to_res=tokenized.token_to_res,
            structure=structure,
            msa={},
            templates=None,
        )

        # Compute features
        try:
            features = self.featurizer.process(
                input_data,
                molecules=molecules,
                random=np.random.default_rng(None),
                training=False,
                max_seqs=self.max_seqs,
                backbone_only=self.backbone_only,
                atom14=self.atom14,
                design=True,
                compute_affinity=self.compute_affinity,
                override_method="X-RAY DIFFRACTION",
                disulfide_prob=self.disulfide_prob,
                disulfide_on=self.disulfide_on,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {path} with error {e}. Skipping.")  # noqa: T201
            raise DataFetchException() from e

        # Set chain design mask
        features["chain_design_mask"] = torch.from_numpy(chain_design_mask)

        # Set conditioning variables that were set during design
        if ss_type is not None:
            features["ss_type"] = torch.from_numpy(ss_type).long()
        if binding_type is not None:
            features["binding_type"] = torch.from_numpy(binding_type).long()
        # Per-residue amino acid constraints for inverse folding
        if aa_constraint_mask is not None:
            features["aa_constraint_mask"] = torch.from_numpy(aa_constraint_mask).float()

        # If we do not want the design mask to impact the featurizer (e.g. represent atoms as atom14), we set the design mask only here.
        if not self.design:
            features["design_mask"] = torch.from_numpy(design_mask).bool()

        # set chain_design_mask
        # Override design mask for inverse folding if the part that should be inverse folded differs from the previously designed part.
        if self.design and self.design_mask_override is not None:
            msg = f"design mask being overridden with user input: {self.design_mask_override}"
            print(msg)
            new_design_mask = parse_redesign_yaml(
                Path(self.design_mask_override), tokenized
            )
            features["inverse_fold_design_mask"] = torch.from_numpy(
                new_design_mask
            ).bool()

        # Perform assertions
        if len(tokenized.tokens) != len(design_mask):
            print(
                f"WARNING: len(tokenized.tokens) [{len(tokenized.tokens)}] != len(design_mask) "
                f"[{len(design_mask)}] for {path}"
            )
            features["exception"] = True
            return features
        else:
            features["exception"] = False

        # Set templates
        if self.target_templates:
            if self.design_mask_templates:
                template_mask = ~features["design_mask"].numpy()
            else:
                template_mask = ~features["chain_design_mask"].numpy()
            templates_features = template_from_tokens(tokenized, template_mask)
        else:
            # Compute template features
            templates_features = load_dummy_templates(
                tdim=1, num_tokens=len(features["res_type"])
            )
        features.update(templates_features)

        features["affinity_token_mask"] = (
            features["mol_type"] == const.chain_type_ids["NONPOLYMER"]
        )

        # Set additional features
        features["str_gen"] = structure
        features["path"] = path
        features["id"] = path.stem
        if "structure" in self.extra_features:
            features["structure"] = structure
        if "tokenized" in self.extra_features:
            features["tokenized"] = tokenized

        return features

    def __len__(self) -> int:
        return len(self.generated_paths) * self.multiplicity


class FromGeneratedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DataConfig,
        return_native: bool = False,
        compute_affinity: bool = False,
        target_templates: bool = False,
        design_mask_templates: bool = False,
        skip_existing: bool = False,
        skip_existing_kind: str = None,
        legacy_gen_suffix: str = "_gen.cif",
        legacy_metadata_suffix: str = "_metadata.npz",
        reference_metadata_dir: Optional[Path] = None,
        design_dir: Optional[str] = None,
        extra_features: Optional[List[str]] = None,
        design_mask_override: Optional[str] = None,
        subset_target_ids: Optional[str] = None,
        skip_specific_ids: Optional[List[str]] = None,
        use_new_design_mask: bool = False,
        fail_if_no_designs: bool = False,
        output_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.return_native = return_native
        self.skip_existing = skip_existing
        self.skip_existing_kind = skip_existing_kind
        self.reference_metadata_dir = (
            Path(reference_metadata_dir) if reference_metadata_dir else None
        )

        self.legacy_gen_suffix = legacy_gen_suffix
        self.legacy_metadata_suffix = legacy_metadata_suffix
        self.compute_affinity = compute_affinity
        self.target_templates = target_templates
        self.design_mask_templates = design_mask_templates
        self.extra_features = extra_features
        self.disulfide_prob = cfg.disulfide_prob
        self.disulfide_on = cfg.disulfide_on
        self.design_mask_override = cfg.design_mask_override
        self.collate = collate
        self.fail_if_no_designs = fail_if_no_designs
        self.subset_target_ids = subset_target_ids
        self.output_dir = Path(output_dir) if output_dir else None

        if design_dir is not None:
            self.init_dataset(
                design_dir,
                skip_specific_ids=skip_specific_ids,
                extra_features=extra_features,
                use_new_design_mask=use_new_design_mask,
            )
        else:
            # Load canonical molecules
            canonicals = load_canonicals(self.cfg.moldir)

            self.predict_set = FromGeneratedDataset(
                generated_paths=[],
                metadata_paths=[],
                native_paths=[],
                canonicals=canonicals,
                moldir=Path(self.cfg.moldir),
                tokenizer=self.cfg.tokenizer,
                featurizer=self.cfg.featurizer,
                return_native=self.return_native,
                reference_metadata_dir=self.reference_metadata_dir,
                target_templates=self.target_templates,
                design_mask_templates=self.design_mask_templates,
                compute_affinity=self.compute_affinity,
                design=self.cfg.design,
                backbone_only=self.cfg.backbone_only,
                atom14=self.cfg.atom14,
                max_seqs=self.cfg.max_seqs,
                inverse_fold=self.cfg.inverse_fold,
                extra_features=self.extra_features,
                disulfide_prob=self.disulfide_prob,
                disulfide_on=self.disulfide_on,
                design_mask_override=self.design_mask_override,
                use_new_design_mask=use_new_design_mask,
                multiplicity=self.cfg.multiplicity,
                return_designfolding=self.cfg.return_designfolding,
            )

    def init_dataset(
        self,
        design_dir,
        skip_specific_ids: Optional[List[str]] = None,
        extra_features: Optional[List[str]] = None,
        use_new_design_mask: bool = False,
    ):
        print(f"Initializing FromGeneratedDataModule datasets for {design_dir}")
        design_dir = Path(design_dir)
        assert design_dir.exists(), f"Path does not exist design_dir: {design_dir}"

        # Aggregate generated structure files (.cif or .pdb) while skipping companion native/metadata files.
        generated_paths = sorted(
            p
            for p in design_dir.iterdir()
            if p.suffix in {".cif", ".pdb"}
            and "_native.cif" not in p.name
            and "_metadata.npz" not in p.name
        )
        if self.fail_if_no_designs and len(generated_paths) == 0:
            raise ValueError(f"No designs found in {design_dir}")

        # skip certain ids
        num_files_before = len(generated_paths)
        print(
            f"[Info] Number of files to process (including already processed ones): {num_files_before}"
        )

        if skip_specific_ids:
            filtered_generated_paths = [
                p
                for p in generated_paths
                if not any(prob_id in p.name for prob_id in skip_specific_ids)
            ]
            num_files_after = len(filtered_generated_paths)
            print(f"[Info] Skipped specific IDs: {skip_specific_ids}")
            print(f"[Info] Number of files after filtering: {num_files_after}")
            generated_paths = filtered_generated_paths

        if self.skip_existing:
            # Functions to map an input path to a list of output paths.
            # If all output paths exist, the input path is skipped.
            def output_path_inverse_fold(input_path):
                assert self.output_dir is not None
                return [
                    self.output_dir / f"{input_path.stem}.cif",
                    self.output_dir / f"{input_path.stem}.npz",
                ]

            def output_path_folded(input_path):
                output_dir = (
                    design_dir / const.folding_dirname
                    if self.output_dir is None
                    else self.output_dir
                )
                return [
                    output_dir / f"{input_path.stem}.npz",
                    output_dir / f"{input_path.stem}.npz",
                ]

            def output_path_design_folded(input_path):
                output_dir = (
                    design_dir / const.refold_design_cif_dirname
                    if self.output_dir is None
                    else self.output_dir
                )
                return [
                    output_dir / f"{input_path.stem}.cif",
                ]

            def output_path_affinity(input_path):
                output_dir = (
                    design_dir / const.affinity_dirname
                    if self.output_dir is None
                    else self.output_dir
                )
                return [
                    output_dir / f"{input_path.stem}.npz",
                ]

            def output_path_analyzed(input_path):
                output_dir = (
                    design_dir / const.metrics_dirname
                    if self.output_dir is None
                    else self.output_dir
                )
                return [
                    output_dir / f"data_{input_path.stem}.npz",
                    output_dir / f"metrics_{input_path.stem}.npz",
                ]

            mappings = {
                "inverse_fold": output_path_inverse_fold,
                "folded": output_path_folded,
                "design_folded": output_path_design_folded,
                "affinity": output_path_affinity,
                "analyzed": output_path_analyzed,
            }
            if self.skip_existing_kind not in mappings:
                msg = f"Invalid skip_existing_kind: {self.skip_existing_kind}. Available kinds: {list(mappings.keys())}"
                raise ValueError(msg)
            selected_mapping = mappings[self.skip_existing_kind]

            generated_paths = [
                p
                for p in generated_paths
                if not all(output_path.exists() for output_path in selected_mapping(p))
            ]
            msg = f"[Info] Skipped already {self.skip_existing_kind} IDs. Number of files after filtering: {len(generated_paths)}"
            print(msg)

        target_ids = [
            re.search(rf"{self.cfg.target_id_regex}", p.stem).group(1)
            for p in generated_paths
        ]
        target_ids = list(set(target_ids))

        if self.cfg.num_targets is not None:
            target_ids = target_ids[: self.cfg.num_targets]
            generated_paths = [
                p
                for p in generated_paths
                if re.search(rf"{self.cfg.target_id_regex}", p.stem).group(1)
                in target_ids
            ]

        filtered_paths = []
        for target_id in target_ids:
            paths_of_target = [
                p
                for p in generated_paths
                if re.search(rf"{self.cfg.target_id_regex}", p.stem).group(1)
                == target_id
            ]

            filtered_paths.extend(paths_of_target[: self.cfg.samples_per_target])

        filtered_paths2 = []
        if self.subset_target_ids is not None:
            subset_ids = [
                l.strip() for l in open(self.subset_target_ids, "r").readlines()
            ]
            for path in filtered_paths:
                if any([sid in str(path) for sid in subset_ids]):
                    filtered_paths2.append(path)
            filtered_paths = filtered_paths2

        metadata_paths = []
        native_paths = []
        # Sort the paths to make sure each subprocess (when using multiple GPUs) has the same order and the index distribution when fetching from the dataset fetches the correct paths instead of fetching the same paths multiple times.
        filtered_paths = sorted(filtered_paths)
        for path in filtered_paths:
            ext = path.suffix

            # Legacy files contain "_gen" before the extension.
            if path.stem.endswith("_gen"):
                metadata_path = path.with_name(
                    path.name.replace(f"_gen{ext}", "_metadata.npz")
                )
                native_path = path.with_name(
                    path.name.replace(f"_gen{ext}", "_native.cif")
                )
            else:
                metadata_path = path.with_suffix(".npz")
                native_path = path.with_name(f"{path.stem}_native.cif")

            if not metadata_path.exists():
                print(f"[WARNING] Path does not exist: {metadata_path}")
            metadata_paths.append(metadata_path)
            if self.return_native:
                if not native_path.exists():
                    print(f"[WARNING] Path does not exist: {native_path}")
                native_paths.append(native_path)
            else:
                native_paths.append(None)
        msg = f"Found {len(target_ids)} targets and {len(filtered_paths)} remaining designs that still need to be processed in this step."
        print(msg)

        # Load canonical molecules
        canonicals = load_canonicals(self.cfg.moldir)

        self.predict_set = FromGeneratedDataset(
            generated_paths=filtered_paths,
            metadata_paths=metadata_paths,
            native_paths=native_paths,
            canonicals=canonicals,
            moldir=Path(self.cfg.moldir),
            tokenizer=self.cfg.tokenizer,
            featurizer=self.cfg.featurizer,
            return_native=self.return_native,
            reference_metadata_dir=self.reference_metadata_dir,
            target_templates=self.target_templates,
            design_mask_templates=self.design_mask_templates,
            compute_affinity=self.compute_affinity,
            design=self.cfg.design,
            backbone_only=self.cfg.backbone_only,
            atom14=self.cfg.atom14,
            max_seqs=self.cfg.max_seqs,
            inverse_fold=self.cfg.inverse_fold,
            extra_mol_dir=design_dir / const.molecules_dirname,
            extra_features=self.extra_features,
            disulfide_prob=self.disulfide_prob,
            disulfide_on=self.disulfide_on,
            design_mask_override=self.design_mask_override,
            use_new_design_mask=use_new_design_mask,
            multiplicity=self.cfg.multiplicity,
            return_designfolding=self.cfg.return_designfolding,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_set,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

    def transfer_batch_to_device(
        self,
        batch: Dict,
        device: torch.device,
        dataloader_idx: int = 0,
    ) -> Dict:
        for key in batch:
            if key not in [
                "metadata",
                "str_gen",
                "id",
                "path",
                "native_metadata",
                "native_str_gen",
                "native_id",
                "native_path",
                "exception",
                "native_exception",
                "skip",
                "native_skip",
                "structure_bonds",
                "native_structure_bonds",
                "extra_mols",
                "native_extra_mols",
                "structure",
                "tokenized",
                "data_sample_idx",
            ]:
                batch[key] = batch[key].to(device)

        return batch
