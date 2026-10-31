"""Synthetic regression tests for diffusion loss batch/sample correspondence."""
# ruff: noqa: CPY001, INP001

import pytest
import torch

from boltzgen.data import const
from boltzgen.model.modules.diffusion import AtomDiffusion


def make_case(
    batch: int, multiplicity: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    n = 5
    features = {
        "atom_resolved_mask": torch.tensor(
            [[1, 1, 1, 1, 0], [1, 0, 1, 1, 1], [1, 1, 1, 0, 1]], dtype=torch.float32
        )[:batch],
        "fake_atom_mask": torch.tensor(
            [[0, 1, 0, 1, 0], [1, 0, 1, 0, 0], [0, 0, 1, 0, 1]], dtype=torch.float32
        )[:batch],
        "atom_to_token": torch.eye(n).expand(batch, -1, -1).clone(),
        "mol_type": torch.tensor([[0, 0, 1, 2, 3], [3, 0, 2, 1, 0], [1, 0, 3, 0, 2]])[
            :batch
        ],
        "design_mask": torch.tensor(
            [[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 0, 1]], dtype=torch.float32
        )[:batch],
    }
    residue_count = len(const.res_type_weight)
    residue_index = (
        torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])[:batch]
        % residue_count
    )
    features["res_type"] = torch.nn.functional.one_hot(
        residue_index, residue_count
    ).float()
    gen = torch.Generator().manual_seed(37)
    truth = torch.randn(batch * multiplicity, n, 3, generator=gen)
    pred = torch.randn(batch * multiplicity, n, 3, generator=gen).requires_grad_()
    outputs = {
        "aligned_true_atom_coords": truth,
        "noised_atom_coords": torch.zeros_like(truth),
        "denoised_atom_coords": pred,
        "sigmas": torch.linspace(0.5, 2.0, batch * multiplicity),
    }
    return features, outputs


def create_model(*, rotational: bool) -> AtomDiffusion:
    # Avoid allocating the network; compute_loss only uses these three attributes.
    model = AtomDiffusion.__new__(AtomDiffusion)
    torch.nn.Module.__init__(model)
    model.sigma_data = 2.0
    model.mse_rotational_alignment = rotational
    model.register_buffer("zero", torch.tensor(0.0))
    return model


def single_sample_oracle(
    model: AtomDiffusion,
    feats: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    batch: int,
    multiplicity: int,
    **kwargs: float | bool,
) -> torch.Tensor:
    # Per-example execution gives every sample its own features, without any
    # vectorized repetition. The same public loss calculates alignment and masks.
    values = []
    for b in range(batch):
        one_feat = {key: value[b : b + 1] for key, value in feats.items()}
        for m in range(multiplicity):
            i = b * multiplicity + m
            one_out = {key: value[i : i + 1] for key, value in outputs.items()}
            values.append(
                model.compute_loss(one_feat, one_out, multiplicity=1, **kwargs)["loss"]
            )
    return torch.stack(values).mean()


def manual_centering_oracle(
    feats: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    batch: int,
    multiplicity: int,
    *,
    fake_weight: float,
    residue_weight: float,
) -> torch.Tensor:
    # Independent weighted centering, per-example reduction and sigma weighting.
    values = []
    for b in range(batch):
        mask = feats["atom_resolved_mask"][b]
        types = feats["mol_type"][b]
        align_w = (
            1
            + 5
            * (
                (types == const.chain_type_ids["DNA"])
                | (types == const.chain_type_ids["RNA"])
            ).float()
            + 10 * (types == const.chain_type_ids["NONPOLYMER"]).float()
        )
        residue_w = torch.tensor(const.res_type_weight)[feats["res_type"][b].argmax(-1)]
        residue_w = (
            (1 - feats["design_mask"][b]) + feats["design_mask"][b] * residue_w
        ) ** residue_weight
        weights = (
            mask
            * align_w
            * (1 + feats["fake_atom_mask"][b] * (fake_weight - 1))
            * residue_w
        )
        for m in range(multiplicity):
            i = b * multiplicity + m
            p, t = (
                outputs["denoised_atom_coords"][i],
                outputs["aligned_true_atom_coords"][i],
            )
            centroid_weight = mask * align_w
            shift = ((p - t) * centroid_weight[:, None]).sum(0) / centroid_weight.sum()
            aligned = (t + shift).detach()
            error = ((p - aligned) ** 2).sum(-1)
            mse = (weights * error).sum() / (3 * weights.sum() + 1e-5)
            sigma = outputs["sigmas"][i]
            values.append(
                mse * (sigma**2 + model_sigma_data**2) / (sigma * model_sigma_data) ** 2
            )
    return torch.stack(values).mean()


model_sigma_data = 2.0


@pytest.mark.parametrize(
    ("batch", "multiplicity"), [(1, 1), (1, 3), (2, 1), (2, 2), (2, 3), (3, 2)]
)
@pytest.mark.parametrize("rotational", [False, True])
@pytest.mark.parametrize("residue_weight", [0.0, 0.7])
def test_loss_matches_individual_examples(
    batch: int, multiplicity: int, rotational: bool, residue_weight: float
) -> None:
    feats, outputs = make_case(batch, multiplicity)
    model = create_model(rotational=rotational)
    kwargs = {
        "fake_atom_weight": 0.25,
        "residue_type_weight": residue_weight,
        "add_smooth_lddt_loss": False,
    }
    actual = model.compute_loss(feats, outputs, multiplicity=multiplicity, **kwargs)[
        "loss"
    ]
    reference = single_sample_oracle(
        model, feats, outputs, batch, multiplicity, **kwargs
    )
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
    (actual_grad,) = torch.autograd.grad(
        actual, outputs["denoised_atom_coords"], retain_graph=True
    )
    (reference_grad,) = torch.autograd.grad(
        reference, outputs["denoised_atom_coords"], retain_graph=True
    )
    torch.testing.assert_close(actual_grad, reference_grad, atol=2e-6, rtol=2e-6)
    if not rotational:
        independent = manual_centering_oracle(
            feats,
            outputs,
            batch,
            multiplicity,
            fake_weight=0.25,
            residue_weight=residue_weight,
        )
        torch.testing.assert_close(actual, independent, atol=2e-6, rtol=2e-6)
        (independent_grad,) = torch.autograd.grad(
            independent, outputs["denoised_atom_coords"]
        )
        torch.testing.assert_close(actual_grad, independent_grad, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(("batch", "multiplicity"), [(1, 1), (1, 3), (2, 1), (2, 3)])
def test_smooth_lddt_matches_individual_examples(batch: int, multiplicity: int) -> None:
    feats, outputs = make_case(batch, multiplicity)
    # Model forward has already expanded coordinates in batch-major sample order.
    feats["coords"] = outputs["aligned_true_atom_coords"]
    model = create_model(rotational=False)
    kwargs = {"fake_atom_weight": 0.25, "add_smooth_lddt_loss": True}
    actual = model.compute_loss(feats, outputs, multiplicity=multiplicity, **kwargs)
    samples = []
    for b in range(batch):
        for m in range(multiplicity):
            i = b * multiplicity + m
            one_feat = {
                key: value[b : b + 1] for key, value in feats.items() if key != "coords"
            }
            one_feat["coords"] = feats["coords"][i : i + 1]
            one_out = {key: value[i : i + 1] for key, value in outputs.items()}
            samples.append(
                model.compute_loss(one_feat, one_out, multiplicity=1, **kwargs)
            )
    for key in ("mse_loss", "smooth_lddt_loss"):
        expected = torch.stack([v["loss_breakdown"][key] for v in samples]).mean()
        torch.testing.assert_close(
            actual["loss_breakdown"][key], expected, atol=2e-6, rtol=2e-6
        )


def test_zero_weight_for_all_fake_atoms_has_zero_loss_and_gradient() -> None:
    feats, outputs = make_case(2, 3)
    feats["fake_atom_mask"] = torch.ones_like(feats["fake_atom_mask"])
    model = create_model(rotational=False)
    loss = model.compute_loss(
        feats, outputs, multiplicity=3, fake_atom_weight=0.0, add_smooth_lddt_loss=False
    )["loss"]
    torch.testing.assert_close(loss, torch.tensor(0.0))
    (grad,) = torch.autograd.grad(loss, outputs["denoised_atom_coords"])
    torch.testing.assert_close(grad, torch.zeros_like(grad))
