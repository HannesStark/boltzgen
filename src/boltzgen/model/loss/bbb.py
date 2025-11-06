"""
BBB (Blood-Brain Barrier) Permeability Loss Functions.

Loss functions for training the BBB permeability prediction head.
Supports both binary classification and regression tasks.
"""

import torch
from torch import Tensor, nn
from typing import Dict, Tuple


def bbb_loss(
    output: Dict[str, Tensor],
    feats: Dict[str, Tensor],
    task_type: str = "binary",
    reduction: str = "mean",
) -> Dict[str, Tensor]:
    """
    Compute BBB permeability prediction loss.
    
    Parameters
    ----------
    output : Dict[str, Tensor]
        Model output dictionary containing BBB predictions:
        - 'bbb_logits_binary': binary classification logits
        - 'bbb_pred_value': continuous permeability score (if regression)
        - 'bbb_probability': probability of BBB permeability
    feats : Dict[str, Tensor]
        Feature dictionary containing targets:
        - 'bbb_label': binary label (0=not permeable, 1=permeable) or
          continuous permeability score
        - 'bbb_mask': optional mask indicating valid samples
    task_type : str, default='binary'
        Type of task: 'binary' (classification) or 'regression'
    reduction : str, default='mean'
        Reduction method: 'mean', 'sum', or 'none'
        
    Returns
    -------
    Dict[str, Tensor]
        Dictionary containing:
        - 'loss': total BBB loss
        - 'loss_breakdown': dictionary with individual loss components
    """
    if "bbb_label" not in feats:
        # If no BBB labels are present, return zero loss
        device = next(iter(output.values())).device
        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
        return {
            "loss": zero_loss,
            "loss_breakdown": {
                "bbb_classification_loss": zero_loss,
                "bbb_regression_loss": zero_loss,
            },
        }

    bbb_mask = feats.get("bbb_mask", None)
    bbb_label = feats["bbb_label"]  # (B,) or (B, 1)

    # Ensure label is the right shape
    if len(bbb_label.shape) > 1:
        bbb_label = bbb_label.squeeze(-1)

    loss_breakdown = {}

    if task_type == "binary" or task_type == "both":
        # Binary classification loss
        bbb_logits = output["bbb_logits_binary"]  # (B*mult, 1)
        bbb_logits = bbb_logits.squeeze(-1)  # (B*mult,)

        # Handle multiplicity: average or take best conformation
        if bbb_logits.shape[0] > bbb_label.shape[0]:
            # Multiple conformations per sample
            multiplicity = bbb_logits.shape[0] // bbb_label.shape[0]
            bbb_logits = bbb_logits.view(-1, multiplicity)
            # Take mean across conformations (alternative: take max)
            bbb_logits = bbb_logits.mean(dim=1)  # (B,)

        # Expand labels if needed
        if bbb_logits.shape[0] > bbb_label.shape[0]:
            # Should not happen after multiplicity handling
            raise ValueError("Shape mismatch after multiplicity handling")

        # Binary cross-entropy loss
        bbb_classification_loss = nn.functional.binary_cross_entropy_with_logits(
            bbb_logits,
            bbb_label.float(),
            reduction=reduction,
            weight=bbb_mask.float() if bbb_mask is not None else None,
        )

        loss_breakdown["bbb_classification_loss"] = bbb_classification_loss

    if task_type == "regression" or task_type == "both":
        # Regression loss for continuous permeability scores
        if "bbb_pred_value" in output:
            bbb_pred = output["bbb_pred_value"]  # (B*mult, 1)
            bbb_pred = bbb_pred.squeeze(-1)  # (B*mult,)

            # Handle multiplicity
            if bbb_pred.shape[0] > bbb_label.shape[0]:
                multiplicity = bbb_pred.shape[0] // bbb_label.shape[0]
                bbb_pred = bbb_pred.view(-1, multiplicity)
                bbb_pred = bbb_pred.mean(dim=1)  # (B,)

            # Mean squared error loss
            bbb_regression_loss = nn.functional.mse_loss(
                bbb_pred,
                bbb_label.float(),
                reduction=reduction,
            )

            if bbb_mask is not None:
                # Apply mask if provided
                bbb_mask_expanded = bbb_mask.float().squeeze(-1)
                if bbb_mask_expanded.shape[0] < bbb_pred.shape[0]:
                    # Handle multiplicity for mask
                    multiplicity = bbb_pred.shape[0] // bbb_mask_expanded.shape[0]
                    bbb_mask_expanded = bbb_mask_expanded.repeat_interleave(
                        multiplicity, 0
                    )
                bbb_regression_loss = (
                    bbb_regression_loss * bbb_mask_expanded
                ).sum() / (bbb_mask_expanded.sum() + 1e-7)

            loss_breakdown["bbb_regression_loss"] = bbb_regression_loss
        else:
            # No regression prediction available
            device = bbb_label.device
            loss_breakdown["bbb_regression_loss"] = torch.tensor(0.0, device=device)

    # Combine losses
    if task_type == "binary":
        total_loss = loss_breakdown["bbb_classification_loss"]
    elif task_type == "regression":
        total_loss = loss_breakdown["bbb_regression_loss"]
    else:  # both
        # Weighted combination (can be made configurable)
        total_loss = (
            loss_breakdown["bbb_classification_loss"]
            + 0.5 * loss_breakdown["bbb_regression_loss"]
        )

    return {
        "loss": total_loss,
        "loss_breakdown": loss_breakdown,
    }


def bbb_loss_fn(
    output: Dict[str, Tensor],
    feats: Dict[str, Tensor],
    task_type: str = "binary",
) -> Tensor:
    """
    Simple wrapper for BBB loss function.
    
    Parameters
    ----------
    output : Dict[str, Tensor]
        Model output dictionary
    feats : Dict[str, Tensor]
        Feature dictionary with labels
    task_type : str, default='binary'
        Task type: 'binary', 'regression', or 'both'
        
    Returns
    -------
    Tensor
        Total BBB loss
    """
    loss_dict = bbb_loss(output, feats, task_type=task_type)
    return loss_dict["loss"]

