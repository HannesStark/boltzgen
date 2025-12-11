from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from boltzgen.model.modules.utils import get_autocast_device_type
    
def res_type_loss_fn(
    output: Dict[str, Tensor],
    feats: Dict[str, Tensor],
) -> Tensor:
    """Compute the res_type loss.

    Parameters
    ----------
    output : Dict[str, Tensor]
        Output of the model
    feats : Dict[str, Tensor]
        Input features

    Returns
    -------
    Tensor
        The globally averaged loss.
    """
    with torch.autocast(get_autocast_device_type(), enabled=False):
        pred = output["res_type"]
        true = feats["res_type"].float()
        multiplicity = pred.shape[0] // true.shape[0]
        assert multiplicity == 1
        true = true.repeat_interleave(multiplicity, 0)

        if "valid_mask" in output:
            valid_mask = output["valid_mask"]
            loss_mask = valid_mask * feats["design_mask"]
        else:
            loss_mask = feats["token_pad_mask"] * feats["design_mask"]
        loss_weight = loss_mask
        loss_weight = loss_weight.repeat_interleave(multiplicity, 0)
        assert loss_weight.sum() > 0, "loss_weight will cause a divide by 0"



        pred_flat = pred.reshape(-1, pred.shape[-1])
        true_flat = true.reshape(-1, true.shape[-1])

        loss = F.cross_entropy(pred_flat, true_flat, reduction="none")
        loss = loss.reshape(pred.shape[0], -1)
        loss = (loss * loss_weight).sum() / loss_weight.sum()

        # get acc
        correct_sum = ((pred.argmax(dim=-1) == true.argmax(dim=-1))*loss_weight).sum()
        total_sum = loss_weight.sum()
        acc = correct_sum / total_sum if total_sum > 0 else torch.tensor(0.0, device=loss.device)

        return loss, acc
