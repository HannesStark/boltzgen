# Copyright 2026 Biohub. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned ESM sampler with the public Anthropic kit's scalar-sync removal.

Modified from the Apache-2.0-licensed layers.py in esm 3.4.1.post1.
See NOTICE.md and LICENSE.anthropic for the license text. The schedule, RNG
draws and numerical operations are unchanged; only loop-invariant GPU-to-host
transfers move.
"""

import torch
from torch import Tensor
import torch.nn.functional as F


@torch.inference_mode()
def sample_without_scalar_sync(
    self,
    z_trunk: Tensor,
    s_inputs: Tensor,
    s_trunk: Tensor | None,
    relative_position_encoding: Tensor,
    ref_pos: Tensor,
    ref_charge: Tensor,
    ref_mask: Tensor,
    ref_element: Tensor,
    ref_atom_name_chars: Tensor,
    ref_space_uid: Tensor,
    tok_idx: Tensor,
    asym_id: Tensor,
    residue_index: Tensor,
    entity_id: Tensor,
    token_index: Tensor,
    sym_id: Tensor,
    token_attention_mask: Tensor | None = None,
    num_diffusion_samples: int = 1,
    num_sampling_steps: int | None = None,
    max_inference_sigma: float | None = 256.0,
    noise_scale: float | None = None,
    step_scale: float | None = None,
    return_atom_repr: bool = False,
    use_inference_cache: bool = True,
) -> dict[str, Tensor | None]:
    """Diffusion sampling (Algorithm 18).

    ``num_sampling_steps`` sets the native schedule length before applying
    ``max_inference_sigma``. The cap removes high-sigma entries, so fewer
    denoising steps may run. This adapter preserves that native behavior.
    """
    n_atoms = tok_idx.shape[1]
    device = s_inputs.device
    target_batch = s_inputs.shape[0] * num_diffusion_samples

    inference_cache: dict[str, Tensor] | None = {} if use_inference_cache else None

    steps = (
        self.inference_num_steps
        if num_sampling_steps is None
        else int(num_sampling_steps)
    )

    schedule = self.inference_noise_schedule(steps, device)
    if max_inference_sigma is not None:
        schedule = schedule[schedule <= float(max_inference_sigma)]
        schedule = F.pad(schedule, (1, 0), value=float(max_inference_sigma))

    lam = self.noise_scale if noise_scale is None else float(noise_scale)
    eta = self.step_scale if step_scale is None else float(step_scale)

    x = schedule[0] * torch.randn(
        target_batch, n_atoms, 3, device=device, dtype=torch.float32
    )
    atom_mask = ref_mask.repeat_interleave(num_diffusion_samples, 0).float()

    gammas = torch.where(
        schedule > self.gamma_min,
        torch.full_like(schedule, self.gamma_0),
        torch.zeros_like(schedule),
    )

    x_denoised_prev: Tensor | None = None
    token_repr: Tensor | None = None
    diff_atom_intermediates: Tensor | None = None

    # Transfer the schedule once instead of synchronizing three GPU scalars
    # in every denoising step (Anthropic kit U2). Preserve the RNG/math order.
    schedule_values, gamma_values = schedule.tolist(), gammas.tolist()
    step_pairs = list(zip(schedule_values[:-1], schedule_values[1:], gamma_values[1:]))
    num_steps = len(step_pairs)

    for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(step_pairs):
        x, x_denoised_prev = self._center_random_augmentation(
            x, atom_mask, second_coords=x_denoised_prev
        )

        sigma_tm_val = float(sigma_tm)
        t_hat_val = sigma_tm_val * (1.0 + float(gamma))
        eps_std = lam * max(t_hat_val**2 - sigma_tm_val**2, 0.0) ** 0.5
        x_noisy = x + eps_std * torch.randn_like(x)

        request_atom_repr = return_atom_repr and step_idx == num_steps - 1

        dm_out = self.diffusion_module(
            x_noisy=x_noisy,
            t_hat=torch.full(
                (target_batch,), t_hat_val, device=device, dtype=torch.float32
            ),
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_mask=ref_mask,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            ref_space_uid=ref_space_uid,
            tok_idx=tok_idx,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            token_index=token_index,
            sym_id=sym_id,
            token_attention_mask=token_attention_mask,
            num_diffusion_samples=num_diffusion_samples,
            return_token_repr=True,
            return_atom_repr=request_atom_repr,
            inference_cache=inference_cache,
        )

        x_denoised = dm_out["x_denoised"]
        token_repr = dm_out["token_repr"]
        if request_atom_repr:
            diff_atom_intermediates = dm_out.get("atom_intermediates")

        # Reverse diffusion alignment (Kabsch)
        with torch.autocast(device_type="cuda", enabled=False):
            x_noisy = self._weighted_rigid_align(
                x_noisy.float(), x_denoised.float(), atom_mask, atom_mask
            )
        x_noisy = x_noisy.to(dtype=x_denoised.dtype)

        # ODE/SDE step
        sigma_t_val = float(sigma_t)
        denoised_over_sigma = (x_noisy - x_denoised) / t_hat_val
        x = x_noisy + eta * (sigma_t_val - t_hat_val) * denoised_over_sigma

        x_denoised_prev = x_denoised

    result: dict[str, Tensor | None] = {
        "sample_atom_coords": x,
        "diff_token_repr": token_repr,
    }
    if return_atom_repr:
        result["diff_atom_intermediates"] = diff_atom_intermediates
    return result
