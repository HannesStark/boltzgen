# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations

from math import sqrt
from math import exp
from scipy.stats import norm
import math

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn
from torch.nn import Module
from typing import Any, Dict, Optional, List

from tqdm import tqdm
import boltzgen.model.layers.initialize as init
from boltzgen.data import const
from boltzgen.model.layers.miniformer import MiniformerModule
from boltzgen.model.layers.pairformer import PairformerModule
from boltzgen.model.loss.diffusion import (
    compute_bond_loss,
    smooth_lddt_loss,
    weighted_rigid_align,
    weighted_rigid_centering,
)
from boltzgen.model.modules.encoders import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    CoordinateConditioning,
    FourierEmbedding,
    SingleConditioning,
)
from boltzgen.model.modules.transformers import (
    ConditionedTransitionBlock,
    DiffusionTransformer,
)
from boltzgen.model.modules.utils import (
    LinearNoBias,
    center,
    center_random_augmentation,
    compute_random_augmentation,
    default,
    get_autocast_device_type,
    log,
)
from scipy.stats import beta


def optionally_tqdm(iterable, use_tqdm=True, **kwargs):
    return tqdm(iterable, **kwargs) if use_tqdm else iterable


"""
b - batch
h - heads
n - residue sequence length
m - atom sequence length
nw - windowed sequence length
ts - feature dimension (single)
tz - feature dimension (pairwise)
as - feature dimension (atompair)
az - feature dimension (atompair input)
"""


class DiffusionModule(Module):
    """Algorithm 20."""

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_layers: int = 1,
        token_transformer_depth: int = 6,
        token_transformer_heads: int = 8,
        use_miniformer: bool = False,
        diffusion_pairformer_args: Dict[str, Any] = None,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        gaussian_random_3d_encoding_dim: int = 0,
        transformer_post_ln: bool = False,
        tfmr_s: Optional[int] = None,
        predict_res_type: bool = False,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing
        if tfmr_s is None:
            tfmr_s = 2 * token_s
        self.tfmr_s = tfmr_s

        # conditioning
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            tfmr_s=tfmr_s,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            gaussian_random_3d_encoding_dim=gaussian_random_3d_encoding_dim,
            transformer_post_layer_norm=transformer_post_ln,
            tfmr_s=tfmr_s,
            use_qk_norm=use_qk_norm,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(tfmr_s), LinearNoBias(tfmr_s, tfmr_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer_layers = nn.ModuleList()
        self.token_pairformer_layers = nn.ModuleList()

        self.token_transformer = DiffusionTransformer(
            dim=tfmr_s,
            dim_single_cond=tfmr_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            use_qk_norm=use_qk_norm,
        )

        self.a_norm = nn.LayerNorm(tfmr_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            tfmr_s=tfmr_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            predict_res_type=predict_res_type,
            use_qk_norm=use_qk_norm,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        r_noisy,  # Float['bm m 3']
        times,  # Float['bm 1 1']
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        if self.activation_checkpointing:
            s, normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )
        else:
            s, normed_fourier = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,  # Float['b m 3'],
            multiplicity=multiplicity,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)

        # run token level transformations
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning["token_trans_bias"].float(),
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update, res_type = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return {
            "r_update": r_update,
            "token_a": a.detach(),
            "res_type": res_type,
        }


class OutTokenFeatUpdate(Module):
    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
    ):
        super().__init__()
        self.sigma_data = sigma_data

        self.norm_next = nn.LayerNorm(2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.transition_block = ConditionedTransitionBlock(
            2 * token_s, 2 * token_s + dim_fourier
        )

    def forward(
        self,
        times,
        acc_a,
        next_a,
    ):
        next_a = self.norm_next(next_a)
        fourier_embed = self.fourier_embed(times)
        normed_fourier = (
            self.norm_fourier(fourier_embed)
            .unsqueeze(1)
            .expand(-1, next_a.shape[1], -1)
        )
        cond_a = torch.cat((acc_a, normed_fourier), dim=-1)

        acc_a = acc_a + self.transition_block(next_a, cond_a)

        return acc_a


class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args,
        num_sampling_steps: int = 5,  # number of sampling steps
        sigma_min: float = 0.0004,  # min noise level
        sigma_max: float = 160.0,  # max noise level
        sigma_data: float = 16.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std: float = 1.5,  # standard deviation of log-normal distribution from which noise is drawn for training
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        step_scale_random: list = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference=None,
        mse_rotational_alignment: bool = False,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
        second_order_correction: bool = False,
        pass_resolved_mask_diff_train: bool = False,
        sampling_schedule: str = "af3",
        noise_scale_function: str = "constant",
        step_scale_function: str = "constant",
        min_noise_scale: float = 1.0,
        max_noise_scale: float = 1.0,
        noise_scale_alpha: float = 1.0,
        noise_scale_beta: float = 1.0,
        min_step_scale: float = 1.0,
        max_step_scale: float = 1.0,
        step_scale_alpha: float = 1.0,
        step_scale_beta: float = 1.0,
        time_dilation: float = 1.0,
        time_dilation_start: float = 0.6,
        time_dilation_end: float = 0.8,
        pred_threshold: Optional[float] = None,
    ):
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std

        if pred_threshold is None:
            # disable nucleation mask
            self.pred_sigma_thresh = float("inf")
        else:
            q = norm.ppf(pred_threshold)
            self.pred_sigma_thresh = self.sigma_data * exp(self.P_mean + self.P_std * q)

        self.num_sampling_steps = num_sampling_steps
        self.sampling_schedule = sampling_schedule
        self.time_dilation = time_dilation
        self.time_dilation_start = time_dilation_start
        self.time_dilation_end = time_dilation_end
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.noise_scale_function = noise_scale_function
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale
        self.noise_scale_alpha = noise_scale_alpha
        self.noise_scale_beta = noise_scale_beta
        self.step_scale = step_scale
        self.step_scale_function = step_scale_function
        self.min_step_scale = min_step_scale
        self.max_step_scale = max_step_scale
        self.step_scale_alpha = step_scale_alpha
        self.step_scale_beta = step_scale_beta
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.mse_rotational_alignment = mse_rotational_alignment
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas
        self.second_order_correction = second_order_correction
        self.pass_resolved_mask_diff_train = pass_resolved_mask_diff_train
        self.token_s = score_model_args["token_s"]

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return (
            log(sigma / self.sigma_data) * 0.25
        )  # note here the AF3 authors divide by sigma_data but not EDM

    def preconditioned_network_forward(
        self,
        noised_atom_coords,  #: Float['b m 3'],
        sigma,  #: Float['b'] | Float[' '] | float,
        network_condition_kwargs: dict,
        training: bool = True,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        if training and self.pass_resolved_mask_diff_train:
            res_mask = (
                network_condition_kwargs["feats"]["atom_resolved_mask"]
                .unsqueeze(-1)
                .float()
            )
            noised_atom_coords = noised_atom_coords * res_mask.repeat_interleave(
                network_condition_kwargs["multiplicity"], 0
            )

        net_out = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * net_out["r_update"]
        )

        return denoised_coords, net_out

    def sample_schedule_af3(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data  # note: done by AF3 but not by EDM

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample_schedule_dilated(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        ts = steps / (num_sampling_steps - 1)

        # remap to dilate a particular interval
        def dilate(ts, start, end, dilation):
            x = end - start
            l = start
            u = 1 - end
            assert (dilation - 1) * x <= l + u, "dilation too large"

            inv_dilation = 1 / dilation
            ratio = (l + u + (1 - dilation) * x) / (l + u)
            inv_ratio = 1 / ratio
            lprime = l * ratio
            uprime = u * ratio
            xprime = x * dilation

            lower_third = ts * inv_ratio
            middle_third = (ts - lprime) * inv_dilation + l
            upper_third = (ts - (lprime + xprime)) * inv_ratio + l + x
            return (
                (ts < lprime) * lower_third
                + ((ts >= lprime) & (ts < lprime + xprime)) * middle_third
                + (ts >= lprime + xprime) * upper_third
            )

        dilated_ts = dilate(
            ts, self.time_dilation_start, self.time_dilation_end, self.time_dilation
        )
        sigmas = (
            self.sigma_max**inv_rho
            + dilated_ts * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data  # note: done by AF3 but not by EDM

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def beta_noise_scale_schedule(self, num_sampling_steps):
        t = np.linspace(0, 1, num_sampling_steps)
        beta_cdf_weights = torch.from_numpy(
            beta.cdf(1 - t, self.noise_scale_alpha, self.noise_scale_beta)
        )
        return (
            self.max_noise_scale
            + (self.min_noise_scale - self.max_noise_scale) * beta_cdf_weights
        )

    def beta_step_scale_schedule(self, num_sampling_steps=None):
        t = np.linspace(0, 1, num_sampling_steps)
        beta_cdf_weights = torch.from_numpy(
            beta.cdf(t, self.step_scale_alpha, self.step_scale_beta)
        )
        return (
            self.min_step_scale
            + (self.max_step_scale - self.min_step_scale) * beta_cdf_weights
        )

    def sample(
        self,
        atom_mask,  #: Bool['b m'] | None = None,
        num_sampling_steps=None,
        multiplicity=1,
        step_scale=None,
        noise_scale=None,
        inference_logging=False,
        **network_condition_kwargs,
    ):
        if self.training and self.step_scale_random is not None:
            step_scales = np.random.choice(self.step_scale_random) * torch.ones(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
        elif self.step_scale_function == "beta":
            step_scales = self.beta_step_scale_schedule(num_sampling_steps)
        else:
            step_scales = default(step_scale, self.step_scale) * torch.ones(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
        if self.noise_scale_function == "constant":
            noise_scales = default(noise_scale, self.noise_scale) * torch.ones(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
        elif self.noise_scale_function == "beta":
            noise_scales = self.beta_noise_scale_schedule(num_sampling_steps)
        else:
            raise ValueError(
                f"Invalid noise scale schedule: {self.noise_scale_function}"
            )
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        if self.sampling_schedule == "af3":
            sigmas = self.sample_schedule_af3(num_sampling_steps)
        elif self.sampling_schedule == "dilated":
            sigmas = self.sample_schedule_dilated(num_sampling_steps)

        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_gammas_ss_ns = list(
            zip(
                sigmas[:-1],
                sigmas[1:],
                gammas[1:],
                step_scales,
                noise_scales,
            )
        )

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        feats = network_condition_kwargs["feats"]

        # gradually denoise
        coords_traj = [atom_coords]
        x0_coords_traj = []
        for step_idx, (
            sigma_tm,
            sigma_t,
            gamma,
            step_scale,
            noise_scale,
        ) in optionally_tqdm(
            enumerate(sigmas_gammas_ss_ns),
            use_tqdm=inference_logging,
            desc="Denoising steps.",
        ):
            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()
            # sigma_tm is sigma_t-1 and sigma_t is sigma_t
            t_hat = sigma_tm * (1 + gamma)
            noise_var = noise_scale**2 * (t_hat**2 - sigma_tm**2)

            atom_coords = center(atom_coords, atom_mask)

            if self.coordinate_augmentation_inference:
                random_R, random_tr = compute_random_augmentation(
                    multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
                )
                atom_coords = (
                    torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
                )

            eps = noise_scale * sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised, net_out = self.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        **network_condition_kwargs,
                    ),
                )

            if self.alignment_reverse_diff:
                with torch.autocast(get_autocast_device_type(), enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            # note here I believe there is a mistake in the AF3 paper where they use atom_coords instead of atom_coords_noisy
            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            coords_traj.append(atom_coords_next)
            x0_coords_traj.append(atom_coords_denoised)
            atom_coords = atom_coords_next
        coords_traj.append(atom_coords)

        result = dict(
            sample_atom_coords=atom_coords,
            coords_traj=coords_traj,
            x0_coords_traj=x0_coords_traj,
        )

        return result

    # training
    def loss_weight(self, sigma):
        # note: in AF3 there is a + at denominator while in EDM a *, we think this is a mistake in the paper
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        # note: in AF3 the sample is scaled by sigma_data while in EDM it is not
        # in practice this just means scaling P_mean by the log

        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        # training diffusion step
        batch_size = feats["coords"].shape[0] // multiplicity
        atom_coords = feats["coords"]
        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)

        padded_sigmas = rearrange(sigmas, "b -> b 1 1")
        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise
        # alphas=1. in paper

        denoised_atom_coords, net_out = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            training=True,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": multiplicity,
                "diffusion_conditioning": diffusion_conditioning,
            },
        )

        out_dict = {
            "noised_atom_coords": noised_atom_coords,
            "denoised_atom_coords": denoised_atom_coords,
            "sigmas": sigmas,
            "aligned_true_atom_coords": atom_coords,
        }
        out_dict.update(net_out)

        return out_dict

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        add_bond_loss=False,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        fake_atom_weight=1.0,
        residue_type_weight=0.0,
        multiplicity=1,
    ):
        with torch.autocast(get_autocast_device_type(), enabled=False):
            denoised_atom_coords = out_dict["denoised_atom_coords"].float()
            noised_atom_coords = out_dict["noised_atom_coords"].float()
            sigmas = out_dict["sigmas"].float()

            resolved_atom_mask_uni = feats["atom_resolved_mask"].float()

            resolved_atom_mask = resolved_atom_mask_uni.repeat_interleave(
                multiplicity, 0
            )

            # fake atom weighting
            fake_atom_mask = feats["fake_atom_mask"]
            fake_atom_weight = (1 - fake_atom_mask) + fake_atom_mask * fake_atom_weight

            # residue type weighting.
            if residue_type_weight > 0.0:
                design_atom_mask = torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["design_mask"].float().unsqueeze(-1),
                ).squeeze(-1)
                _res_type_weight = torch.tensor(
                    const.res_type_weight, device=denoised_atom_coords.device
                )
                _res_type_weight = torch.bmm(
                    feats["atom_to_token"].float(),
                    (feats["res_type"].float() @ _res_type_weight)
                    .unsqueeze(-1)
                    .float(),
                ).squeeze(-1)
                res_type_weight = (
                    1.0 - design_atom_mask
                ) + design_atom_mask * _res_type_weight
                res_type_weight = res_type_weight**residue_type_weight
            else:
                res_type_weight = 1.0

            align_weights = noised_atom_coords.new_ones(noised_atom_coords.shape[:2])
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

            align_weights = (
                align_weights
                * (
                    1
                    + nucleotide_loss_weight
                    * (
                        torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                        + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                    )
                    + ligand_loss_weight
                    * torch.eq(
                        atom_type_mult, const.chain_type_ids["NONPOLYMER"]
                    ).float()
                ).float()
            )

            atom_coords = out_dict["aligned_true_atom_coords"].float()
            if self.mse_rotational_alignment:
                atom_coords_aligned_ground_truth = weighted_rigid_align(
                    atom_coords.detach(),
                    denoised_atom_coords.detach(),
                    align_weights.detach(),
                    mask=feats["atom_resolved_mask"]
                    .float()
                    .repeat_interleave(multiplicity, 0)
                    .detach(),
                )
            else:
                atom_coords_aligned_ground_truth = weighted_rigid_centering(
                    atom_coords,
                    denoised_atom_coords,
                    align_weights,
                    mask=feats["atom_resolved_mask"]
                    .float()
                    .repeat_interleave(multiplicity, 0),
                )

            # Cast back
            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )

            # weighted MSE loss of denoised atom positions
            mse_loss = (
                (denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2
            ).sum(dim=-1)
            mse_loss = torch.sum(
                mse_loss
                * align_weights
                * fake_atom_weight
                * res_type_weight
                * resolved_atom_mask,
                dim=-1,
            ) / (
                torch.sum(
                    3
                    * align_weights
                    * fake_atom_weight
                    * res_type_weight
                    * resolved_atom_mask,
                    dim=-1,
                )
                + 1e-5
            )
            # weight by sigma factor
            loss_weights = self.loss_weight(sigmas)
            mse_loss = (mse_loss * loss_weights).mean()

            total_loss = mse_loss

            if add_bond_loss:
                bond_loss, num_bonds = compute_bond_loss(
                    pred_atom_coords=out_dict["sample_atom_coords"].float(),
                    true_coords=atom_coords_aligned_ground_truth,
                    feats=feats,
                )
                total_loss += bond_loss
            else:
                bond_loss = self.zero

            # proposed auxiliary smooth lddt loss
            lddt_loss = self.zero
            if add_smooth_lddt_loss:
                lddt_loss = smooth_lddt_loss(
                    denoised_atom_coords,
                    feats["coords"],
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=resolved_atom_mask_uni,
                    multiplicity=multiplicity,
                )

                total_loss = total_loss + lddt_loss

            loss_breakdown = {
                "mse_loss": mse_loss,
                "bond_loss": bond_loss,
                "smooth_lddt_loss": lddt_loss,
            }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}
