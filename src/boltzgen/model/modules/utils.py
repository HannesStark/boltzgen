# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import (
    Linear,
    Module,
)
from torch.types import Device

LinearNoBias = partial(Linear, bias=False)


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


class SwiGLU(Module):
    def forward(
        self,
        x,  #: Float['... d']
    ):  # -> Float[' ... (d//2)']:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


def center(atom_coords, atom_mask):
    atom_mean = torch.sum(
        atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
    ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
    atom_coords = atom_coords - atom_mean
    return atom_coords


def compute_random_augmentation(
    multiplicity,
    s_trans=1.0,
    device=None,
    dtype=torch.float32
):
    R = random_rotations(multiplicity, dtype=dtype, device=device)
    random_trans = torch.randn((multiplicity, 1, 3), dtype=dtype, device=device) * s_trans
    return R, random_trans


def randomly_rotate(coords, return_second_coords=False, second_coords=None):
    R = random_rotations(len(coords), coords.dtype, coords.device)

    if return_second_coords:
        return torch.einsum("bmd,bds->bms", coords, R), torch.einsum(
            "bmd,bds->bms", second_coords, R
        ) if second_coords is not None else None

    return torch.einsum("bmd,bds->bms", coords, R)


def center_random_augmentation(
    atom_coords,
    atom_mask,
    s_trans=1.0,
    augmentation=True,
    centering=True,
    return_second_coords=False,
    second_coords=None,
):
    """Algorithm 19"""
    if centering:
        atom_mean = torch.sum(
            atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
        ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
        atom_coords = atom_coords - atom_mean

        if second_coords is not None:
            # apply same transformation also to this input
            second_coords = second_coords - atom_mean

    if augmentation:
        atom_coords, second_coords = randomly_rotate(
            atom_coords, return_second_coords=True, second_coords=second_coords
        )
        random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans

        if second_coords is not None:
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords

    return atom_coords


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist - self.offset.view(1, 1, 1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class GaussianRandom3DEncodings(torch.nn.Module):
    def __init__(self, dim=50):
        super().__init__()
        center = torch.randn((dim, 3))
        std = torch.rand(dim)
        self.dim = dim
        self.register_buffer("center", center)
        self.register_buffer("std", std)

    def forward(self, coords):
        B, N, _ = coords.shape
        dist2 = (
            (coords.view(B, N, 1, 3) - self.center.view(1, 1, self.dim, 3)) ** 2
        ).sum(dim=-1)
        emb = torch.exp(-dist2 / (2 * self.std.view(1, 1, self.dim)))
        return emb


# the following is copied from Torch3D, BSD License, Copyright (c) Meta Platforms, Inc. and affiliates.


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns
    -------
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns
    -------
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns
    -------
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns
    -------
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)

def get_autocast_device_type() -> str:
    """Infer device type from system availability for torch.autocast.

    Returns
    -------
        The device type for torch.autocast.
    """
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_type = "xpu"
    elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device_type = "cuda"
    else:
        device_type = "cpu"
    return device_type
