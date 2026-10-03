"""
Replacement functions for torch_scatter operations using native PyTorch APIs.
"""
import torch
from torch import Tensor
from typing import Optional


def scatter_sum(
    src: Tensor, 
    index: Tensor, 
    dim: int = -1,
    dim_size: Optional[int] = None
) -> Tensor:
    """
    Replacement for torch_scatter.scatter_sum using native PyTorch.
    
    Args:
        src: Source tensor to scatter
        index: Index tensor for scattering
        dim: Dimension along which to scatter
        dim_size: Optional size of the output dimension
        
    Returns:
        Scattered sum tensor
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    
    # Create output shape
    shape = list(src.shape)
    shape[dim] = dim_size
    
    # Initialize output tensor with zeros
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    
    # Expand index to match src dimensions
    expanded_index = index
    for _ in range(len(src.shape) - len(index.shape)):
        expanded_index = expanded_index.unsqueeze(-1)
    expanded_index = expanded_index.expand_as(src)
    
    # Use scatter_add_ for summation
    out.scatter_add_(dim, expanded_index, src)
    
    return out


def scatter_softmax(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None
) -> Tensor:
    """
    Replacement for torch_scatter.scatter_softmax using native PyTorch.
    
    Args:
        src: Source tensor to apply softmax
        index: Index tensor for grouping
        dim: Dimension along which to apply softmax
        dim_size: Optional size of the output dimension
        
    Returns:
        Tensor with softmax applied within each group
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    
    # Get max values for numerical stability
    max_value_per_index = scatter_max(src, index, dim, dim_size)
    
    # Expand index to match src dimensions for gather
    expanded_index = index
    for _ in range(len(src.shape) - len(index.shape)):
        expanded_index = expanded_index.unsqueeze(-1)
    expanded_index = expanded_index.expand_as(src)
    
    max_src = max_value_per_index.gather(dim, expanded_index)
    
    # Compute exp(src - max)
    exp_src = torch.exp(src - max_src)
    
    # Sum exp values per group
    sum_exp = scatter_sum(exp_src, index, dim, dim_size)
    
    # Gather sum for each element
    sum_exp_per_src = sum_exp.gather(dim, expanded_index)
    
    # Compute softmax
    out = exp_src / (sum_exp_per_src + 1e-10)  # Add small epsilon for numerical stability
    
    return out


def scatter_max(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None
) -> Tensor:
    """
    Helper function to compute scatter max using native PyTorch.
    
    Args:
        src: Source tensor
        index: Index tensor for grouping
        dim: Dimension along which to compute max
        dim_size: Optional size of the output dimension
        
    Returns:
        Tensor with max values per group
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Create output shape
    shape = list(src.shape)
    shape[dim] = dim_size
    
    # Initialize with -inf for max operation
    out = torch.full(shape, float('-inf'), dtype=src.dtype, device=src.device)
    
    # Create expanded index for scatter
    expanded_index = index
    for _ in range(len(src.shape) - len(index.shape)):
        expanded_index = expanded_index.unsqueeze(-1)
    expanded_index = expanded_index.expand_as(src)
    
    # aten::scatter_reduce is not implemented natively by torch-npu. Compute
    # the same grouped maximum from native tensor operations.
    src_moved = src.movedim(dim, 0)
    idx_moved = expanded_index.movedim(dim, 0)
    neg_inf = torch.tensor(float("-inf"), dtype=src.dtype, device=src.device)
    for group in range(out.shape[dim]):
        mask = idx_moved == group
        out.select(dim, group).copy_(
            torch.where(mask, src_moved, neg_inf).amax(dim=0)
        )
    
    # Replace -inf with 0 for indices that were never written to
    out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    
    return out
