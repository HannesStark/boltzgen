"""
Utility functions for loading and processing Tau protein embeddings.

Provides functions to extract embeddings from protein language models
(ESM-2, ProtT5) for Tau conditioning.
"""

import torch
import numpy as np
from typing import Optional, Union
from pathlib import Path


def load_esm2_embeddings(
    sequence: str,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load ESM-2 embeddings for a Tau sequence.
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence
    model_name : str, default='facebook/esm2_t33_650M_UR50D'
        ESM-2 model name (see https://huggingface.co/facebook/esm2_t33_650M_UR50D)
        Options:
        - 'facebook/esm2_t6_8M_UR50D' (8M params)
        - 'facebook/esm2_t12_35M_UR50D' (35M params)
        - 'facebook/esm2_t33_650M_UR50D' (650M params) - default
        - 'facebook/esm2_t36_3B_UR50D' (3B params)
    device : str, default='cpu'
        Device to run model on
        
    Returns
    -------
    torch.Tensor
        Per-token embeddings (1, L, embedding_dim)
        For 650M model: (1, L, 1280)
    """
    try:
        from transformers import EsmModel, EsmTokenizer
        
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name).to(device)
        model.eval()
        
        # Tokenize sequence
        tokens = tokenizer(sequence, return_tensors="pt", padding=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state  # (1, L+2, embedding_dim)
            # Remove special tokens (CLS and SEP/EOS)
            embeddings = embeddings[:, 1:-1, :]  # (1, L, embedding_dim)
        
        return embeddings
    except ImportError:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        )


def load_prott5_embeddings(
    sequence: str,
    model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load ProtT5 embeddings for a Tau sequence.
    
    Parameters
    ----------
    sequence : str
        Amino acid sequence
    model_name : str, default='Rostlab/prot_t5_xl_half_uniref50-enc'
        ProtT5 model name
        Options:
        - 'Rostlab/prot_t5_xl_half_uniref50-enc' (encoder-only, recommended)
        - 'Rostlab/prot_t5_base_mt_uniref50' (full T5)
    device : str, default='cpu'
        Device to run model on
        
    Returns
    -------
    torch.Tensor
        Per-token embeddings (1, L, embedding_dim)
        For XL encoder: (1, L, 1024)
    """
    try:
        from transformers import T5EncoderModel, T5Tokenizer
        
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(model_name).to(device)
        model.eval()
        
        # Tokenize sequence (ProtT5 uses space-separated AA tokens)
        sequence_space = " ".join(sequence)
        tokens = tokenizer(
            sequence_space, add_special_tokens=True, padding=True, return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state  # (1, L+1, embedding_dim)
            # Remove special token (usually just one)
            embeddings = embeddings[:, 1:, :]  # (1, L, embedding_dim)
        
        return embeddings
    except ImportError:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        )


def load_precomputed_embeddings(
    filepath: Union[str, Path],
    format: str = "npy",
) -> torch.Tensor:
    """
    Load pre-computed Tau embeddings from file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to embedding file
    format : str, default='npy'
        File format: 'npy', 'pt', 'npz'
        
    Returns
    -------
    torch.Tensor
        Embeddings (1, L, embedding_dim) or (L, embedding_dim)
    """
    filepath = Path(filepath)
    
    if format == "npy":
        embeddings = np.load(filepath)
    elif format == "pt":
        embeddings = torch.load(filepath)
        if isinstance(embeddings, torch.Tensor):
            return embeddings
        else:
            embeddings = embeddings.numpy()
    elif format == "npz":
        data = np.load(filepath)
        # Assume first array is embeddings
        embeddings = data[list(data.keys())[0]]
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Convert to torch and ensure correct shape
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    
    # Ensure batch dimension
    if len(embeddings.shape) == 2:
        embeddings = embeddings.unsqueeze(0)  # (1, L, dim)
    
    return embeddings


def get_tau_embeddings(
    sequence: Optional[str] = None,
    embedding_file: Optional[Union[str, Path]] = None,
    model_type: str = "esm2",
    model_name: Optional[str] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    High-level function to get Tau embeddings.
    
    Parameters
    ----------
    sequence : str, optional
        Amino acid sequence (if computing from scratch)
    embedding_file : str or Path, optional
        Path to pre-computed embeddings
    model_type : str, default='esm2'
        Model type: 'esm2', 'prott5', or 'precomputed'
    model_name : str, optional
        Specific model name (uses defaults if None)
    device : str, default='cpu'
        Device for model computation
        
    Returns
    -------
    torch.Tensor
        Tau embeddings (1, L, embedding_dim)
    """
    if embedding_file is not None:
        return load_precomputed_embeddings(embedding_file)
    
    if sequence is None:
        raise ValueError("Either sequence or embedding_file must be provided")
    
    if model_type == "esm2":
        if model_name is None:
            model_name = "facebook/esm2_t33_650M_UR50D"
        return load_esm2_embeddings(sequence, model_name=model_name, device=device)
    elif model_type == "prott5":
        if model_name is None:
            model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
        return load_prott5_embeddings(sequence, model_name=model_name, device=device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def prepare_tau_conditioning_features(
    tau_embeddings: torch.Tensor,
    tau_mask: Optional[torch.Tensor] = None,
) -> dict:
    """
    Prepare Tau embeddings for use in model features.
    
    Parameters
    ----------
    tau_embeddings : torch.Tensor
        Tau embeddings (B, L_tau, embedding_dim) or (L_tau, embedding_dim)
    tau_mask : torch.Tensor, optional
        Mask for Tau tokens (B, L_tau) or (L_tau,)
        
    Returns
    -------
    dict
        Dictionary with 'tau_embeddings' and optionally 'tau_mask'
    """
    # Ensure batch dimension
    if len(tau_embeddings.shape) == 2:
        tau_embeddings = tau_embeddings.unsqueeze(0)  # (1, L, dim)
    
    features = {"tau_embeddings": tau_embeddings}
    
    if tau_mask is not None:
        if len(tau_mask.shape) == 1:
            tau_mask = tau_mask.unsqueeze(0)  # (1, L)
        features["tau_mask"] = tau_mask
    
    return features

