"""
BBB Permeability Scorer - Wrapper for Trained Classifier.

This module provides a convenient interface for predicting BBB permeability
of peptides using the trained MLP classifier.
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
from pathlib import Path
from typing import List, Union, Dict
import sys
sys.path.append('src')
from features.peptide_features import extract_peptide_features


class BBBClassifier(nn.Module):
    """MLP classifier for BBB permeability prediction (must match training architecture)."""

    def __init__(self, input_dim=9, hidden_dim1=64, hidden_dim2=32, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x):
        return self.network(x)


class BBBScorer:
    """
    BBB Permeability Scorer.

    Predicts blood-brain barrier permeability probability for peptides.

    Examples
    --------
    >>> scorer = BBBScorer.from_checkpoint('models/bbb_classifier.pt')
    >>> prob = scorer.predict('YGRKKRRQRRR')  # TAT peptide
    >>> print(f"P(BBB+) = {prob:.3f}")  # Should be > 0.5
    0.856

    >>> probs = scorer.predict_batch(['YGRKKRRQRRR', 'DDDEEEE'])
    >>> print(probs)
    [0.856, 0.123]
    """

    def __init__(self, model: nn.Module, scaler, feature_cols: List[str]):
        """
        Initialize BBB scorer.

        Parameters
        ----------
        model : nn.Module
            Trained BBB classifier model
        scaler : sklearn.preprocessing.StandardScaler
            Fitted feature scaler
        feature_cols : list of str
            Feature column names in correct order
        """
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.model.eval()  # Set to evaluation mode

    @classmethod
    def from_checkpoint(cls, model_path: Union[str, Path] = 'models/bbb_classifier.pt'):
        """
        Load BBB scorer from saved checkpoint.

        Parameters
        ----------
        model_path : str or Path, default='models/bbb_classifier.pt'
            Path to model checkpoint file

        Returns
        -------
        BBBScorer
            Loaded scorer instance

        Raises
        ------
        FileNotFoundError
            If model checkpoint or scaler file not found
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Recreate model
        model = BBBClassifier(
            input_dim=checkpoint['input_dim'],
            hidden_dim1=checkpoint.get('hidden_dim1', 64),
            hidden_dim2=checkpoint.get('hidden_dim2', 32),
            dropout=checkpoint.get('dropout', 0.3),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load scaler
        scaler_path = model_path.parent / 'bbb_scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        feature_cols = checkpoint['feature_cols']

        return cls(model, scaler, feature_cols)

    def predict(self, sequence: str) -> float:
        """
        Predict BBB permeability probability for a single peptide.

        Parameters
        ----------
        sequence : str
            Amino acid sequence (single-letter codes)

        Returns
        -------
        float
            Probability of BBB permeability (0-1)
            Higher values indicate higher likelihood of crossing BBB

        Examples
        --------
        >>> scorer = BBBScorer.from_checkpoint()
        >>> prob = scorer.predict('YGRKKRRQRRR')
        >>> print(f"BBB+ probability: {prob:.3f}")
        """
        # Extract features
        features = extract_peptide_features(sequence)
        feature_values = np.array([features[col.replace('feat_', '')] for col in self.feature_cols])

        # Scale features
        feature_values_scaled = self.scaler.transform(feature_values.reshape(1, -1))

        # Predict
        with torch.no_grad():
            logit = self.model(torch.FloatTensor(feature_values_scaled)).item()
            prob = 1 / (1 + np.exp(-logit))  # Sigmoid

        return prob

    def predict_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Predict BBB permeability for multiple peptides.

        Parameters
        ----------
        sequences : list of str
            List of amino acid sequences

        Returns
        -------
        np.ndarray
            Array of BBB permeability probabilities (shape: n_sequences)

        Examples
        --------
        >>> scorer = BBBScorer.from_checkpoint()
        >>> seqs = ['YGRKKRRQRRR', 'DDDEEEE', 'RQIKIWFQNRRMKWKK']
        >>> probs = scorer.predict_batch(seqs)
        >>> for seq, prob in zip(seqs, probs):
        ...     print(f"{seq}: {prob:.3f}")
        YGRKKRRQRRR: 0.856
        DDDEEEE: 0.123
        RQIKIWFQNRRMKWKK: 0.782
        """
        return np.array([self.predict(seq) for seq in sequences])

    def classify(self, sequence: str, threshold: float = 0.5) -> bool:
        """
        Classify a peptide as BBB+ or BBB-.

        Parameters
        ----------
        sequence : str
            Amino acid sequence
        threshold : float, default=0.5
            Probability threshold for classification

        Returns
        -------
        bool
            True if BBB+ (permeable), False if BBB- (non-permeable)

        Examples
        --------
        >>> scorer = BBBScorer.from_checkpoint()
        >>> scorer.classify('YGRKKRRQRRR')
        True  # BBB+
        >>> scorer.classify('DDDEEEE')
        False  # BBB-
        """
        prob = self.predict(sequence)
        return prob > threshold

    def get_features(self, sequence: str) -> Dict[str, float]:
        """
        Get feature values for a peptide.

        Useful for debugging and understanding predictions.

        Parameters
        ----------
        sequence : str
            Amino acid sequence

        Returns
        -------
        dict
            Dictionary of feature names and values

        Examples
        --------
        >>> scorer = BBBScorer.from_checkpoint()
        >>> features = scorer.get_features('YGRKKRRQRRR')
        >>> print(features['net_charge'])
        8.99
        >>> print(features['hydrophobicity_mean'])
        -3.49
        """
        features = extract_peptide_features(sequence)
        return {col.replace('feat_', ''): features[col.replace('feat_', '')]
                for col in self.feature_cols}


def demo():
    """Demonstrate BBB scorer on known peptides."""
    print("=" * 80)
    print("BBB Scorer Demo")
    print("=" * 80)

    # Load scorer
    scorer = BBBScorer.from_checkpoint('models/bbb_classifier.pt')

    # Test peptides
    test_peptides = [
        # Known BBB+ peptides (CPPs)
        ("YGRKKRRQRRR", "BBB+", "TAT peptide"),
        ("RQIKIWFQNRRMKWKK", "BBB+", "Penetratin"),
        ("GRKKRRQRRRPPQ", "BBB+", "TAT-short"),
        ("RRRRRRRRR", "BBB+", "R9"),
        ("THRPPMWSPVWP", "BBB+", "T7"),

        # Known BBB- peptides
        ("DDDEEEEDDDEEEE", "BBB-", "Acidic peptide"),
        ("SSSSSSSSSSSSSS", "BBB-", "Polar peptide"),
        ("GPAGPAGYPG", "BBB-", "Random fragment"),
        ("AAAAAAAAAAAA", "BBB-", "Poly-alanine"),
    ]

    print("\nPredictions:")
    print("-" * 80)
    print(f"{'Sequence':25s} | {'Expected':8s} | {'P(BBB+)':7s} | {'Pred':5s} | Name")
    print("-" * 80)

    for seq, expected, name in test_peptides:
        prob = scorer.predict(seq)
        prediction = "BBB+" if prob > 0.5 else "BBB-"
        correct = "✓" if prediction == expected else "✗"

        print(f"{seq:25s} | {expected:8s} | {prob:7.3f} | {prediction:5s} {correct} | {name}")

    print("\n" + "=" * 80)
    print("Feature Analysis (TAT peptide):")
    print("=" * 80)
    features = scorer.get_features("YGRKKRRQRRR")
    for name, value in features.items():
        print(f"  {name:25s}: {value:10.3f}")


if __name__ == "__main__":
    demo()
