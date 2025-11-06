"""
Partition Coefficient Calculation Module.

This module calculates partition coefficients (LogP) and related
properties for assessing peptide lipophilicity and BBB permeability.

Key calculations:
- LogP (octanol-water partition coefficient)
- LogD (distribution coefficient at specific pH)
- Membrane-water partition coefficient (K_m/w)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import warnings


class LogPCalculator:
    """
    Calculator for LogP (octanol-water partition coefficient).
    
    LogP is a key descriptor for BBB permeability, with favorable
    values typically in the range 1-3 for passive diffusion.
    
    Parameters
    ----------
    method : str, default="fragment_based"
        Method for LogP calculation:
        - "fragment_based": sum of fragment contributions
        - "atom_based": sum of atom contributions
        - "ml_based": machine learning prediction
    """
    
    def __init__(self, method: str = "fragment_based"):
        self.method = method
        
        # Fragment-based LogP contributions (simplified)
        # In practice, use databases like Wildman-Crippen, XLogP, etc.
        self.fragment_contributions = {
            # Amino acid contributions (approximate)
            "ALA": 0.31,
            "ARG": -4.20,
            "ASN": -1.82,
            "ASP": -0.77,
            "CYS": 0.79,
            "GLN": -1.82,
            "GLU": -0.64,
            "GLY": -0.40,
            "HIS": -1.67,
            "ILE": 1.80,
            "LEU": 1.70,
            "LYS": -3.00,
            "MET": 0.64,
            "PHE": 1.79,
            "PRO": 0.72,
            "SER": -0.04,
            "THR": 0.26,
            "TRP": 0.88,
            "TYR": 0.89,
            "VAL": 1.22,
        }
    
    def calculate_logp(
        self,
        sequence: str,
        structure: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Calculate LogP for a peptide sequence.
        
        Parameters
        ----------
        sequence : str
            Peptide sequence (one-letter or three-letter codes)
        structure : np.ndarray, optional
            Peptide structure coordinates (for structure-based methods)
            
        Returns
        -------
        Dict
            LogP results:
            - 'logp': calculated LogP value
            - 'method': method used
            - 'interpretation': BBB permeability interpretation
        """
        if self.method == "fragment_based":
            logp = self._calculate_fragment_based(sequence)
        elif self.method == "atom_based":
            logp = self._calculate_atom_based(sequence, structure)
        elif self.method == "ml_based":
            logp = self._calculate_ml_based(sequence, structure)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Interpret LogP for BBB permeability
        if logp < 0:
            interpretation = "too_hydrophilic"
            permeability_likelihood = 0.1
        elif logp < 1:
            interpretation = "low_lipophilicity"
            permeability_likelihood = 0.3
        elif logp <= 3:
            interpretation = "optimal_lipophilicity"
            permeability_likelihood = 0.9
        elif logp <= 5:
            interpretation = "high_lipophilicity"
            permeability_likelihood = 0.6
        else:
            interpretation = "too_lipophilic"
            permeability_likelihood = 0.2
        
        return {
            "logp": logp,
            "method": self.method,
            "interpretation": interpretation,
            "permeability_likelihood": permeability_likelihood,
        }
    
    def _calculate_fragment_based(self, sequence: str) -> float:
        """
        Calculate LogP using fragment contributions.
        
        Sums contributions from amino acid fragments.
        """
        # Convert to three-letter codes if needed
        if len(sequence) > 0 and len(sequence[0]) == 1:
            # One-letter code - would need conversion
            # Simplified: assume three-letter codes
            pass
        
        # Split into residues (assuming space-separated three-letter codes)
        if " " in sequence:
            residues = sequence.split()
        else:
            # Assume one-letter codes
            aa_map = {
                "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP",
                "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
                "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
                "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
                "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
            }
            residues = [aa_map.get(aa, "GLY") for aa in sequence]
        
        logp = sum(self.fragment_contributions.get(res, 0.0) for res in residues)
        
        # Account for peptide bonds (typically -0.5 per bond)
        n_bonds = len(residues) - 1
        logp -= 0.5 * n_bonds
        
        return logp
    
    def _calculate_atom_based(
        self,
        sequence: str,
        structure: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate LogP using atom-based contributions.
        
        In practice, would use methods like Wildman-Crippen,
        XLogP, or similar atom-based approaches.
        """
        warnings.warn(
            "Atom-based LogP calculation is simplified. In production, "
            "use established methods like Wildman-Crippen or XLogP."
        )
        # Fallback to fragment-based
        return self._calculate_fragment_based(sequence)
    
    def _calculate_ml_based(
        self,
        sequence: str,
        structure: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate LogP using machine learning models.
        
        In practice, would use trained models or external APIs
        like SwissADME, admetSAR, etc.
        """
        warnings.warn(
            "ML-based LogP calculation is not implemented. In production, "
            "use tools like SwissADME, admetSAR, or trained models."
        )
        # Fallback to fragment-based
        return self._calculate_fragment_based(sequence)


class PartitionCoefficient:
    """
    Calculator for membrane-water partition coefficient (K_m/w).
    
    K_m/w is directly related to BBB permeability and can be estimated
    from MD simulations or calculated from LogP.
    
    Parameters
    ----------
    method : str, default="logp_based"
        Method for calculation:
        - "logp_based": estimate from LogP
        - "md_based": calculate from MD simulations
    """
    
    def __init__(self, method: str = "logp_based"):
        self.method = method
        self.logp_calculator = LogPCalculator()
    
    def calculate_kmw(
        self,
        sequence: str,
        md_trajectory: Optional[np.ndarray] = None,
        logp: Optional[float] = None,
    ) -> Dict:
        """
        Calculate membrane-water partition coefficient (K_m/w).
        
        Parameters
        ----------
        sequence : str
            Peptide sequence
        md_trajectory : np.ndarray, optional
            MD trajectory data for membrane-water distribution
        logp : float, optional
            Pre-calculated LogP value
            
        Returns
        -------
        Dict
            K_m/w results:
            - 'kmw': partition coefficient
            - 'log_kmw': log10(K_m/w)
            - 'method': method used
        """
        if self.method == "md_based" and md_trajectory is not None:
            kmw = self._calculate_from_md(md_trajectory)
        else:
            # Use LogP-based estimation
            if logp is None:
                logp_result = self.logp_calculator.calculate_logp(sequence)
                logp = logp_result["logp"]
            
            # Empirical relationship: Log(K_m/w) ≈ LogP - 0.5
            # (membrane is more polar than octanol)
            log_kmw = logp - 0.5
            kmw = 10 ** log_kmw
        
        return {
            "kmw": kmw,
            "log_kmw": np.log10(kmw) if kmw > 0 else None,
            "method": self.method,
        }
    
    def _calculate_from_md(self, trajectory: np.ndarray) -> float:
        """
        Calculate K_m/w from MD simulation data.
        
        K_m/w = C_membrane / C_water
        
        Where concentrations are estimated from time spent in each region.
        """
        # Simplified: would analyze actual MD trajectory
        # to determine time spent in membrane vs water
        warnings.warn(
            "MD-based K_m/w calculation is simplified. In production, "
            "analyze actual MD trajectories to determine membrane/water distribution."
        )
        
        # Placeholder
        return 10.0  # Example value
    
    def estimate_permeability(
        self,
        kmw: float,
        molecular_weight: Optional[float] = None,
    ) -> Dict:
        """
        Estimate BBB permeability from K_m/w.
        
        Parameters
        ----------
        kmw : float
            Membrane-water partition coefficient
        molecular_weight : float, optional
            Molecular weight in Da
            
        Returns
        -------
        Dict
            Permeability estimates:
            - 'permeability_score': estimated permeability (0-1)
            - 'interpretation': qualitative interpretation
        """
        log_kmw = np.log10(kmw) if kmw > 0 else None
        
        if log_kmw is None:
            permeability_score = 0.0
            interpretation = "unknown"
        elif log_kmw < 0:
            permeability_score = 0.1
            interpretation = "very_low"
        elif log_kmw < 0.5:
            permeability_score = 0.3
            interpretation = "low"
        elif log_kmw < 2.5:
            permeability_score = 0.7
            interpretation = "moderate"
        elif log_kmw < 4.0:
            permeability_score = 0.9
            interpretation = "high"
        else:
            permeability_score = 0.5
            interpretation = "very_high_but_may_be_too_lipophilic"
        
        # Adjust for molecular weight
        if molecular_weight is not None:
            if molecular_weight > 600:
                permeability_score *= 0.5  # Penalize large molecules
            elif molecular_weight > 500:
                permeability_score *= 0.7
        
        return {
            "permeability_score": permeability_score,
            "interpretation": interpretation,
            "log_kmw": log_kmw,
        }

