"""
Structural Property Analysis Module.

This module calculates structural properties relevant for BBB permeability:
- Polar Surface Area (PSA)
- Molecular weight
- Hydrogen bond donors/acceptors (HBA/HBD)
- Net charge
- Flexibility/rigidity
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import warnings


class PSACalculator:
    """
    Calculator for Polar Surface Area (PSA).
    
    PSA is a key descriptor for BBB permeability. Favorable values
    are typically < 90 Å² for passive diffusion.
    
    Parameters
    ----------
    method : str, default="geometric"
        Method for PSA calculation:
        - "geometric": geometric surface area calculation
        - "topological": topological PSA (tPSA)
    """
    
    def __init__(self, method: str = "geometric"):
        self.method = method
        
        # Topological PSA contributions (approximate, in Å²)
        # Based on Ertl et al. (2000) J. Med. Chem.
        self.tpsa_contributions = {
            # Amino acid side chains
            "ALA": 0.0,
            "ARG": 75.1,  # guanidine
            "ASN": 43.7,  # amide
            "ASP": 54.4,  # carboxyl
            "CYS": 0.0,
            "GLN": 43.7,  # amide
            "GLU": 54.4,  # carboxyl
            "GLY": 0.0,
            "HIS": 30.9,  # imidazole
            "ILE": 0.0,
            "LEU": 0.0,
            "LYS": 3.2,   # amine
            "MET": 0.0,
            "PHE": 0.0,
            "PRO": 0.0,
            "SER": 20.2,  # hydroxyl
            "THR": 20.2,  # hydroxyl
            "TRP": 15.8,  # indole N
            "TYR": 20.2,  # hydroxyl
            "VAL": 0.0,
            # Backbone contributions
            "peptide_bond": 23.5,  # amide
            "n_terminal": 3.2,     # amine
            "c_terminal": 54.4,    # carboxyl
        }
    
    def calculate_psa(
        self,
        sequence: str,
        structure: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Calculate Polar Surface Area (PSA).
        
        Parameters
        ----------
        sequence : str
            Peptide sequence
        structure : np.ndarray, optional
            Peptide structure coordinates (for geometric method)
            
        Returns
        -------
        Dict
            PSA results:
            - 'psa': PSA value in Å²
            - 'method': method used
            - 'interpretation': BBB permeability interpretation
        """
        if self.method == "geometric" and structure is not None:
            psa = self._calculate_geometric_psa(structure)
        else:
            psa = self._calculate_topological_psa(sequence)
        
        # Interpret PSA for BBB permeability
        if psa < 60:
            interpretation = "excellent_permeability"
            permeability_likelihood = 0.95
        elif psa < 90:
            interpretation = "good_permeability"
            permeability_likelihood = 0.8
        elif psa < 120:
            interpretation = "moderate_permeability"
            permeability_likelihood = 0.5
        elif psa < 150:
            interpretation = "poor_permeability"
            permeability_likelihood = 0.2
        else:
            interpretation = "very_poor_permeability"
            permeability_likelihood = 0.05
        
        return {
            "psa": psa,
            "method": self.method,
            "interpretation": interpretation,
            "permeability_likelihood": permeability_likelihood,
        }
    
    def _calculate_topological_psa(self, sequence: str) -> float:
        """
        Calculate topological PSA (tPSA).
        
        Sums contributions from polar atoms/groups.
        """
        # Convert to three-letter codes if needed
        if len(sequence) > 0 and len(sequence[0]) == 1:
            aa_map = {
                "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP",
                "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
                "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
                "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
                "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
            }
            residues = [aa_map.get(aa, "GLY") for aa in sequence]
        else:
            residues = sequence.split() if " " in sequence else [sequence]
        
        # Sum contributions
        psa = 0.0
        psa += self.tpsa_contributions.get("n_terminal", 0.0)
        psa += self.tpsa_contributions.get("c_terminal", 0.0)
        
        for res in residues:
            psa += self.tpsa_contributions.get(res, 0.0)
            psa += self.tpsa_contributions.get("peptide_bond", 0.0)
        
        # Subtract one peptide bond (already counted in terminal)
        psa -= self.tpsa_contributions.get("peptide_bond", 0.0)
        
        return psa
    
    def _calculate_geometric_psa(self, structure: np.ndarray) -> float:
        """
        Calculate PSA from geometric surface area.
        
        In practice, would use tools like MSMS, PyMOL, or similar
        to calculate solvent-accessible surface area of polar atoms.
        """
        warnings.warn(
            "Geometric PSA calculation is simplified. In production, "
            "use tools like MSMS, PyMOL, or MDAnalysis for accurate calculation."
        )
        
        # Simplified: estimate from structure
        # Would need atom types and proper surface calculation
        return 80.0  # Placeholder


class ChargeCalculator:
    """
    Calculator for net charge and charge distribution.
    
    Net charge affects BBB permeability, with neutral or slightly
    positive charges being favorable.
    """
    
    def __init__(self, ph: float = 7.4):
        self.ph = ph
        
        # pKa values for amino acids (approximate)
        self.pka_values = {
            "N_terminal": 8.0,
            "C_terminal": 3.1,
            "LYS": 10.5,
            "ARG": 12.5,
            "HIS": 6.0,
            "ASP": 3.9,
            "GLU": 4.3,
            "TYR": 10.1,
            "CYS": 8.3,
        }
    
    def calculate_charge(
        self,
        sequence: str,
        ph: Optional[float] = None,
    ) -> Dict:
        """
        Calculate net charge at specified pH.
        
        Parameters
        ----------
        sequence : str
            Peptide sequence
        ph : float, optional
            pH value (defaults to self.ph)
            
        Returns
        -------
        Dict
            Charge results:
            - 'net_charge': net charge
            - 'positive_charge': positive charge
            - 'negative_charge': negative charge
            - 'interpretation': BBB permeability interpretation
        """
        ph = ph or self.ph
        
        # Convert to three-letter codes
        if len(sequence) > 0 and len(sequence[0]) == 1:
            aa_map = {
                "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP",
                "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
                "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
                "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
                "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
            }
            residues = [aa_map.get(aa, "GLY") for aa in sequence]
        else:
            residues = sequence.split() if " " in sequence else [sequence]
        
        positive_charge = 0.0
        negative_charge = 0.0
        
        # N-terminal
        if self._is_protonated("N_terminal", ph):
            positive_charge += 1.0
        
        # C-terminal
        if not self._is_protonated("C_terminal", ph):
            negative_charge += 1.0
        
        # Side chains
        for res in residues:
            if res in ["LYS", "ARG"]:
                if self._is_protonated(res, ph):
                    positive_charge += 1.0
            elif res == "HIS":
                if self._is_protonated(res, ph):
                    positive_charge += 0.5  # Partial at pH 7.4
            elif res in ["ASP", "GLU"]:
                if not self._is_protonated(res, ph):
                    negative_charge += 1.0
            elif res == "TYR":
                if not self._is_protonated(res, ph):
                    negative_charge += 0.1  # Very weak
        
        net_charge = positive_charge - negative_charge
        
        # Interpret for BBB permeability
        if abs(net_charge) < 0.5:
            interpretation = "neutral_optimal"
            permeability_likelihood = 0.9
        elif 0.5 <= net_charge <= 2.0:
            interpretation = "slightly_positive_acceptable"
            permeability_likelihood = 0.7
        elif net_charge > 2.0:
            interpretation = "too_positive"
            permeability_likelihood = 0.3
        elif net_charge < -0.5:
            interpretation = "negative_unfavorable"
            permeability_likelihood = 0.2
        else:
            interpretation = "unknown"
            permeability_likelihood = 0.5
        
        return {
            "net_charge": net_charge,
            "positive_charge": positive_charge,
            "negative_charge": negative_charge,
            "ph": ph,
            "interpretation": interpretation,
            "permeability_likelihood": permeability_likelihood,
        }
    
    def _is_protonated(self, group: str, ph: float) -> bool:
        """Check if a group is protonated at given pH."""
        pka = self.pka_values.get(group, 7.0)
        return ph < pka


class StructuralPropertyAnalyzer:
    """
    Comprehensive analyzer for structural properties relevant to BBB permeability.
    
    This class combines multiple property calculations to provide
    a comprehensive assessment of BBB permeability potential.
    
    Parameters
    ----------
    ph : float, default=7.4
        pH for charge calculations
    """
    
    def __init__(self, ph: float = 7.4):
        from boltzgen.molecular_dynamics.partition_coefficient import LogPCalculator
        self.ph = ph
        self.psa_calculator = PSACalculator()
        self.charge_calculator = ChargeCalculator(ph=ph)
        self.logp_calculator = LogPCalculator()
    
    def analyze(
        self,
        sequence: str,
        structure: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Perform comprehensive structural property analysis.
        
        Parameters
        ----------
        sequence : str
            Peptide sequence
        structure : np.ndarray, optional
            Peptide structure coordinates
            
        Returns
        -------
        Dict
            Comprehensive analysis results:
            - 'molecular_weight': MW in Da
            - 'psa': Polar Surface Area
            - 'logp': LogP value
            - 'net_charge': net charge
            - 'hba': hydrogen bond acceptors
            - 'hbd': hydrogen bond donors
            - 'bbb_score': composite BBB permeability score
            - 'interpretation': overall interpretation
        """
        # Calculate molecular weight
        mw = self._calculate_molecular_weight(sequence)
        
        # Calculate PSA
        psa_result = self.psa_calculator.calculate_psa(sequence, structure)
        
        # Calculate LogP
        logp_result = self.logp_calculator.calculate_logp(sequence, structure)
        
        # Calculate charge
        charge_result = self.charge_calculator.calculate_charge(sequence, self.ph)
        
        # Count HBA/HBD
        hba, hbd = self._count_hbonds(sequence)
        
        # Calculate composite BBB score
        bbb_score = self._calculate_bbb_score(
            mw=mw,
            psa=psa_result["psa"],
            logp=logp_result["logp"],
            charge=charge_result["net_charge"],
            hba=hba,
            hbd=hbd,
        )
        
        # Overall interpretation
        interpretation = self._interpret_bbb_score(bbb_score)
        
        return {
            "molecular_weight": mw,
            "psa": psa_result["psa"],
            "logp": logp_result["logp"],
            "net_charge": charge_result["net_charge"],
            "hba": hba,
            "hbd": hbd,
            "bbb_score": bbb_score,
            "interpretation": interpretation,
            "details": {
                "psa": psa_result,
                "logp": logp_result,
                "charge": charge_result,
            },
        }
    
    def _calculate_molecular_weight(self, sequence: str) -> float:
        """Calculate molecular weight in Da."""
        # Amino acid molecular weights (residue mass, not including water)
        aa_weights = {
            "A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09,
            "C": 103.14, "Q": 128.13, "E": 129.12, "G": 57.05,
            "H": 137.14, "I": 113.16, "L": 113.16, "K": 128.17,
            "M": 131.19, "F": 147.18, "P": 97.12, "S": 87.08,
            "T": 101.11, "W": 186.21, "Y": 163.18, "V": 99.13,
        }
        
        # Add water for peptide bonds
        mw = 18.02  # H2O for terminal
        for aa in sequence:
            mw += aa_weights.get(aa.upper(), 100.0)  # Default ~100 if unknown
        
        return mw
    
    def _count_hbonds(self, sequence: str) -> Tuple[int, int]:
        """
        Count hydrogen bond acceptors and donors.
        
        Returns
        -------
        Tuple[int, int]
            (HBA, HBD) counts
        """
        # HBA: O, N (not in NH3+)
        # HBD: N-H, O-H
        
        hba_map = {
            "N": 1, "Q": 2, "D": 2, "E": 2, "S": 1, "T": 1,
            "Y": 1, "W": 1, "H": 1,
        }
        hbd_map = {
            "N": 1, "Q": 1, "R": 4, "K": 1, "H": 1,
            "S": 1, "T": 1, "Y": 1, "W": 1,
        }
        
        hba = 2  # Backbone (N-terminal N, C-terminal O)
        hbd = 1  # Backbone (N-terminal)
        
        for aa in sequence.upper():
            hba += hba_map.get(aa, 0)
            hbd += hbd_map.get(aa, 0)
        
        return hba, hbd
    
    def _calculate_bbb_score(
        self,
        mw: float,
        psa: float,
        logp: float,
        charge: float,
        hba: int,
        hbd: int,
    ) -> float:
        """
        Calculate composite BBB permeability score (0-1).
        
        Based on Lipinski's Rule of Five and BBB-specific criteria.
        """
        score = 1.0
        
        # Molecular weight penalty (< 600 Da favorable)
        if mw > 600:
            score *= 0.3
        elif mw > 500:
            score *= 0.6
        
        # PSA penalty (< 90 Å² favorable)
        if psa > 150:
            score *= 0.2
        elif psa > 120:
            score *= 0.4
        elif psa > 90:
            score *= 0.7
        
        # LogP penalty (1-3 favorable)
        if logp < 0:
            score *= 0.3
        elif logp < 1:
            score *= 0.6
        elif logp > 5:
            score *= 0.4
        
        # Charge penalty (neutral favorable)
        if abs(charge) > 2:
            score *= 0.3
        elif abs(charge) > 1:
            score *= 0.6
        
        # HBA/HBD penalty (< 6 each favorable)
        if hba > 10 or hbd > 10:
            score *= 0.5
        elif hba > 6 or hbd > 6:
            score *= 0.7
        
        return score
    
    def _interpret_bbb_score(self, score: float) -> str:
        """Interpret BBB score."""
        if score >= 0.8:
            return "high_bbb_permeability"
        elif score >= 0.6:
            return "moderate_bbb_permeability"
        elif score >= 0.4:
            return "low_bbb_permeability"
        else:
            return "very_low_bbb_permeability"

