"""
MD Trajectory Analysis Module.

This module provides tools for analyzing MD trajectories to assess
peptide-membrane interactions and BBB permeability.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import warnings


class MDTrajectoryAnalyzer:
    """
    Analyzer for MD trajectories.
    
    This class provides methods for analyzing MD simulation trajectories
    to extract information about peptide-membrane interactions.
    
    Parameters
    ----------
    timestep : float, default=0.002
        MD timestep in ps
    """
    
    def __init__(self, timestep: float = 0.002):
        self.timestep = timestep
    
    def load_trajectory(
        self,
        trajectory_file: Path,
        topology_file: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Load MD trajectory from file.
        
        Parameters
        ----------
        trajectory_file : Path
            Path to trajectory file (e.g., .xtc, .dcd, .nc)
        topology_file : Path, optional
            Path to topology file (e.g., .pdb, .gro)
            
        Returns
        -------
        np.ndarray
            Trajectory data (T, N, 3)
        """
        # In practice, would use MDAnalysis, mdtraj, or similar
        warnings.warn(
            "Trajectory loading is simplified. In production, use "
            "MDAnalysis, mdtraj, or similar libraries."
        )
        
        # Placeholder
        return np.zeros((100, 100, 3))
    
    def analyze_center_of_mass(
        self,
        trajectory: np.ndarray,
        atom_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate center of mass trajectory.
        
        Parameters
        ----------
        trajectory : np.ndarray
            Trajectory data (T, N, 3)
        atom_indices : np.ndarray, optional
            Indices of atoms to include (default: all)
            
        Returns
        -------
        np.ndarray
            Center of mass trajectory (T, 3)
        """
        if atom_indices is not None:
            traj_subset = trajectory[:, atom_indices, :]
        else:
            traj_subset = trajectory
        
        com = np.mean(traj_subset, axis=1)
        return com
    
    def calculate_rmsd(
        self,
        trajectory: np.ndarray,
        reference: np.ndarray,
        atom_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate RMSD from reference structure.
        
        Parameters
        ----------
        trajectory : np.ndarray
            Trajectory data (T, N, 3)
        reference : np.ndarray
            Reference structure (N, 3)
        atom_indices : np.ndarray, optional
            Indices of atoms to include
            
        Returns
        -------
        np.ndarray
            RMSD values over time (T,)
        """
        if atom_indices is not None:
            traj_subset = trajectory[:, atom_indices, :]
            ref_subset = reference[atom_indices, :]
        else:
            traj_subset = trajectory
            ref_subset = reference
        
        # Center structures
        traj_centered = traj_subset - np.mean(traj_subset, axis=1, keepdims=True)
        ref_centered = ref_subset - np.mean(ref_subset)
        
        # Calculate RMSD (simplified - would need proper alignment)
        rmsd = np.sqrt(
            np.mean(
                np.sum((traj_centered - ref_centered) ** 2, axis=2),
                axis=1
            )
        )
        
        return rmsd
    
    def calculate_distance(
        self,
        trajectory: np.ndarray,
        group1_indices: np.ndarray,
        group2_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate distance between two groups over time.
        
        Parameters
        ----------
        trajectory : np.ndarray
            Trajectory data (T, N, 3)
        group1_indices : np.ndarray
            Indices of first group
        group2_indices : np.ndarray
            Indices of second group
            
        Returns
        -------
        np.ndarray
            Distances over time (T,)
        """
        com1 = np.mean(trajectory[:, group1_indices, :], axis=1)
        com2 = np.mean(trajectory[:, group2_indices, :], axis=1)
        
        distances = np.linalg.norm(com1 - com2, axis=1)
        return distances


class PermeabilityAnalyzer:
    """
    Analyzer for BBB permeability from MD simulations.
    
    This class combines multiple analyses to assess BBB permeability
    from MD simulation results.
    
    Parameters
    ----------
    membrane_center : float, default=0.0
        Z-coordinate of membrane center
    membrane_thickness : float, default=4.0
        Membrane thickness in nm (typically -2 to +2 nm)
    """
    
    def __init__(
        self,
        membrane_center: float = 0.0,
        membrane_thickness: float = 4.0,
    ):
        self.membrane_center = membrane_center
        self.membrane_thickness = membrane_thickness
        self.membrane_bounds = (
            membrane_center - membrane_thickness / 2,
            membrane_center + membrane_thickness / 2,
        )
    
    def analyze_permeation(
        self,
        peptide_com_trajectory: np.ndarray,
        time: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Analyze peptide permeation through membrane.
        
        Parameters
        ----------
        peptide_com_trajectory : np.ndarray
            Peptide center of mass trajectory (T, 3)
        time : np.ndarray, optional
            Time array (if None, uses frame indices)
            
        Returns
        -------
        Dict
            Permeation analysis:
            - 'insertion_depth': depth of insertion over time
            - 'crossing_events': number of membrane crossings
            - 'residence_time': time spent in membrane
            - 'permeation_rate': estimated permeation rate
        """
        z_coords = peptide_com_trajectory[:, 2]
        
        # Calculate insertion depth
        insertion_depth = np.abs(z_coords - self.membrane_center)
        
        # Determine if peptide is in membrane
        in_membrane = (
            (z_coords > self.membrane_bounds[0]) &
            (z_coords < self.membrane_bounds[1])
        )
        
        # Count crossing events
        crossing_events = 0
        was_inside = in_membrane[0]
        for is_inside in in_membrane[1:]:
            if is_inside != was_inside:
                crossing_events += 1
            was_inside = is_inside
        
        # Calculate residence time
        if time is None:
            time = np.arange(len(z_coords)) * 0.002  # Assume 0.002 ps timestep
        
        residence_time = np.sum(in_membrane) * (time[1] - time[0]) if len(time) > 1 else 0.0
        
        # Estimate permeation rate (simplified)
        total_time = time[-1] - time[0] if len(time) > 1 else 1.0
        permeation_rate = crossing_events / total_time if total_time > 0 else 0.0
        
        return {
            "insertion_depth": insertion_depth,
            "crossing_events": crossing_events,
            "residence_time": residence_time,
            "permeation_rate": permeation_rate,
            "fraction_in_membrane": np.mean(in_membrane),
        }
    
    def calculate_permeability_coefficient(
        self,
        dg_perm: float,
        temperature: float = 310.0,
    ) -> Dict:
        """
        Calculate permeability coefficient from free energy.
        
        Uses empirical correlation:
        log P_app ≈ -0.4 × ΔG_perm + 5.5
        
        Parameters
        ----------
        dg_perm : float
            Free energy of permeation in kcal/mol
        temperature : float, default=310.0
            Temperature in K
            
        Returns
        -------
        Dict
            Permeability results:
            - 'log_p_app': log10 of apparent permeability
            - 'p_app': apparent permeability in cm/s
            - 'interpretation': permeability interpretation
        """
        # Empirical correlation
        log_p_app = -0.4 * dg_perm + 5.5
        p_app = 10 ** log_p_app  # cm/s
        
        # Interpret permeability
        if p_app > 1e-5:
            interpretation = "high_permeability"
        elif p_app > 1e-6:
            interpretation = "moderate_permeability"
        elif p_app > 1e-7:
            interpretation = "low_permeability"
        else:
            interpretation = "very_low_permeability"
        
        return {
            "log_p_app": log_p_app,
            "p_app": p_app,
            "interpretation": interpretation,
            "dg_perm": dg_perm,
        }
    
    def comprehensive_analysis(
        self,
        peptide_trajectory: np.ndarray,
        dg_perm: Optional[float] = None,
        structural_properties: Optional[Dict] = None,
    ) -> Dict:
        """
        Perform comprehensive BBB permeability analysis.
        
        Combines MD trajectory analysis with free energy and
        structural property calculations.
        
        Parameters
        ----------
        peptide_trajectory : np.ndarray
            Peptide trajectory (T, N, 3)
        dg_perm : float, optional
            Free energy of permeation
        structural_properties : Dict, optional
            Structural property analysis results
            
        Returns
        -------
        Dict
            Comprehensive analysis results
        """
        analyzer = MDTrajectoryAnalyzer()
        com_traj = analyzer.analyze_center_of_mass(peptide_trajectory)
        
        # Analyze permeation
        permeation_analysis = self.analyze_permeation(com_traj)
        
        # Calculate permeability coefficient if ΔG_perm available
        permeability_results = None
        if dg_perm is not None:
            permeability_results = self.calculate_permeability_coefficient(dg_perm)
        
        # Combine results
        results = {
            "permeation_analysis": permeation_analysis,
            "permeability_coefficient": permeability_results,
            "structural_properties": structural_properties,
        }
        
        # Calculate composite BBB score
        bbb_score = self._calculate_composite_score(
            permeation_analysis,
            permeability_results,
            structural_properties,
        )
        
        results["bbb_score"] = bbb_score
        results["interpretation"] = self._interpret_bbb_score(bbb_score)
        
        return results
    
    def _calculate_composite_score(
        self,
        permeation_analysis: Dict,
        permeability_results: Optional[Dict],
        structural_properties: Optional[Dict],
    ) -> float:
        """Calculate composite BBB permeability score (0-1)."""
        score = 1.0
        
        # Factor 1: Residence time in membrane
        if permeation_analysis["fraction_in_membrane"] > 0.5:
            score *= 0.8  # Good membrane interaction
        elif permeation_analysis["fraction_in_membrane"] < 0.1:
            score *= 0.3  # Poor membrane interaction
        
        # Factor 2: Permeability coefficient
        if permeability_results is not None:
            p_app = permeability_results["p_app"]
            if p_app > 1e-5:
                score *= 1.0
            elif p_app > 1e-6:
                score *= 0.7
            else:
                score *= 0.4
        
        # Factor 3: Structural properties
        if structural_properties is not None:
            struct_score = structural_properties.get("bbb_score", 0.5)
            score *= struct_score
        
        return score
    
    def _interpret_bbb_score(self, score: float) -> str:
        """Interpret composite BBB score."""
        if score >= 0.7:
            return "high_bbb_permeability"
        elif score >= 0.5:
            return "moderate_bbb_permeability"
        elif score >= 0.3:
            return "low_bbb_permeability"
        else:
            return "very_low_bbb_permeability"

