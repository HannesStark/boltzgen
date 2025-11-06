"""
Free Energy Calculation Module.

This module implements methods for calculating free energy profiles
of peptide permeation through membranes using enhanced sampling techniques.

Key methods:
- Umbrella Sampling
- Metadynamics
- WHAM (Weighted Histogram Analysis Method)
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import warnings


class UmbrellaSampling:
    """
    Umbrella Sampling for calculating free energy profiles.
    
    Umbrella Sampling uses harmonic restraints along a reaction coordinate
    (typically the z-distance from membrane center) to sample different
    regions of configuration space.
    
    Parameters
    ----------
    reaction_coordinate : str, default="z_distance"
        Reaction coordinate name ("z_distance", "insertion_depth", etc.)
    n_windows : int, default=15
        Number of umbrella sampling windows
    z_range : Tuple[float, float], default=(-5.0, 5.0)
        Range of z-coordinates to sample (nm)
    force_constant : float, default=1000.0
        Harmonic restraint force constant (kJ/mol/nm²)
    """
    
    def __init__(
        self,
        reaction_coordinate: str = "z_distance",
        n_windows: int = 15,
        z_range: Tuple[float, float] = (-5.0, 5.0),
        force_constant: float = 1000.0,  # kJ/mol/nm²
    ):
        self.reaction_coordinate = reaction_coordinate
        self.n_windows = n_windows
        self.z_range = z_range
        self.force_constant = force_constant
        
        # Define window centers
        self.window_centers = np.linspace(z_range[0], z_range[1], n_windows)
        
    def setup_windows(
        self,
        system: Dict,
        membrane_center: float = 0.0,
    ) -> List[Dict]:
        """
        Setup umbrella sampling windows.
        
        Parameters
        ----------
        system : Dict
            System dictionary
        membrane_center : float, default=0.0
            Z-coordinate of membrane center
            
        Returns
        -------
        List[Dict]
            List of window configurations
        """
        windows = []
        for i, z_center in enumerate(self.window_centers):
            window = {
                "window_id": i,
                "z_center": z_center,
                "force_constant": self.force_constant,
                "system": system.copy(),
            }
            windows.append(window)
        
        return windows
    
    def run_umbrella_sampling(
        self,
        windows: List[Dict],
        n_steps_per_window: int = 10000000,  # 20 ns per window
        output_freq: int = 10000,
    ) -> Dict:
        """
        Run umbrella sampling simulations for all windows.
        
        Parameters
        ----------
        windows : List[Dict]
            List of window configurations
        n_steps_per_window : int, default=10000000
            Number of MD steps per window
        output_freq : int, default=10000
            Output frequency
            
        Returns
        -------
        Dict
            Umbrella sampling results:
            - 'window_data': data for each window
            - 'reaction_coordinate_values': sampled RC values
        """
        # In practice, this would run MD simulations with restraints
        warnings.warn(
            "Umbrella sampling execution is a placeholder. In production, "
            "interface with GROMACS, NAMD, or OpenMM with PLUMED/Colvars."
        )
        
        window_data = []
        for window in windows:
            # Placeholder: would run actual MD simulation
            rc_values = np.random.normal(
                window["z_center"],
                0.5,  # std dev in nm
                n_steps_per_window // output_freq
            )
            window_data.append({
                "window_id": window["window_id"],
                "z_center": window["z_center"],
                "rc_values": rc_values,
            })
        
        return {
            "window_data": window_data,
            "reaction_coordinate_values": np.concatenate([w["rc_values"] for w in window_data]),
        }
    
    def calculate_pmf(
        self,
        umbrella_results: Dict,
        temperature: float = 310.0,
        method: str = "WHAM",
    ) -> Dict:
        """
        Calculate Potential of Mean Force (PMF) from umbrella sampling.
        
        Parameters
        ----------
        umbrella_results : Dict
            Results from run_umbrella_sampling
        temperature : float, default=310.0
            Temperature in K
        method : str, default="WHAM"
            Method for PMF calculation ("WHAM", "MBAR", etc.)
            
        Returns
        -------
        Dict
            PMF results:
            - 'z_values': reaction coordinate values
            - 'pmf': free energy values (kcal/mol)
            - 'dg_perm': free energy of permeation
        """
        # Simplified WHAM implementation
        # In practice, use pyWHAM, pymbar, or similar tools
        
        window_data = umbrella_results["window_data"]
        
        # Combine all RC values
        all_rc = np.concatenate([w["rc_values"] for w in window_data])
        
        # Create histogram
        z_bins = np.linspace(self.z_range[0], self.z_range[1], 100)
        z_centers = (z_bins[:-1] + z_bins[1:]) / 2
        
        # Simplified PMF calculation (would use proper WHAM in practice)
        hist, _ = np.histogram(all_rc, bins=z_bins)
        hist = hist + 1e-10  # Avoid log(0)
        pmf = -0.001987 * temperature * np.log(hist)  # Convert to kcal/mol
        pmf = pmf - pmf.min()  # Set minimum to 0
        
        # Calculate ΔG_perm (difference between water and membrane center)
        water_region = (z_centers < -3.0) | (z_centers > 3.0)
        membrane_region = (z_centers > -1.0) & (z_centers < 1.0)
        
        if np.any(water_region) and np.any(membrane_region):
            g_water = np.mean(pmf[water_region])
            g_membrane = np.mean(pmf[membrane_region])
            dg_perm = g_membrane - g_water
        else:
            dg_perm = None
        
        return {
            "z_values": z_centers,
            "pmf": pmf,
            "dg_perm": dg_perm,
            "temperature": temperature,
        }


class Metadynamics:
    """
    Metadynamics for free energy calculation.
    
    Metadynamics uses history-dependent bias potentials to explore
    free energy landscapes and reconstruct PMFs.
    
    Parameters
    ----------
    reaction_coordinate : str, default="z_distance"
        Reaction coordinate name
    height : float, default=1.0
        Height of Gaussian hills (kJ/mol)
    width : float, default=0.2
        Width of Gaussian hills (nm)
    pace : int, default=500
        Frequency of adding Gaussian hills (steps)
    """
    
    def __init__(
        self,
        reaction_coordinate: str = "z_distance",
        height: float = 1.0,  # kJ/mol
        width: float = 0.2,  # nm
        pace: int = 500,  # steps
    ):
        self.reaction_coordinate = reaction_coordinate
        self.height = height
        self.width = width
        self.pace = pace
        
    def run_metadynamics(
        self,
        system: Dict,
        n_steps: int = 50000000,  # 100 ns
        z_range: Tuple[float, float] = (-5.0, 5.0),
    ) -> Dict:
        """
        Run metadynamics simulation.
        
        Parameters
        ----------
        system : Dict
            System dictionary
        n_steps : int, default=50000000
            Number of MD steps
        z_range : Tuple[float, float], default=(-5.0, 5.0)
            Range of reaction coordinate
            
        Returns
        -------
        Dict
            Metadynamics results
        """
        # In practice, this would use PLUMED, Colvars, or similar
        warnings.warn(
            "Metadynamics execution is a placeholder. In production, "
            "use PLUMED, Colvars, or similar enhanced sampling tools."
        )
        
        # Placeholder results
        return {
            "trajectory": None,
            "bias_potential": None,
            "pmf": None,
        }
    
    def calculate_pmf(
        self,
        metadynamics_results: Dict,
        temperature: float = 310.0,
    ) -> Dict:
        """
        Calculate PMF from metadynamics.
        
        In well-tempered metadynamics, the PMF is directly related
        to the accumulated bias potential.
        """
        # Placeholder
        return {
            "z_values": np.linspace(-5.0, 5.0, 100),
            "pmf": np.zeros(100),
            "dg_perm": None,
        }


class FreeEnergyCalculator:
    """
    High-level interface for free energy calculations.
    
    This class provides a unified interface for different free energy
    calculation methods and interprets results for BBB permeability.
    
    Parameters
    ----------
    method : str, default="umbrella_sampling"
        Method to use ("umbrella_sampling", "metadynamics")
    """
    
    def __init__(self, method: str = "umbrella_sampling"):
        self.method = method
        
        if method == "umbrella_sampling":
            self.calculator = UmbrellaSampling()
        elif method == "metadynamics":
            self.calculator = Metadynamics()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_permeation_free_energy(
        self,
        system: Dict,
        membrane_center: float = 0.0,
        **kwargs
    ) -> Dict:
        """
        Calculate free energy of permeation (ΔG_perm).
        
        Parameters
        ----------
        system : Dict
            System dictionary
        membrane_center : float, default=0.0
            Z-coordinate of membrane center
        **kwargs
            Additional arguments for the specific method
            
        Returns
        -------
        Dict
            Results containing:
            - 'dg_perm': free energy of permeation (kcal/mol)
            - 'pmf': full PMF profile
            - 'interpretation': permeability interpretation
        """
        if self.method == "umbrella_sampling":
            windows = self.calculator.setup_windows(system, membrane_center)
            results = self.calculator.run_umbrella_sampling(windows, **kwargs)
            pmf_results = self.calculator.calculate_pmf(results, **kwargs)
        else:  # metadynamics
            results = self.calculator.run_metadynamics(system, **kwargs)
            pmf_results = self.calculator.calculate_pmf(results, **kwargs)
        
        dg_perm = pmf_results.get("dg_perm")
        
        # Interpret results
        if dg_perm is not None:
            if dg_perm < 5.0:
                interpretation = "high_permeability"
                permeability_score = 1.0
            elif dg_perm < 10.0:
                interpretation = "moderate_permeability"
                permeability_score = 0.6
            elif dg_perm < 20.0:
                interpretation = "low_permeability"
                permeability_score = 0.3
            else:
                interpretation = "very_low_permeability"
                permeability_score = 0.1
        else:
            interpretation = "unknown"
            permeability_score = 0.5
        
        # Calculate permeability coefficient using empirical correlation
        # log P_app ≈ -0.4 × ΔG_perm + 5.5
        if dg_perm is not None:
            log_p_app = -0.4 * dg_perm + 5.5
            p_app = 10 ** log_p_app  # cm/s
        else:
            log_p_app = None
            p_app = None
        
        return {
            "dg_perm": dg_perm,
            "pmf": pmf_results,
            "interpretation": interpretation,
            "permeability_score": permeability_score,
            "log_p_app": log_p_app,
            "p_app": p_app,
        }

