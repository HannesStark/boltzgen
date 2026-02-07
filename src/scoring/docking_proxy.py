"""
Docking Proxy for TAU-Peptide Interaction Scoring.

MSA-weighted interaction scoring as a fast proxy for molecular docking.
Uses conservation scores to weight interactions.
"""

import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path


# Amino acid properties for interaction scoring
# Charge at pH 7.4
AA_CHARGE = {
    'R': +1, 'K': +1, 'H': +0.5,  # Positive
    'D': -1, 'E': -1,               # Negative
}

# Hydrophobicity (Kyte-Doolittle scale)
AA_HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'F': 2.8, 'I': 4.5, 'L': 3.8, 'M': 1.9,
    'V': 4.2, 'W': -0.9, 'Y': -1.3, 'P': -1.6, 'G': -0.4,
    'S': -0.8, 'T': -0.7, 'N': -3.5, 'Q': -3.5, 'D': -3.5,
    'E': -3.5, 'K': -3.9, 'R': -4.5, 'H': -3.2,
}

# Hydrogen bonding capacity
AA_HBOND_DONOR = {'R', 'K', 'H', 'N', 'Q', 'S', 'T', 'W', 'Y'}
AA_HBOND_ACCEPTOR = {'D', 'E', 'N', 'Q', 'S', 'T', 'Y'}


class DockingProxy:
    """
    Fast docking proxy for TAU-peptide interactions.

    Uses simple interaction potentials weighted by TAU conservation scores.
    """

    def __init__(
        self,
        tau_sequence: str,
        tau_weights: Optional[np.ndarray] = None,
        tau_target_regions: Optional[List[Dict]] = None,
    ):
        """
        Initialize docking proxy.

        Parameters
        ----------
        tau_sequence : str
            TAU protein sequence
        tau_weights : np.ndarray, optional
            Conservation/importance weights per position (length = TAU length)
            If None, uniform weights
        tau_target_regions : list of dict, optional
            Target regions from tau_target_regions.json
        """
        self.tau_sequence = tau_sequence
        self.tau_length = len(tau_sequence)

        # Set weights
        if tau_weights is not None:
            self.tau_weights = tau_weights
        else:
            self.tau_weights = np.ones(self.tau_length)

        # Normalize weights
        self.tau_weights = self.tau_weights / self.tau_weights.max()

        # Target regions
        self.target_regions = tau_target_regions or []

    @classmethod
    def from_files(
        cls,
        sequence_file: str = "data/tau/tau_sequence_info.txt",
        weights_file: str = "data/tau/tau_entropy.npy",
        regions_file: str = "data/tau/tau_target_regions.json",
    ):
        """
        Load docking proxy from saved files.

        Parameters
        ----------
        sequence_file : str
            Path to TAU sequence file
        weights_file : str
            Path to conservation/weights file
        regions_file : str
            Path to target regions JSON

        Returns
        -------
        DockingProxy
            Loaded instance
        """
        # Load sequence
        with open(sequence_file, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith("Sequence:"):
                sequence = ''.join(lines[i+1:]).replace('\n', '').replace(' ', '')
                break

        # Load weights
        if Path(weights_file).exists():
            weights = np.load(weights_file)
            # Convert conservation to importance weights (higher = better target)
            # weights = conservation scores (0-1), we want to use them directly
        else:
            weights = None

        # Load target regions
        if Path(regions_file).exists():
            with open(regions_file, 'r') as f:
                target_regions = json.load(f)
        else:
            target_regions = []

        return cls(sequence, weights, target_regions)

    def compute_electrostatic_score(
        self,
        peptide_seq: str,
        tau_region: str,
    ) -> float:
        """
        Compute electrostatic interaction score.

        Favorable: opposite charges
        Unfavorable: same charges

        Parameters
        ----------
        peptide_seq : str
            Peptide sequence
        tau_region : str
            TAU region sequence

        Returns
        -------
        float
            Electrostatic score (positive = favorable)
        """
        score = 0.0

        for pep_aa in peptide_seq:
            pep_charge = AA_CHARGE.get(pep_aa, 0)
            if pep_charge == 0:
                continue

            for tau_aa in tau_region:
                tau_charge = AA_CHARGE.get(tau_aa, 0)
                # Opposite charges attract
                score += -pep_charge * tau_charge

        # Normalize by number of interactions
        n_interactions = len(peptide_seq) * len(tau_region)
        if n_interactions > 0:
            score /= n_interactions

        return score

    def compute_hydrophobic_score(
        self,
        peptide_seq: str,
        tau_region: str,
    ) -> float:
        """
        Compute hydrophobic interaction score.

        Favorable: hydrophobic-hydrophobic contacts

        Parameters
        ----------
        peptide_seq : str
            Peptide sequence
        tau_region : str
            TAU region sequence

        Returns
        -------
        float
            Hydrophobic score (positive = favorable)
        """
        score = 0.0

        for pep_aa in peptide_seq:
            pep_hydro = AA_HYDROPHOBICITY.get(pep_aa, 0)
            if pep_hydro <= 0:  # Not hydrophobic
                continue

            for tau_aa in tau_region:
                tau_hydro = AA_HYDROPHOBICITY.get(tau_aa, 0)
                if tau_hydro > 0:  # Both hydrophobic
                    score += pep_hydro * tau_hydro

        # Normalize
        n_interactions = len(peptide_seq) * len(tau_region)
        if n_interactions > 0:
            score /= n_interactions

        return score

    def compute_hbond_score(
        self,
        peptide_seq: str,
        tau_region: str,
    ) -> float:
        """
        Compute hydrogen bonding score.

        Favorable: donor-acceptor pairs

        Parameters
        ----------
        peptide_seq : str
            Peptide sequence
        tau_region : str
            TAU region sequence

        Returns
        -------
        float
            H-bond score (positive = favorable)
        """
        score = 0.0

        for pep_aa in peptide_seq:
            is_donor = pep_aa in AA_HBOND_DONOR
            is_acceptor = pep_aa in AA_HBOND_ACCEPTOR

            for tau_aa in tau_region:
                tau_is_donor = tau_aa in AA_HBOND_DONOR
                tau_is_acceptor = tau_aa in AA_HBOND_ACCEPTOR

                # Donor-acceptor pairs
                if (is_donor and tau_is_acceptor) or (is_acceptor and tau_is_donor):
                    score += 1.0

        # Normalize
        n_interactions = len(peptide_seq) * len(tau_region)
        if n_interactions > 0:
            score /= n_interactions

        return score

    def score_peptide_region(
        self,
        peptide_seq: str,
        region_start: int,
        region_end: int,
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Score peptide binding to a specific TAU region.

        Parameters
        ----------
        peptide_seq : str
            Peptide sequence
        region_start : int
            Region start (1-indexed)
        region_end : int
            Region end (1-indexed, inclusive)
        weights : np.ndarray, optional
            Per-residue importance weights (overrides default)

        Returns
        -------
        dict
            Scores: electrostatic, hydrophobic, hbond, total, weighted_total
        """
        # Extract TAU region
        start_idx = max(0, region_start - 1)
        end_idx = min(self.tau_length, region_end)
        tau_region = self.tau_sequence[start_idx:end_idx]

        # Compute interaction scores
        elec_score = self.compute_electrostatic_score(peptide_seq, tau_region)
        hydro_score = self.compute_hydrophobic_score(peptide_seq, tau_region)
        hbond_score = self.compute_hbond_score(peptide_seq, tau_region)

        # Total score (weighted combination)
        total_score = 0.3 * elec_score + 0.4 * hydro_score + 0.3 * hbond_score

        # Apply TAU conservation weights
        if weights is None:
            weights = self.tau_weights[start_idx:end_idx]
        else:
            weights = weights[start_idx:end_idx]

        avg_weight = weights.mean()
        weighted_total = total_score * avg_weight

        return {
            'electrostatic': elec_score,
            'hydrophobic': hydro_score,
            'hbond': hbond_score,
            'total': total_score,
            'weighted_total': weighted_total,
            'region_weight': avg_weight,
        }

    def score_peptide_all_regions(self, peptide_seq: str) -> List[Dict]:
        """
        Score peptide against all target regions.

        Parameters
        ----------
        peptide_seq : str
            Peptide sequence

        Returns
        -------
        list of dict
            Scores for each target region
        """
        results = []

        for region in self.target_regions:
            scores = self.score_peptide_region(
                peptide_seq,
                region['start'],
                region['end'],
            )

            results.append({
                'region_name': region['name'],
                'region_start': region['start'],
                'region_end': region['end'],
                'priority': region.get('priority', 'Unknown'),
                **scores,
            })

        # Sort by weighted total score
        results.sort(key=lambda x: x['weighted_total'], reverse=True)

        return results

    def score_peptide_best(self, peptide_seq: str) -> Dict:
        """
        Score peptide and return best target region.

        Parameters
        ----------
        peptide_seq : str
            Peptide sequence

        Returns
        -------
        dict
            Best scoring region with all scores
        """
        all_scores = self.score_peptide_all_regions(peptide_seq)

        if not all_scores:
            return {'error': 'No target regions defined'}

        return all_scores[0]


def demo():
    """Demonstrate docking proxy on test peptides."""
    print("=" * 80)
    print("DOCKING PROXY DEMO")
    print("=" * 80)

    # Load proxy
    proxy = DockingProxy.from_files()

    print(f"\nTAU sequence length: {proxy.tau_length} aa")
    print(f"Target regions: {len(proxy.target_regions)}")

    # Test peptides
    test_peptides = [
        ("YGRKKRRQRRR", "TAT (poly-basic)"),
        ("RQIKIWFQNRRMKWKK", "Penetratin"),
        ("AILAILAIL", "Hydrophobic peptide"),
        ("DDDEEEE", "Acidic peptide"),
        ("NQSTNQST", "Polar peptide"),
    ]

    print("\n" + "=" * 80)
    print("PEPTIDE SCORES")
    print("=" * 80)

    for peptide, name in test_peptides:
        print(f"\n{name}: {peptide}")
        print("-" * 80)

        best = proxy.score_peptide_best(peptide)

        print(f"Best target: {best['region_name']} (Pos {best['region_start']}-{best['region_end']})")
        print(f"  Electrostatic: {best['electrostatic']:7.3f}")
        print(f"  Hydrophobic:   {best['hydrophobic']:7.3f}")
        print(f"  H-bond:        {best['hbond']:7.3f}")
        print(f"  Total:         {best['total']:7.3f}")
        print(f"  Weighted:      {best['weighted_total']:7.3f}")


if __name__ == "__main__":
    demo()
