"""
Composite Scorer for Multi-Objective Peptide Design.

Combines BBB permeability and TAU binding scores for integrated ranking.
"""

import numpy as np
from typing import Dict, List, Optional
from .bbb_scorer import BBBScorer
from .docking_proxy import DockingProxy


class CompositeScorer:
    """
    Composite scorer combining BBB and TAU scoring.

    Implements multi-objective optimization for peptide design:
    - Objective 1: BBB permeability (maximize)
    - Objective 2: TAU binding affinity (maximize)
    """

    def __init__(
        self,
        bbb_scorer: BBBScorer,
        docking_proxy: DockingProxy,
        bbb_weight: float = 0.6,
        tau_weight: float = 0.4,
    ):
        """
        Initialize composite scorer.

        Parameters
        ----------
        bbb_scorer : BBBScorer
            BBB permeability scorer
        docking_proxy : DockingProxy
            TAU docking proxy
        bbb_weight : float, default=0.6
            Weight for BBB score (0-1)
        tau_weight : float, default=0.4
            Weight for TAU score (0-1)
        """
        self.bbb_scorer = bbb_scorer
        self.docking_proxy = docking_proxy
        self.bbb_weight = bbb_weight
        self.tau_weight = tau_weight

        # Normalize weights
        total_weight = bbb_weight + tau_weight
        self.bbb_weight /= total_weight
        self.tau_weight /= total_weight

    @classmethod
    def from_checkpoints(
        cls,
        bbb_model_path: str = "models/bbb_classifier.pt",
        tau_sequence_file: str = "data/tau/tau_sequence_info.txt",
        tau_weights_file: str = "data/tau/tau_entropy.npy",
        tau_regions_file: str = "data/tau/tau_target_regions.json",
        bbb_weight: float = 0.6,
        tau_weight: float = 0.4,
    ):
        """
        Load composite scorer from saved files.

        Parameters
        ----------
        bbb_model_path : str
            Path to BBB classifier checkpoint
        tau_sequence_file : str
            Path to TAU sequence file
        tau_weights_file : str
            Path to TAU conservation weights
        tau_regions_file : str
            Path to TAU target regions
        bbb_weight : float
            Weight for BBB score
        tau_weight : float
            Weight for TAU score

        Returns
        -------
        CompositeScorer
            Loaded instance
        """
        bbb_scorer = BBBScorer.from_checkpoint(bbb_model_path)
        docking_proxy = DockingProxy.from_files(
            tau_sequence_file,
            tau_weights_file,
            tau_regions_file,
        )

        return cls(bbb_scorer, docking_proxy, bbb_weight, tau_weight)

    def score(self, peptide_seq: str) -> Dict[str, float]:
        """
        Compute all scores for a peptide.

        Parameters
        ----------
        peptide_seq : str
            Peptide sequence

        Returns
        -------
        dict
            Scores:
            - bbb_prob: BBB permeability probability
            - tau_score: TAU binding score (best region)
            - tau_region: Best TAU target region
            - composite: Weighted composite score
            - bbb_weighted: Weighted BBB contribution
            - tau_weighted: Weighted TAU contribution
        """
        # BBB score
        bbb_prob = self.bbb_scorer.predict(peptide_seq)

        # TAU score
        tau_result = self.docking_proxy.score_peptide_best(peptide_seq)
        tau_score = tau_result.get('weighted_total', 0.0)

        # Normalize TAU score to [0, 1] range (assume max ~2.0)
        tau_score_norm = min(tau_score / 2.0, 1.0)

        # Weighted scores
        bbb_weighted = self.bbb_weight * bbb_prob
        tau_weighted = self.tau_weight * tau_score_norm

        # Composite score
        composite = bbb_weighted + tau_weighted

        return {
            'sequence': peptide_seq,
            'bbb_prob': bbb_prob,
            'tau_score': tau_score,
            'tau_score_normalized': tau_score_norm,
            'tau_region': tau_result.get('region_name', 'Unknown'),
            'bbb_weighted': bbb_weighted,
            'tau_weighted': tau_weighted,
            'composite': composite,
        }

    def score_batch(self, peptide_seqs: List[str]) -> List[Dict]:
        """
        Score multiple peptides.

        Parameters
        ----------
        peptide_seqs : list of str
            List of peptide sequences

        Returns
        -------
        list of dict
            Scores for each peptide
        """
        return [self.score(seq) for seq in peptide_seqs]

    def rank_peptides(
        self,
        peptide_seqs: List[str],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Rank peptides by composite score.

        Parameters
        ----------
        peptide_seqs : list of str
            List of peptide sequences
        top_k : int, optional
            Return only top K peptides

        Returns
        -------
        list of dict
            Ranked peptides with scores
        """
        scores = self.score_batch(peptide_seqs)
        scores.sort(key=lambda x: x['composite'], reverse=True)

        if top_k is not None:
            scores = scores[:top_k]

        return scores


def demo():
    """Demonstrate composite scoring."""
    print("=" * 80)
    print("COMPOSITE SCORER DEMO")
    print("=" * 80)

    # Load scorer
    print("\nLoading scorers...")
    scorer = CompositeScorer.from_checkpoints(
        bbb_weight=0.6,
        tau_weight=0.4,
    )

    print(f"  BBB weight: {scorer.bbb_weight:.2f}")
    print(f"  TAU weight: {scorer.tau_weight:.2f}")

    # Test peptides
    test_peptides = [
        "YGRKKRRQRRR",         # TAT (BBB+, poly-basic)
        "RQIKIWFQNRRMKWKK",    # Penetratin (BBB+)
        "DDDEEEEDDDEEEE",      # Acidic (BBB-)
        "AILAILAILAIL",        # Hydrophobic
        "GRKKRRQRRRPPQ",       # TAT-short (BBB+)
    ]

    print("\n" + "=" * 80)
    print("RANKING PEPTIDES")
    print("=" * 80)

    ranked = scorer.rank_peptides(test_peptides)

    print(f"\n{'Rank':4s} | {'Sequence':20s} | {'BBB':5s} | {'TAU':5s} | {'Composite':9s} | Target")
    print("-" * 80)

    for i, result in enumerate(ranked, 1):
        print(f"{i:4d} | {result['sequence']:20s} | "
              f"{result['bbb_prob']:.3f} | {result['tau_score_normalized']:.3f} | "
              f"{result['composite']:.3f}     | {result['tau_region']}")

    print("\n" + "=" * 80)
    print("DETAILED SCORES (Top Peptide)")
    print("=" * 80)

    top = ranked[0]
    print(f"\nSequence: {top['sequence']}")
    print(f"\nBBB Permeability:")
    print(f"  Probability:       {top['bbb_prob']:.3f}")
    print(f"  Weighted (60%):    {top['bbb_weighted']:.3f}")
    print(f"\nTAU Binding:")
    print(f"  Score (raw):       {top['tau_score']:.3f}")
    print(f"  Score (norm):      {top['tau_score_normalized']:.3f}")
    print(f"  Best target:       {top['tau_region']}")
    print(f"  Weighted (40%):    {top['tau_weighted']:.3f}")
    print(f"\nComposite Score:     {top['composite']:.3f}")


if __name__ == "__main__":
    demo()
