"""
TAU Protein Analysis for Target Region Selection.

Combines conservation scores (from MSA) and disorder predictions
to identify optimal binding regions for peptide design.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


# Known TAU functional regions (from literature)
# Based on human TAU 2N4R (UniProt P10636-1, 758 aa, but actual sequence is 441 aa)
# Note: UniProt includes signal peptide, actual mature protein is 441 aa

TAU_REGIONS = {
    "N-terminal_projection": (1, 150),      # Projection domain
    "proline_rich_1": (151, 198),           # Proline-rich region 1
    "proline_rich_2": (199, 244),           # Proline-rich region 2
    "microtubule_binding_R1": (244, 274),   # Repeat 1
    "microtubule_binding_R2": (275, 305),   # Repeat 2
    "microtubule_binding_R3": (306, 336),   # Repeat 3
    "microtubule_binding_R4": (337, 368),   # Repeat 4
    "C-terminal": (369, 441),               # C-terminal region
}

# PHF (Paired Helical Filament) core regions (aggregation-prone)
PHF_CORE_REGIONS = {
    "PHF6_star": (275, 280),  # VQIINK (R2)
    "PHF6": (306, 311),        # VQIVYK (R3)
}

# Known phosphorylation sites (hyperphosphorylation related to AD)
PHOSPHORYLATION_SITES = [
    ("Ser199", 199),
    ("Ser202", 202),
    ("Thr205", 205),
    ("Thr212", 212),
    ("Ser214", 214),
    ("Thr217", 217),
    ("Thr231", 231),
    ("Ser235", 235),
    ("Ser396", 396),
    ("Ser404", 404),
    ("Ser422", 422),
]


class TauAnalyzer:
    """
    TAU Protein Analyzer.

    Analyzes TAU protein to identify targetable regions based on:
    1. Conservation (from MSA)
    2. Disorder (from structure prediction)
    3. Functional importance (literature)
    """

    def __init__(
        self,
        sequence_file: str = "data/tau/tau_sequence_info.txt",
        conservation_file: str = "data/tau/tau_entropy.npy",
        disorder_file: str = None,  # Optional pLDDT scores
    ):
        """
        Initialize TAU analyzer.

        Parameters
        ----------
        sequence_file : str
            Path to TAU sequence info file
        conservation_file : str
            Path to conservation scores (.npy file)
        disorder_file : str, optional
            Path to disorder scores (1 - pLDDT/100)
        """
        # Load sequence
        self.sequence = self._load_sequence(sequence_file)
        self.length = len(self.sequence)

        # Load conservation
        self.conservation = np.load(conservation_file)

        # Load or create disorder scores
        if disorder_file and Path(disorder_file).exists():
            self.disorder = np.load(disorder_file)
        else:
            # Create placeholder disorder scores based on known TAU properties
            # TAU is an intrinsically disordered protein (IDP)
            self.disorder = self._estimate_disorder()

    def _load_sequence(self, sequence_file: str) -> str:
        """Load TAU sequence from info file."""
        with open(sequence_file, 'r') as f:
            lines = f.readlines()

        # Find sequence (after "Sequence:" line)
        for i, line in enumerate(lines):
            if line.startswith("Sequence:"):
                sequence = ''.join(lines[i+1:]).replace('\n', '').replace(' ', '')
                return sequence

        raise ValueError(f"No sequence found in {sequence_file}")

    def _estimate_disorder(self) -> np.ndarray:
        """
        Estimate disorder scores based on known TAU properties.

        TAU is an intrinsically disordered protein (IDP).
        Known structured regions:
        - Microtubule binding repeats (R1-R4): slightly more structured
        - Rest: highly disordered

        Returns
        -------
        np.ndarray
            Disorder score per position (0 = structured, 1 = disordered)
        """
        disorder = np.ones(self.length) * 0.8  # Default: highly disordered

        # Microtubule binding repeats are slightly more structured
        for region_name, (start, end) in TAU_REGIONS.items():
            if "microtubule_binding" in region_name:
                # Adjust indices (convert to 0-based)
                start_idx = max(0, start - 1)
                end_idx = min(self.length, end)
                disorder[start_idx:end_idx] = 0.6  # Moderately disordered

        # PHF core regions (aggregation-prone) have some structure
        for region_name, (start, end) in PHF_CORE_REGIONS.items():
            start_idx = max(0, start - 1)
            end_idx = min(self.length, end)
            disorder[start_idx:end_idx] = 0.5  # Less disordered

        return disorder

    def compute_targetability_score(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> np.ndarray:
        """
        Compute targetability score combining conservation and disorder.

        T_i = α * (1 - disorder_i) + β * conservation_i

        Higher score = better target (conserved + structured)

        Parameters
        ----------
        alpha : float, default=0.5
            Weight for structure (1 - disorder)
        beta : float, default=0.5
            Weight for conservation

        Returns
        -------
        np.ndarray
            Targetability score per position
        """
        # Ensure same length
        min_len = min(len(self.conservation), len(self.disorder))
        conservation = self.conservation[:min_len]
        disorder = self.disorder[:min_len]

        # Compute targetability
        structure_score = 1.0 - disorder
        targetability = alpha * structure_score + beta * conservation

        return targetability

    def identify_hotspots(
        self,
        window_size: int = 10,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> List[Dict]:
        """
        Identify high-scoring target hotspots.

        Parameters
        ----------
        window_size : int, default=10
            Window size for smoothing scores
        top_k : int, default=5
            Number of top hotspots to return
        min_score : float, default=0.5
            Minimum score threshold

        Returns
        -------
        list of dict
            Hotspots with keys: start, end, score, sequence, region
        """
        targetability = self.compute_targetability_score()

        # Smooth with moving average
        smoothed = np.convolve(
            targetability,
            np.ones(window_size) / window_size,
            mode='same'
        )

        # Find peaks
        hotspots = []
        for i in range(len(smoothed) - window_size):
            window_score = smoothed[i:i+window_size].mean()

            if window_score >= min_score:
                hotspot = {
                    'start': i + 1,  # 1-indexed
                    'end': i + window_size,
                    'score': window_score,
                    'sequence': self.sequence[i:i+window_size],
                    'region': self._identify_region(i + 1),
                }
                hotspots.append(hotspot)

        # Sort by score and take top_k
        hotspots.sort(key=lambda x: x['score'], reverse=True)

        return hotspots[:top_k]

    def _identify_region(self, position: int) -> str:
        """Identify which functional region a position belongs to."""
        for region_name, (start, end) in TAU_REGIONS.items():
            if start <= position <= end:
                return region_name

        return "Unknown"


def select_target_regions(
    output_file: str = "data/tau/tau_target_regions.json",
    alpha: float = 0.4,  # Less weight on structure (TAU is IDP)
    beta: float = 0.6,   # More weight on conservation
) -> List[Dict]:
    """
    Select target regions for peptide design.

    Based on:
    1. Functional importance (PHF core, phosphorylation sites)
    2. Targetability score (conservation + structure)
    3. Literature (known binding sites)

    Parameters
    ----------
    output_file : str
        Path to save target regions JSON
    alpha : float, default=0.4
        Weight for structure
    beta : float, default=0.6
        Weight for conservation

    Returns
    -------
    list of dict
        Selected target regions
    """
    print("=" * 80)
    print("TAU TARGET REGION SELECTION")
    print("=" * 80)

    # Initialize analyzer
    analyzer = TauAnalyzer()

    print(f"\nTAU sequence length: {analyzer.length} aa")
    print(f"Disorder (mean): {analyzer.disorder.mean():.2f}")
    print(f"Conservation (mean): {analyzer.conservation.mean():.2f}")

    # Identify computational hotspots
    print("\n1. Identifying computational hotspots...")
    hotspots = analyzer.identify_hotspots(window_size=10, top_k=10, min_score=0.5)

    print(f"\nFound {len(hotspots)} hotspots:")
    for i, hs in enumerate(hotspots, 1):
        print(f"  {i}. Pos {hs['start']:3d}-{hs['end']:3d} | "
              f"Score: {hs['score']:.3f} | Region: {hs['region']}")

    # Add literature-based regions
    print("\n2. Adding literature-based target regions...")

    target_regions = []

    # PHF core regions (critical for aggregation)
    for region_name, (start, end) in PHF_CORE_REGIONS.items():
        end_idx = min(analyzer.length, end)
        target_regions.append({
            'name': region_name,
            'start': start,
            'end': end_idx,
            'sequence': analyzer.sequence[start-1:end_idx],
            'rationale': 'PHF core region - critical for TAU aggregation',
            'source': 'Literature (von Bergen et al., 2000)',
            'priority': 'High',
        })

    # Microtubule binding repeats (functional importance)
    for region_name, (start, end) in TAU_REGIONS.items():
        if 'microtubule_binding' in region_name:
            end_idx = min(analyzer.length, end)
            target_regions.append({
                'name': region_name,
                'start': start,
                'end': end_idx,
                'sequence': analyzer.sequence[start-1:end_idx],
                'rationale': 'Microtubule binding repeat - functional target',
                'source': 'Literature (Goedert & Spillantini, 2006)',
                'priority': 'Medium',
            })

    # Proline-rich region (phosphorylation sites)
    start, end = TAU_REGIONS['proline_rich_2']
    end_idx = min(analyzer.length, end)
    target_regions.append({
        'name': 'proline_rich_phospho',
        'start': start,
        'end': end_idx,
        'sequence': analyzer.sequence[start-1:end_idx],
        'rationale': 'Proline-rich region with multiple phosphorylation sites',
        'source': 'Literature (Buee et al., 2000)',
        'priority': 'Medium',
    })

    print(f"\nSelected {len(target_regions)} target regions")

    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(target_regions, f, indent=2)

    print(f"\nTarget regions saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SELECTED TARGET REGIONS")
    print("=" * 80)

    for i, region in enumerate(target_regions, 1):
        print(f"\n{i}. {region['name']} (Priority: {region['priority']})")
        print(f"   Position: {region['start']}-{region['end']} ({region['end']-region['start']+1} aa)")
        print(f"   Sequence: {region['sequence']}")
        print(f"   Rationale: {region['rationale']}")

    print("\n" + "=" * 80)
    print("TARGET SELECTION COMPLETE")
    print("=" * 80)

    return target_regions


if __name__ == "__main__":
    # Run target selection
    regions = select_target_regions()
