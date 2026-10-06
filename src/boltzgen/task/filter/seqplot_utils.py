import os
import tempfile
import warnings
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import logomaker

from boltzgen.data import const

# Define amino acid properties for peptide visualization
HYDROPHOBIC = set("ACILMFWV")
NEUTRAL = set("GHPTY")
HYDROPHILIC = set("RNDQEK")
POSITIVE = set("RHK")
NEGATIVE = set("DE")
AA20 = list("ACDEFGHIKLMNPQRSTVWY")

COLOR_TABLE = {
    ("hydrophobic", "positive"): "#d62728",
    ("hydrophobic", "negative"): "#1f77b4",
    ("hydrophobic", "uncharged"): "#2ca02c",
    ("neutral", "positive"): "#ff7f0e",
    ("neutral", "negative"): "#17becf",
    ("neutral", "uncharged"): "#bcbd22",
    ("hydrophilic", "positive"): "#e377c2",
    ("hydrophilic", "negative"): "#1f99d4",
    ("hydrophilic", "uncharged"): "#9467bd",
}


def _hydropathy_class(res):
    if res in HYDROPHOBIC:
        return "hydrophobic"
    elif res in HYDROPHILIC:
        return "hydrophilic"
    else:
        return "neutral"


def _charge_class(res):
    if res in POSITIVE:
        return "positive"
    elif res in NEGATIVE:
        return "negative"
    else:
        return "uncharged"


hydrophobicity_colors = {
    aa: COLOR_TABLE[(_hydropathy_class(aa), _charge_class(aa))] for aa in AA20
}


def create_alignment_logo(sequences, name, width=10):
    """Create sequence logo from aligned sequences."""

    aligned_sequences, _ = align_peptide_sequences(sequences)
    if not aligned_sequences or len(aligned_sequences) < 2:
        msg = f"Warning: Not enough sequences for logo: {len(aligned_sequences)}"
        print(msg)
        return

    # Count amino acids at each position
    max_len = max(len(seq) for seq in aligned_sequences)
    counts = {aa: np.zeros(max_len, dtype=int) for aa in AA20}

    for i, seq in enumerate(aligned_sequences):
        for pos, aa in enumerate(seq):
            if pos < max_len and aa in AA20:
                counts[aa][pos] += 1

    # Convert to DataFrame
    counts_df = pd.DataFrame(counts)
    counts_df.index.name = "position"

    return draw_logo(counts_df, name, width=width)


def aa_composition_pie(sequences, name):
    # Generate amino acid composition analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Hydrophobicity distribution
    hydrophobicity_counts = {
        "hydrophobic": 0,
        "neutral": 0,
        "hydrophilic": 0,
    }
    for seq in sequences:
        for aa in seq:
            if aa in AA20:
                hydrophobicity_counts[_hydropathy_class(aa)] += 1

    if not any(hydrophobicity_counts.values()):
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No recognized amino acids", ha="center", va="center")
            ax.set_axis_off()
        fig.suptitle(name)
        return fig

    ax1.pie(
        hydrophobicity_counts.values(),
        labels=hydrophobicity_counts.keys(),
        autopct="%1.1f%%",
    )
    ax1.set_title(f"{name}\n Hydrophobicity Distribution", y=1.02, wrap=True)

    # Charge distribution
    charge_counts = {"positive": 0, "negative": 0, "uncharged": 0}
    for seq in sequences:
        for aa in seq:
            if aa in AA20:
                charge_counts[_charge_class(aa)] += 1

    ax2.pie(
        charge_counts.values(),
        labels=charge_counts.keys(),
        autopct="%1.1f%%",
    )
    ax2.set_title(f"{name}\n  Charge Distribution", y=1.02, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def align_peptide_sequences(sequences):
    """Align peptide sequences using BioPython's multiple sequence alignment."""

    if len(sequences) < 2:
        print("Warning: Only one sequence, no alignment needed")
        return sequences, []

    # Fallback to pairwise alignment (less optimal but functional)
    from Bio import Align
    from Bio.Align import substitution_matrices

    # Use BLOSUM62 matrix for protein alignment
    matrix = substitution_matrices.load("BLOSUM62")

    # Create pairwise aligner
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1

    # Start with first sequence as reference
    aligned_seqs = [sequences[0]]
    alignment_scores = []

    # Align each sequence to the reference
    for i, seq in enumerate(sequences[1:], 1):
        try:
            alignment = aligner.align(sequences[0], seq)
            if alignment:
                # Get the QUERY sequence (the different one) with gaps
                aligned_seq = str(alignment[0].query)

                aligned_seqs.append(aligned_seq)
                alignment_scores.append(alignment[0].score)
            else:
                aligned_seqs.append(seq)
                alignment_scores.append(0)
        except Exception as e:
            print(f"      DEBUG: Alignment failed: {e}")
            # Fallback: just add the sequence as-is
            aligned_seqs.append(seq)
            alignment_scores.append(0)
    return aligned_seqs, alignment_scores


def create_temp_fasta(sequences, names):
    """Create a temporary FASTA file from sequences."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        for i, (seq, name) in enumerate(zip(sequences, names)):
            f.write(f">{name}\n{seq}\n")
        return f.name


def build_region_alignment(chains, region_name):
    """Build alignment for a specific CDR region."""
    if not chains:
        return [], []

    all_positions = set()
    per_chain_pos_to_aa = []
    for ch in chains:
        reg = ch.regions.get(region_name, {})
        pos_to_aa = {str(pos): aa for pos, aa in reg.items()}
        per_chain_pos_to_aa.append(pos_to_aa)
        all_positions.update(pos_to_aa.keys())

    def pos_key(p):
        p = p.replace("H", "")
        num = ""
        ins = ""
        for c in p:
            if c.isdigit():
                num += c
            else:
                ins += c
        return (int(num), ins or "")

    ordered_positions = sorted(all_positions, key=pos_key)
    msa = []
    ids = []
    for i, ch in enumerate(chains):
        pos_to_aa = per_chain_pos_to_aa[i]
        row = "".join(pos_to_aa.get(p, "-") for p in ordered_positions)
        name = ch.name if getattr(ch, "name", None) else f"seq_{i + 1}"
        ids.append(name)
        msa.append(row)
    return ids, msa


def counts_matrix_from_msa(msa, alphabet):
    """Compute counts matrix from MSA."""
    if not msa:
        return pd.DataFrame()

    msa = [row.upper() for row in msa]
    L = len(msa[0])
    counts = {aa: np.zeros(L, dtype=int) for aa in alphabet}

    for row in msa:
        assert len(row) == L, "All rows must have same length"
        for j, aa in enumerate(row):
            if aa in ["-", "X", "J", "U", "O"]:
                continue
            if aa in counts:
                counts[aa][j] += 1

    df = pd.DataFrame(counts)
    df.index.name = "position"
    return df


def draw_logo(counts, title, width=10):
    """Draw sequence logo using logomaker."""
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    entropy = (
        -(probs.replace(0, np.nan) * np.log2(probs.replace(0, np.nan)))
        .sum(axis=1)
        .fillna(0)
    )
    info = np.log2(20) - entropy
    heights = probs.mul(info, axis=0)

    fig, ax = plt.subplots(figsize=(width, 4))
    logo = logomaker.Logo(
        heights,
        color_scheme=hydrophobicity_colors,
        shade_below=0.5,
        fade_below=0.5,
        vpad=0.05,
        width=0.8,
        ax=ax,
    )

    legend_elements = []
    for (hydro, charge), color in COLOR_TABLE.items():
        label = f"{hydro.capitalize()} + {charge}"
        legend_elements.append(Patch(facecolor=color, edgecolor="black", label=label))

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(title)
    ax.set_xlabel("Aligned position")
    ax.set_ylabel("Information (bits)")

    plt.tight_layout()
    return fig


def cdr_logo(sequences, name):
    try:
        from abnumber import Chain
    except ImportError:
        warnings.warn(
            "Skipping optional CDR logos: the abnumber package and its dependencies "
            "are required. Other sequence and liability plots are still available.",
            stacklevel=2,
        )
        return None

    # Create temporary FASTA
    names = [f"seq_{i + 1}" for i in range(len(sequences))]
    temp_fasta = create_temp_fasta(sequences, names)

    # Load chains using AbNumber
    chains = []
    for ch in Chain.from_fasta(
        temp_fasta,
        scheme="chothia",
        cdr_definition="chothia",
        as_generator=True,
        allowed_species=["alpaca"],
    ):
        if ch.chain_type == "H":
            chains.append(ch)

    # Generate composite figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    for ax, cdr in zip(axes, ["CDR1", "CDR2", "CDR3"]):
        ids, msa = build_region_alignment(chains, cdr)
        if not msa:
            ax.set_axis_off()
            continue

        counts = counts_matrix_from_msa(msa, AA20)
        probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        entropy = (
            -(probs.replace(0, np.nan) * np.log2(probs.replace(0, np.nan)))
            .sum(axis=1)
            .fillna(0)
        )
        info = np.log2(20) - entropy
        heights = probs.mul(info, axis=0)

        logo = logomaker.Logo(
            heights,
            ax=ax,
            color_scheme=hydrophobicity_colors,
            shade_below=0.5,
            fade_below=0.5,
            vpad=0.05,
            width=0.8,
        )
        ax.set_title(f"{name} {cdr}")
        ax.set_xlabel("")
        ax.set_ylabel("bits")

    # Add legend
    legend_elements = [
        Patch(
            facecolor=color,
            edgecolor="black",
            label=f"{hydro.capitalize()} + {charge}",
        )
        for (hydro, charge), color in COLOR_TABLE.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    os.unlink(temp_fasta)
    return fig


def plot_seq_liabilities(
    sequence,
    name,
    violations,
    total_score=None,
):
    """
    Plot sequence with colored backgrounds for each motif region.

    Args:
        sequence: The amino acid sequence
        violations: List of violation dictionaries
        total_score: Total liability score
    """
    n = len(sequence)
    sev_arr = [0] * n
    violation_types = {}  # Track unique violation types and their max severity

    # Process violations and compute positions for those that don't have explicit pos/len
    for v in violations:
        motif_name = v["motif"]

        # Handle violations that need position computation
        if motif_name == "ConsecIdentical" and v.get("pos") is None:
            # Find consecutive identical residues
            for i in range(n - 1):
                if sequence[i] == sequence[i + 1]:
                    # Color both positions
                    sev_arr[i] = max(sev_arr[i], v["severity"])
                    sev_arr[i + 1] = max(sev_arr[i + 1], v["severity"])
        elif motif_name == "LongHydrophobic" and v.get("pos") is None:
            # Find stretches of >4 consecutive hydrophobic residues
            hydrophobic_residues = set("FILVWY")
            for i in range(n - 4):
                stretch = sequence[i : i + 5]
                if all(aa in hydrophobic_residues for aa in stretch):
                    # Color the entire stretch
                    for j in range(i, i + 5):
                        if j < n:
                            sev_arr[j] = max(sev_arr[j], v["severity"])
        elif v.get("pos") and v.get("len"):
            # Standard violations with explicit positions
            for i in range(v["pos"] - 1, v["pos"] - 1 + v["len"]):
                if 0 <= i < n:  # Ensure index is within bounds
                    sev_arr[i] = max(sev_arr[i], v["severity"])

        # Track violation types for legend
        if motif_name not in violation_types:
            violation_types[motif_name] = v["severity"]
        else:
            violation_types[motif_name] = max(
                violation_types[motif_name], v["severity"]
            )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "HeatMap", ["white", "yellow", "red"]
    )
    norm = mcolors.Normalize(vmin=0, vmax=max(const.liability_severity.values()))

    # Wrap the already-scored full sequence to keep glyphs legible at PDF width.
    columns = 40
    row_count = max(1, (n + columns - 1) // columns)
    legend_columns = 2
    legend_rows = (len(violation_types) + legend_columns - 1) // legend_columns
    legend_height = max(1.5, 0.35 * legend_rows)
    fig, axes = plt.subplots(
        row_count + 2,
        1,
        figsize=(8.5, 1.2 * row_count + legend_height + 1),
        gridspec_kw={"height_ratios": [1.2] * row_count + [0.3, legend_height]},
        squeeze=False,
    )
    sequence_axes = axes[:row_count, 0]
    ax_colorbar, ax_legend = axes[-2:, 0]
    for row, ax in enumerate(sequence_axes):
        start = row * columns
        stop = min(start + columns, n)
        for idx in range(start, stop):
            ax.text(
                idx - start,
                0.4,
                sequence[idx],
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(
                    facecolor=cmap(norm(sev_arr[idx])),
                    edgecolor="none",
                    boxstyle="square,pad=0.1",
                ),
            )
        ax.set_xlim(-0.5, max(1, min(n, columns)) - 0.5)
        ax.set_ylim(0, 1)
        ax.axis("off")
        if n:
            ax.text(
                0,
                0.95,
                f"Residues {start + 1}-{stop}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
            )

    if total_score is not None:
        name += f" (Score: {total_score})"
    sequence_axes[0].set_title(name)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=ax_colorbar, orientation="horizontal", label="Severity")

    # Legend showing violation types with motif patterns
    if violation_types:
        ax_legend.axis("off")
        legend_items = []
        legend_labels = []

        # Sort by severity (descending)
        sorted_violations = sorted(
            violation_types.items(), key=lambda x: x[1], reverse=True
        )

        # Motif pattern mapping for legend
        motif_patterns = {
            "DeAmdH": "N[GS]",
            "DeAmdM": "N[AHNT]",
            "DeAmdL": "[STK]N",
            "Ngly": "N[^P][ST]",
            "Isom": "D[DG HST]",
            "Isomer": "DG|DS|DD",
            "FragH": "DP",
            "FragM": "TS",
            "TrpOx": "W",
            "MetOx": "M",
            "Hydro": "NP",
            "IntBind": "GPR|RGD|RYD|LDV|DGE|KGD|NGR",
            "Polyreactive": "GGG|WWW|GG|RR|VG|VVV|YY|WxW",
            "AggPatch": "FHW",
            "ViscPatch": "HYF|HWH",
            "UnpairedCys": "C",
            "HighNetCharge": "net charge > +1",
            "AspBridge": "N[GSQA]",
            "AspCleave": "D[PGS]",
            "NTCycl": "^[QN]",
            "ProtTryp": "[KR](?=.)",
            "DPP4": "^[PX]?[AP]",
            "CysOx": "C",
            "HydroPatch": "[FILVWY]{3,}",
            "LowHydrophilic": "< 40% hydrophilic",
            "ConsecIdentical": "consecutive identical",
            "LongHydrophobic": "> 4 consecutive hydrophobic",
        }

        for motif_name, severity in sorted_violations:
            color = cmap(norm(severity))
            legend_items.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black")
            )
            pattern = motif_patterns.get(motif_name, "unknown")
            legend_labels.append(f"{motif_name}: {pattern}")

        ax_legend.legend(
            legend_items,
            legend_labels,
            loc="center",
            title="Violations",
            ncol=legend_columns,
            fontsize=8,
        )
    else:
        # No violations detected
        ax_legend.axis("off")
        ax_legend.text(
            0.5,
            0.5,
            "No violations detected",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        )

        ax_legend.set_title("Violations", fontsize=10)

    plt.tight_layout()

    return fig
