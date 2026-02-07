"""
Download TAU Protein Sequences from UniProt.

Downloads human TAU isoforms and mammalian orthologs for MSA construction.
"""

import requests
from pathlib import Path
from typing import List, Dict
import time


# Human TAU isoforms (UniProt P10636)
HUMAN_TAU_ISOFORMS = {
    "P10636-1": "Tau-F (2N4R) - longest isoform, 441 aa",
    "P10636-2": "Tau-E (1N4R) - 412 aa",
    "P10636-3": "Tau-D (0N4R) - 383 aa",
    "P10636-4": "Tau-C (2N3R) - 410 aa",
    "P10636-5": "Tau-B (1N3R) - 381 aa",
    "P10636-6": "Tau-A (0N3R) - 352 aa (shortest brain isoform)",
}

# Mammalian TAU orthologs (for MSA)
TAU_ORTHOLOGS = {
    "P10636": ("Homo sapiens", "Human"),
    "P10637": ("Mus musculus", "Mouse"),
    "P19332": ("Rattus norvegicus", "Rat"),
    "O02804": ("Bos taurus", "Bovine"),
    "P50428": ("Gallus gallus", "Chicken"),
    "Q4R572": ("Macaca fascicularis", "Macaque"),
}


def download_uniprot_fasta(uniprot_id: str, output_dir: Path) -> str:
    """
    Download FASTA sequence from UniProt.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession ID (e.g., 'P10636' or 'P10636-1' for isoform)
    output_dir : Path
        Output directory

    Returns
    -------
    str
        Path to downloaded FASTA file
    """
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"

    print(f"Downloading {uniprot_id}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save FASTA
        output_file = output_dir / f"{uniprot_id}.fasta"
        with open(output_file, 'w') as f:
            f.write(response.text)

        # Extract sequence info
        lines = response.text.strip().split('\n')
        header = lines[0]
        sequence = ''.join(lines[1:])

        print(f"  Downloaded: {len(sequence)} aa")
        print(f"  Header: {header[:80]}...")

        return str(output_file)

    except Exception as e:
        print(f"  Error downloading {uniprot_id}: {e}")
        return None


def download_tau_sequences(output_dir: str = "data/tau"):
    """
    Download all TAU sequences for MSA construction.

    Parameters
    ----------
    output_dir : str
        Output directory for FASTA files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TAU SEQUENCE DOWNLOAD")
    print("=" * 80)

    # Download human isoforms
    print("\n1. Downloading Human TAU Isoforms")
    print("-" * 80)

    human_files = []
    for isoform_id, description in HUMAN_TAU_ISOFORMS.items():
        print(f"\n{description}")
        fasta_file = download_uniprot_fasta(isoform_id, output_dir)
        if fasta_file:
            human_files.append(fasta_file)
        time.sleep(1)  # Be nice to UniProt servers

    # Download orthologs
    print("\n" + "=" * 80)
    print("2. Downloading Mammalian Orthologs")
    print("-" * 80)

    ortholog_files = []
    for uniprot_id, (species_sci, species_common) in TAU_ORTHOLOGS.items():
        print(f"\n{species_common} ({species_sci})")
        fasta_file = download_uniprot_fasta(uniprot_id, output_dir)
        if fasta_file:
            ortholog_files.append(fasta_file)
        time.sleep(1)

    # Merge all sequences into one file for MSA
    print("\n" + "=" * 80)
    print("3. Creating Combined FASTA")
    print("-" * 80)

    combined_file = output_dir / "tau_all_sequences.fasta"
    with open(combined_file, 'w') as outf:
        all_files = human_files + ortholog_files
        for fasta_file in all_files:
            if fasta_file:
                with open(fasta_file, 'r') as inf:
                    content = inf.read()
                    outf.write(content)
                    if not content.endswith('\n'):
                        outf.write('\n')

    print(f"\nCombined FASTA saved to: {combined_file}")
    print(f"Total sequences: {len(all_files)}")

    # Download canonical human TAU (longest isoform) separately
    print("\n" + "=" * 80)
    print("4. Downloading Canonical Human TAU (2N4R)")
    print("-" * 80)

    canonical_file = download_uniprot_fasta("P10636-1", output_dir / ".." / "..")
    if canonical_file:
        # Copy to data/tau as well
        import shutil
        shutil.copy(canonical_file, output_dir / "tau_2N4R_human.fasta")
        print(f"\nCanonical TAU saved to: {output_dir / 'tau_2N4R_human.fasta'}")

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nFiles created:")
    print(f"  1. {combined_file} - All sequences for MSA")
    print(f"  2. {output_dir / 'tau_2N4R_human.fasta'} - Canonical human TAU")
    print(f"  3. Individual FASTA files in {output_dir}/")

    print("\nNext steps:")
    print("  1. Build MSA with: scripts/build_tau_msa.py")
    print("  2. Run ESMFold on canonical TAU")
    print("  3. Compute conservation and disorder scores")


if __name__ == "__main__":
    download_tau_sequences()
