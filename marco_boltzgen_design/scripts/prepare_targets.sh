#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./scripts/prepare_targets.sh /path/human.cif A /path/mouse.cif A

HUMAN_SRC="${1:-}"
HUMAN_CHAIN="${2:-A}"
MOUSE_SRC="${3:-}"
MOUSE_CHAIN="${4:-A}"

if [[ -z "$HUMAN_SRC" || -z "$MOUSE_SRC" ]]; then
  echo "Usage: $0 <human_structure.(cif|pdb)> <human_chain> <mouse_structure.(cif|pdb)> <mouse_chain>"
  exit 1
fi

mkdir -p ../targets
cp "$HUMAN_SRC" ../targets/human_MARCO_input.cif
cp "$MOUSE_SRC" ../targets/mouse_Marco_input.cif

cat > ../targets/target_manifest.tsv <<MANIFEST
species\tlabel\tstructure_file\tprimary_chain\tnotes
human\thuman_MARCO\thuman_MARCO_input.cif\t${HUMAN_CHAIN}\tfill SRCR domain residue ranges in README checklist
mouse\tmouse_Marco\tmouse_Marco_input.cif\t${MOUSE_CHAIN}\tfill SRCR domain residue ranges in README checklist
MANIFEST

echo "Prepared targets/ and manifest."
