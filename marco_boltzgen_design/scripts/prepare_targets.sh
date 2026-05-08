#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./scripts/prepare_targets.sh /path/human.cif A /path/mouse.cif A

HUMAN_SRC="${1:-}"
HUMAN_CHAIN="${2:-A}"
MOUSE_SRC="${3:-}"
MOUSE_CHAIN="${4:-A}"

if [[ -z "$HUMAN_SRC" || -z "$MOUSE_SRC" ]]; then
  echo "Usage: $0 <human_structure.(cif|pdb)> <human_chain> <mouse_structure.(cif|pdb)> <mouse_chain>" >&2
  exit 1
fi

# Validate inputs are readable
for label in "human" "mouse"; do
  src_var="${label}_SRC"
  src_val="${!src_var}"
  if [[ ! -r "$src_val" ]]; then
    echo "ERROR: ${label} structure not found or not readable: $src_val" >&2
    exit 1
  fi
  case "${src_val}" in
    *.cif|*.mmcif|*.pdb) ;;   # known extensions
    *)
      echo "WARNING: ${label} file does not have .cif/.mmcif/.pdb extension: $src_val" >&2
      ;;
  esac
done

# Derive absolute paths from script location to avoid relative-path fragility
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGETS_DIR="${SCRIPT_DIR}/../targets"
mkdir -p "$TARGETS_DIR"
cp "$HUMAN_SRC" "$TARGETS_DIR/human_MARCO_input.cif"
cp "$MOUSE_SRC" "$TARGETS_DIR/mouse_Marco_input.cif"

cat > "$TARGETS_DIR/target_manifest.tsv" <<MANIFEST
species\tlabel\tstructure_file\tprimary_chain\tnotes
human\thuman_MARCO\thuman_MARCO_input.cif\t${HUMAN_CHAIN}\tfill SRCR domain residue ranges in README checklist
mouse\tmouse_Marco\tmouse_Marco_input.cif\t${MOUSE_CHAIN}\tfill SRCR domain residue ranges in README checklist
MANIFEST

echo "Prepared $TARGETS_DIR and manifest."
