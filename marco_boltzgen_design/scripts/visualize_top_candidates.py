"""
PyMOL visualization script for MARCO nanobody candidates
==========================================================
Ranks 1-3 from Boltz API results docked onto human MARCO SRCR (2OYA).

Usage in PyMOL:
    run boltzgen/marco_boltzgen_design/scripts/visualize_top_candidates.py

Or from command line:
    pymol -r visualize_top_candidates.py

Chains:
  2OYA (crystal structure):  Chain A = human MARCO SRCR
  Boltz API structures:      Chain A = MARCO (100 aa), Chain B = nanobody (~134 aa)

Epitope positions (1-indexed in both 2OYA and Boltz API chain A):
  R5, I6, G8, S10, R12, G13, A15, Y18, Y19, S20,
  G48, A50, Y52, V96, S99

Author: Auto-generated for MARCO nanobody design campaign
Date: 2026-07-10
"""

import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "boltz_api_nanobody_results_2026-07-09")
TOP3_CIFS = [
    ("pres_7tckIrdh48jvmrRIDKW2", "batch4", 0.941, 0.7,  0.727),  # Rank 1
    ("pres_vG1sxNuCwnvUtHurTTvP", "batch3", 0.932, 0.9,  0.615),  # Rank 2
    ("pres_EZoV78yHit3sQ3rBaLdK", "batch1", 0.931, 1.0,  0.724),  # Rank 3
]
TWOYA_PDB = os.path.join(BASE, "..", "targets", "2OYA.pdb")

# Epitope residues (1-indexed, same in 2OYA and Boltz API chain A)
EPITOPE_RESIS = ["5", "6", "8", "10", "12", "13", "15", "18", "19", "20",
                 "48", "50", "52", "96", "99"]

# ── SCRIPT BODY ───────────────────────────────────────────────────────────────
cmd.reinitialize()

# 1. Load 2OYA crystal structure
cmd.load(TWOYA_PDB, object="crystal_2oya")
cmd.hide(representation="everything", selection="crystal_2oya")
cmd.show(representation="cartoon", selection="crystal_2oya")
cmd.color("marine", "crystal_2oya")
util.cbam("crystal_2oya", "0x4040ff", "0xff8080")
print("[OK] Loaded 2OYA crystal structure")

# 2. Highlight epitope residues on 2OYA
epitope_sel = "crystal_2oya and resi " + "+".join(EPITOPE_RESIS)
cmd.show(representation="sticks", selection=epitope_sel)
cmd.color("orange", epitope_sel)
cmd.set("stick_color", "orange", epitope_sel)
cmd.set("stick_radius", 0.15, epitope_sel)
print(f"[OK] Epitope residues highlighted: {', '.join(EPITOPE_RESIS)}")

# 3. Load top 3 Boltz API nanobody candidates
for i, (cand_id, batch, ipTM, PAE, conf) in enumerate(TOP3_CIFS, 1):
    cif_path = os.path.join(DATA_DIR, "structures", batch, f"{cand_id}_predicted.cif")
    if not os.path.exists(cif_path):
        print(f"[WARN] File not found: {cif_path}")
        continue

    obj_name = f"nb{i}_{cand_id[:8]}"
    cmd.load(cif_path, object=obj_name)

    # Chains: A = MARCO target, B = nanobody
    # Color scheme per rank
    colors = ["green", "cyan", "salmon"]
    nanobody_color = colors[i - 1]

    cmd.hide(representation="everything", selection=obj_name)
    cmd.show(representation="cartoon", selection=f"{obj_name} and chain B")
    cmd.color(nanobody_color, f"{obj_name} and chain B")
    cmd.util.cbam(f"{obj_name} and chain B")
    cmd.show(representation="lines", selection=f"{obj_name} and chain B")

    # MARCO chain A in this complex — make it faded
    cmd.show(representation="cartoon", selection=f"{obj_name} and chain A")
    cmd.color("gray70", f"{obj_name} and chain A")
    cmd.set("cartoon_transparency", 0.7, selection=f"{obj_name} and chain A")

    print(f"[OK] Loaded {obj_name}: ipTM={ipTM}, PAE={PAE}, conf={conf}")

# 4. Create epitope label
cmd.label(selection=epitope_sel, expression="resi", format="resi")

# 5. Create groups
cmd.group("nanobodies", members="nb1_* nb2_* nb3_*")
cmd.group("epitope", members=epitope_sel)

# 6. Orient the view to 2OYA chain A
cmd.zoom(selection="crystal_2oya", animate=0)
cmd.center("crystal_2oya")
cmd.move("z", -20)

# 7. Legend / info
print("""
╔══════════════════════════════════════════════════════════════════╗
║  MARCO Nanobody Candidates — Boltz API Top 3                     ║
╠══════════════╦═══════════╦═══════╦══════════╦═══════════════════╣
║  #1 nb1_7tck  ║  ipTM=0.941 ║ PAE=0.7  ║ conf=0.727 ║ green          ║
║  #2 nb2_vG1s  ║  ipTM=0.932 ║ PAE=0.9  ║ conf=0.615 ║ cyan          ║
║  #3 nb3_EZoV  ║  ipTM=0.931 ║ PAE=1.0  ║ conf=0.724 ║ salmon         ║
╠══════════════╩═══════════╩═══════╩══════════╩═════════════════════╣
║  Orange sticks = epitope residues (beta-edge of MARCO SRCR)       ║
║  Cartoon = full 2OYA crystal structure + Boltz models              ║
║  Nanobody chain B shown; chain A (MARCO) faded for clarity        ║
╚══════════════════════════════════════════════════════════════════╝
""")

# 8. Pretty settings
cmd.set("ray_opaque_background", 0)
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_ring_mode", 3)
cmd.set("dash_width", 2.0)
cmd.refresh()
print("[OK] Visualization complete. Zoom/rotate as needed.")