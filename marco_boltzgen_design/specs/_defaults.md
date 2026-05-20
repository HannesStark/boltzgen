# ============================================================================== 
# Common defaults for BoltzGen design specs in this directory.
# BoltzGen does NOT read this file — it is documentation only.
#
# MARCO SRCR is highly polar and beta-rich. Prefer nanobody-specific generation,
# small diffusion batches for length diversity, H-bond/SASA-biased filtering, and
# a relaxed refolding RMSD threshold for VHH loop flexibility.
#
# Recommended CLI flags for `boltzgen run`:
#
#   boltzgen run spec.yaml \
#     --protocol nanobody-anything \
#     --output runs/<name> \
#     --num_designs N \
#     --diffusion_batch_size 2 \
#     --budget B \
#     --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 \
#     --refolding_rmsd_threshold 3.0
#
# Notes:
#   diffusion_batch_size   — keep at 1-2 during exploration because designs in the
#                            same diffusion batch share length.
#   plip_hbonds_refolded   — lower override value makes buried/interface H-bonds
#                            more important during filtering.
#   delta_sasa_refolded    — emphasizes buried interface area; useful for MARCO's
#                            polar SRCR surface.
#   refolding_rmsd_threshold — 3.0 is deliberately relaxed for flexible CDR loops.
#
# Optional diffusion tuning:
#   --diffusion.num_steps 200
#   --diffusion.guidance_scale 0.1
#   --diffusion.temperature 1.0
# ============================================================================== 
