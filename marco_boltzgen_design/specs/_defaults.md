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
#
# ==============================================================================
# N-GLYCOSYLATION SITE EXCLUSION (BoltzProt-1 best practice)
# ==============================================================================
# All 36 NXS/T sequons are EXCLUDED at generation time, not filtered post-hoc.
# This doubles confirmed-binder rate vs applying the filter only at ranking.
# Source: BoltzProt-1 Technical Report Appendix E.2, Listing 1 lines 27–32.
#
# In the YAML binder_specification.rules, add:
#
#   rules:
#     excluded_sequence_motifs:
#       - NAS,NAT,NCS,NCT,NDS,NDT,NES,NET,NFS,NFT
#       - NGS,NGT,NHS,NHT,NIS,NIT,NKS,NKT,NLS,NLT
#       - NMS,NMT,NNS,NNT,NQS,NQT,NRS,NRT,NSS,NST
#       - NTS,NTT,NVS,NVT,NWS,NWT,NYS,NYT
#
# This is applied automatically when using --protocol nanobody-anything with
# --binder_specification boltz_curated. For custom specs, add explicitly.
# ============================================================================== 
