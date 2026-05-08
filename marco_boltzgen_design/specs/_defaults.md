# ==============================================================================
# Common defaults for BoltzGen design specs in this directory.
# BoltzGen does NOT read this file — it is documentation only.
#
# Recommended CLI flags for `boltzgen run` (override per spec via EXTRA_ARGS env var):
#
#   boltzgen run spec.yaml --num_designs N --budget B --devices G \
#     --diffusion.num_steps 200 \
#     --diffusion.guidance_scale 0.0 \
#     --diffusion.temperature 1.0
#
# diffusion:
#   num_steps      — diffusion steps (200 = default, 300+ for harder targets)
#   guidance_scale — classifier-free guidance weight (0 = no guidance; 0.1–0.5 for
#                     stricter hotspot adherence; too high can distort geometry)
#   temperature    — sampling temperature (0.8 = sharper/more compact; 1.2 = diverse)
#
# convergence:
#   patience       — early stopping patience (default 10)
#   min_steps      — minimum diffusion steps before convergence check (default 50)
# ==============================================================================

# YAML anchor alias (use in specs via: *defaults)
# Not supported by BoltzGen — use as documentation only.