# Session Log

This file is maintained by Claude Code. It is updated at the end of each work section without prompting.

---

## 2026-03-27

- Initialized `docs/` folder and `session_log.md` for persistent session tracking.
- Created `CLAUDE.md` to auto-load this log and enforce update behavior on session end.
- Conducted full code-level workflow assessment of boltzgen pipeline (7 steps: design → inverse_folding → folding → design_folding → affinity → analysis → filtering).
- Identified extension points for developability features in schema.py, analyze.py, analyze_utils.py, filter.py, and boltzgen.py.
- Designed implementation plan for two features:
  1. Selectivity against decoys (Boltz2 iptm/pae as affinity proxy, decoys specified in design YAML).
  2. HIS-tag assessment (two-tier: sequence-level screen + explicit tag re-folding).
- Initialized Claude Code memory system with project goals and user profile.
- Plan saved at `~/.claude/plans/elegant-mapping-gray.md`.
- Wrote implementation plan to `docs/implementation_plan.md` for portability.

## 2026-04-14

- Created `feat/expression-tag-assessment` branch.
- Revised implementation plan: Tier 1 (fast geometric screen) runs in analysis/filtering for all designs; Tier 2 (expensive refolding) runs post-filtering only on top `--budget` finalists with multiple tag variants.
- Key design decisions: filter rejects if **either** terminus has clash risk; tag variants defined in design YAML (not CLI); both N and C termini always computed.
- Implemented Tier 1 (Phase 1):
  - `analyze_utils.py`: added `compute_tag_tier1_metrics()` — computes terminus-to-interface distance, SASA, and clash risk for both termini.
  - `analysis.yaml`: added `tag_assessment: false` config flag.
  - `analyze.py`: wired `tag_assessment` param and call into `compute_metrics()`.
  - `filter.py`: added `filter_tag_clash` param with hard filters on `tag_N_clash_risk` and `tag_C_clash_risk`.
  - `filtering.yaml`: added `filter_tag_clash: false` config flag.
  - `boltzgen.py`: added `--skip_tag_analysis` CLI arg, wired into analysis and filtering steps.
- Plan saved at `~/.claude/plans/mighty-wandering-alpaca.md`.
- Next steps: Phase 2 (Tier 2) — parse `tag_variants` from design YAML, implement tag sequence appending in `data_from_generated.py`, add `tag_folding` pipeline step, summary enrichment.
