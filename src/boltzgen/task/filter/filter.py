import json
from boltzgen.utils.quiet import quiet_startup


quiet_startup()
from typing import Dict
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from pathlib import Path
from Bio import Align
import numpy as np
import random
from tqdm.auto import tqdm
import heapq
from textwrap import fill, dedent
import re

from boltzgen.task.analyze.analyze_utils import compute_liability_scores
from boltzgen.task.filter.seqplot_utils import (
    aa_composition_pie,
    cdr_logo,
    create_alignment_logo,
    plot_seq_liabilities,
)
from boltzgen.task.task import Task


class Filter(Task):
    """
    This step consumes the aggregated metrics produced by **Analyze** (i.e.
    ``aggregate_metrics_*.csv`` in ``design_dir``), applies hard
    filtering thresholds, ranks candidates by a composite quality key, and then
    performs a lazy-greedy **diversity selection** over sequence identity.
    It writes:
        • the top-`budget` quality+diversity designs (+ refolded CIFs),
        • a CSV with all metrics and ranks,
        • a summary PDF with tables, histograms, scatter plots, and (optionally)
            sequence logos and liability heat-maps.

    The step is designed to be **fast** (≈30 s per 100k designs on typical CPUs);
    the most expensive part is PDF plotting.

    Parameters
    ----------
    design_dir : str | Path
        Directory containing design artifacts and metrics, including:
        - ``aggregate_metrics_*.csv`` (required)
        - ``refold_cif/`` (for copying refolded structures)
        - ``ca_coords_sequences.pkl.gz`` (id→sequence mapping for diversity)
    outdir : str | Path, default=None
        Defaults to design_dir. Parent directory where results will be written. The step creates
        ``{outdir}/final_ranked_designs`` and subfolders.
    budget : int, default=30
        Number of designs to select by quality+diversity (Diverse set).
    use_affinity : bool, default=False
        Switch to affinity-oriented metrics (for small-molecule binders).
    from_inverse_folded : bool, default=True
        If True, use backbone RMSDs (``bb_rmsd``) for filtering and delta-SASA from
        refolded structures. If False, use all-atom RMSDs and original ΔSASA.
    filter_designfolding : bool, default=True
        Also require the isolated design (no target) to refold to the same shape,
        via ``designfolding-*_rmsd``.
    filter_bindingsite : bool, default=False
        Keep only designs with at least one residue within the specified distance of
        a binding-site residue (e.g., ``bindsite_under_8rmsd > 0``).
    filter_target_aligned : bool, default=False
        Require target backbone alignment (``bb_target_aligned<2.5`` flag).
    filter_biased : bool, default=True
        Remove amino-acid composition outliers (default caps on ALA/GLY/GLU/LEU/VAL).
    refolding_rmsd_threshold : float, default=2.5
        Threshold used for the RMSD-based filters (lower is better).
    modality : {"peptide","antibody"}, default="peptide"
        Affects liability scoring and optional sequence visualizations.
    alpha : float in [0,1], default=0.1
        Trade-off for sequence diversity selection: 0=quality-only, 1=diversity-only.
    metrics_override : Dict[str, float | None], default=None
        Per-metric *inverse-importance* weights for ranking.
        - A larger value **down-weights** that metric’s rank (rank / weight).
        - ``None`` removes the metric entirely from the ranking key.
        Example: ``{"plip_hbonds_refolded": 4, "delta_sasa_refolded": 2, "neg_min_design_to_target_pae": 1}``.
    num_liability_plots : int, default=0
        If >0, produce per-residue developability heat-maps for the first N top designs.
    plot_seq_logos : bool, default=False
        If True, include alignment logos and AA composition pies for All/Top/Diverse sets.
    additional_filters : list[Dict], default=[]
        Extra hard filters of the form:
        ``{"feature": "<column>", "lower_is_better": bool, "threshold": float}``.
    size_buckets : list[Dict], default=[]
        Optional constraint for the maximum number of designs returned in a certain size range:
        ``{"num_designs": int, "min": int, "max": int}``.

    Ranking & Diversity
    -------------------
    **Filtering.** Each design must pass all hard thresholds (``pass_<feature>_filter``).
    The step also adds convenience columns (e.g., ``filter_rmsd``,
    ``designfolding-filter_rmsd``, signed variants like ``neg_min_design_to_target_pae``).

    **Ranking (quality).** For each metric in ``self.metrics``, compute the row's
    rank on the tuple ``(num_filters_passed, metric)`` (so designs that fail filters
    are pushed down), then divide by the metric’s *inverse-importance* weight.
    The **worst** (max) scaled rank across metrics becomes the design’s
    quality key. The Top set is the best `budget` designs by this key (tie-broken
    by iPTM).

    **Diversity.** A lazy-greedy selection chooses `div_budget` designs maximizing:
    ``(1 - alpha) * quality + alpha * (1 - seq_identity)``.
    Sequence identity is computed via pairwise alignment on the full chain or the
    designed segment (auto-chosen based on typical length ratio). Optional
    ``size_buckets`` limit the number of selections per length range.

    Inputs
    ------
    • ``design_dir/aggregate_metrics_*.csv`` (required)
    • ``design_dir/ca_coords_sequences.pkl.gz`` (required for diversity)
    • ``design_dir/refold_cif/`` (optional but recommended, copied to outputs)

    Outputs
    -------
    On disk (under ``{outdir}/final_ranked_designs``):
        • ``final_{budget}_designs`` folder with .mmcif files
        • ``metrics_*.csv`` – full table with ranks/flags
        • ``diverse_selected_{div_budget}.csv`` – rows/IDs for Diverse set
        • ``results_overview_*.pdf`` – summary report with tables/plots

    Notes
    -----
    - ``metrics_override`` lets you:
        * increase a weight (de-emphasize that metric),
        * set a metric to ``None`` (remove from ranking),
        * include a new metric that is already in the csv for ranking.

    Examples
    --------
    >>> from boltzgen.task.filter.filter import Filter
    >>> filter = Filter(
    ...     design_dir="workbench/run123",
    ...     budget=50, div_budget=30, alpha=0.25,
    ...     metrics_override={"design_ptm": 2, "neg_min_design_to_target_pae": 1},
    ...     additional_filters=[{"feature":"design_ptm","lower_is_better":False,"threshold":0.7}],
    ... )
    >>> filter.run(jupyter_nb=False)  # writes CSVs, PDF, and copies selected structures
    """

    def __init__(
        self,
        design_dir: str,
        budget: int = 30,
        top_budget: int = 10,
        outdir: str = None,
        use_affinity: bool = False,  # This changes the filtering metrics to metrics more amenable to small molecule binder design
        filter_cysteine: bool = True,  # This filters out all designs that have designed cysteins in them (prespecified cysteins in the design are not counted)
        from_inverse_folded: bool = True,  # This makes it so that we use the backbone refolding rmsd instead of the all-atom RMSD
        filter_designfolding: bool = True,  # Additionally filter based on the RMSD from refolding the design in isolation. This makes sure the design has the same shape with and without the target being present.
        filter_bindingsite: bool = False,  # This filters out everything that does not have a residue within 4A of a binding site residue
        filter_target_aligned: bool = False,
        filter_biased: bool = True,  # This filters out sequences that are alanine rich, 30% alanine is threshold
        refolding_rmsd_threshold: float = 2.5,
        modality: str = "peptide",  # peptide, antibody
        peptide_type: str = "linear",  # linear, cyclic
        alpha: float = 0.1,  # 0 = quality-only, 1 = diversity-only
        random_state: int = 0,
        metrics_override: Dict = None,  # overrides metrics, None values delete keys
        num_liability_plots: int = 0,
        plot_seq_logos: bool = False,  # make sequence logo diagrams of designed sequence
        additional_filters: list[
            Dict
        ] = [],  # For example: [{"feature": "design_ALA", "lower_is_better": True, "threshold": 0.3}],
        size_buckets: list[Dict] = [],
    ):
        super().__init__()
        assert modality in ["peptide", "antibody"]
        assert peptide_type in ["linear", "cyclic"]
        self.design_dir = Path(design_dir)
        self.top_budget = top_budget
        self.use_affinity = use_affinity
        self.filter_cysteine = filter_cysteine
        self.from_inverse_folded = from_inverse_folded
        self.filter_bindingsite = filter_bindingsite
        self.budget = budget
        self.alpha = alpha
        self.random_state = random_state
        self.num_liability_plots = num_liability_plots
        self.plot_seq_logos = plot_seq_logos
        self.modality = modality
        self.peptide_type = peptide_type
        self.size_buckets = size_buckets

        if outdir is None:
            outdir = design_dir
        self.outdir = Path(f"{outdir}") / "final_ranked_designs"
        self.top_dir = self.outdir / f"intermediate_ranked_{top_budget}_designs"
        self.div_dir = self.outdir / f"final_{budget}_designs"
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.top_dir.mkdir(parents=True, exist_ok=True)
        self.div_dir.mkdir(parents=True, exist_ok=True)

        # we want to maximize all these metrics
        self.metrics: dict = {
            "design_to_target_iptm": 1,
            "design_ptm": 1,
            "neg_min_design_to_target_pae": 1,
            "plip_hbonds" + ("_refolded" if from_inverse_folded else ""): 2,
            "plip_saltbridge" + ("_refolded" if from_inverse_folded else ""): 2,
            "delta_sasa_refolded" if from_inverse_folded else "delta_sasa_original": 2,
        }
        if use_affinity:
            self.metrics: dict = {
                "design_to_target_iptm": 1.1,
                "design_ptm": 1.1,
                "neg_min_design_to_target_pae": 1.1,
                "affinity_probability_binary1": 1,
                "plip_hbonds" + ("_refolded" if from_inverse_folded else ""): 2,
                "plip_saltbridge" + ("_refolded" if from_inverse_folded else ""): 2,
                "delta_sasa_refolded"
                if from_inverse_folded
                else "delta_sasa_original": 2,
            }

        # override metrics
        if not metrics_override is None:
            for k in metrics_override:
                if metrics_override[k] is None:
                    del self.metrics[k]
                else:
                    self.metrics[k] = metrics_override[k]

        # Define how to Filter
        self.filters = [
            {"feature": "has_x", "lower_is_better": True, "threshold": 0},
            {
                "feature": "filter_rmsd",
                "lower_is_better": True,
                "threshold": refolding_rmsd_threshold,
            },
            {
                "feature": "filter_rmsd_design",
                "lower_is_better": True,
                "threshold": refolding_rmsd_threshold,
            },
        ]
        if filter_designfolding:
            self.filters.append(
                {
                    "feature": "designfolding-filter_rmsd",
                    "lower_is_better": True,
                    "threshold": refolding_rmsd_threshold,
                }
            )
        if filter_bindingsite:
            self.filters.append(
                {
                    "feature": "bindsite_under_8rmsd",  # center_coord RMSD
                    "lower_is_better": False,
                    "threshold": 0.0001,  # at least one binding site residue
                },
            )
        if filter_target_aligned:
            self.filters.append(
                {
                    "feature": "bb_target_aligned<2.5",
                    "lower_is_better": False,
                }
            )
        if filter_cysteine:
            self.filters.append(
                {
                    "feature": "CYS_fraction",
                    "lower_is_better": True,
                    "threshold": 0,
                },
            )
        if filter_biased:
            self.filters.extend(
                [
                    {
                        "feature": "ALA_fraction",
                        "lower_is_better": True,
                        "threshold": 0.3,
                    },
                    {
                        "feature": "GLY_fraction",
                        "lower_is_better": True,
                        "threshold": 0.3,
                    },
                    {
                        "feature": "GLU_fraction",
                        "lower_is_better": True,
                        "threshold": 0.3,
                    },
                    {
                        "feature": "LEU_fraction",
                        "lower_is_better": True,
                        "threshold": 0.3,
                    },
                    {
                        "feature": "VAL_fraction",
                        "lower_is_better": True,
                        "threshold": 0.3,
                    },
                ]
            )
        self.filters.extend(additional_filters)

        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def run(self, config=None, jupyter_nb=False):
        self.load_dataframe()
        self.reset_outdir()
        self.filter_df()
        self.absolute_metrics()
        self.sort_df()
        self.optimize_diversity()
        self.write_outdir()

        # Visualizations
        print(
            "\nWriting design files is done. Now making plots for a final summary .pdf file with statistics."
        )
        (
            hist_metrics,
            extra_pairs,
            row_headers,
            rows,
            metric_rows,
            intro_text,
            csv_expl_rows,
        ) = self.prepare_visualization()
        self.make_visualization(
            hist_metrics,
            extra_pairs,
            row_headers,
            rows,
            metric_rows,
            intro_text,
            csv_expl_rows,
            jupyter_nb=jupyter_nb,
        )

    def reset_outdir(self):
        if self.outdir.exists():
            shutil.rmtree(self.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.top_dir.mkdir(parents=True, exist_ok=True)
        self.div_dir.mkdir(parents=True, exist_ok=True)

    def load_dataframe(self):
        csv_files = sorted(self.design_dir.glob("aggregate_metrics_*.csv"))
        if not csv_files:
            msg = (
                f"No file starting with 'aggregate_metrics_' found in {self.design_dir}"
            )
            raise FileNotFoundError(msg)
        df_in = pd.read_csv(csv_files[0])

        self.df_in = df_in.copy()
        df = df_in.copy()

        if self.from_inverse_folded:
            df["filter_rmsd"] = df["bb_rmsd"]
            df["filter_rmsd_design"] = df["bb_rmsd_design"]
        else:
            df["filter_rmsd"] = df["rmsd"]
            df["filter_rmsd_design"] = df["rmsd_design"]
        if "designfolding-rmsd" in df:
            df["designfolding-filter_rmsd"] = df["designfolding-rmsd"]
        if "designfolding-bb_rmsd" in df and self.from_inverse_folded:
            df["designfolding-filter_rmsd"] = df["designfolding-bb_rmsd"]
        if "min_design_to_target_pae" in df:
            df["neg_min_design_to_target_pae"] = -df["min_design_to_target_pae"]
        if "design_hydrophobicity" in df:
            df["neg_design_hydrophobicity"] = -df["design_hydrophobicity"]
        if "design_largest_hydrophobic_patch_refolded" in df:
            df["neg_design_largest_hydrophobic_patch_refolded"] = -df[
                "design_largest_hydrophobic_patch_refolded"
            ]
        df["neg_min_interaction_pae"] = -df["min_interaction_pae"]
        df["has_x"] = df["designed_sequence"].str.contains("X")
        self.df = df

        print(f"Total number of designs: {len(self.df):>5}")
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset="designed_sequence", keep="first")
        msg = f"Duplicates found: {before - len(self.df)}. Removing duplicates. {len(self.df)} designs remain.\n"
        print(msg)

    def filter_df(self):
        filter_cols = []
        self.df["num_filters_passed"] = 0

        for filter in self.filters:
            feat = filter["feature"]
            low = filter["lower_is_better"]
            threshold = filter["threshold"]

            filter_col = f"pass_{feat}_filter"
            filter_cols.append(filter_col)
            if low:
                self.df[filter_col] = self.df[feat] <= threshold
            else:
                self.df[filter_col] = self.df[feat] >= threshold

            self.df["num_filters_passed"] += self.df[filter_cols].all(axis=1)
            self.df["pass_filters"] = self.df[filter_cols].all(axis=1)

            msg = f"Num designs that pass the {feat} filter with threshold {threshold} where {'lower' if low else 'higher'} is better: {self.df[filter_col].sum()}"
            print(msg)
            print(f"Remaining designs: {self.df['pass_filters'].sum()}")

        num_pass = self.df["pass_filters"].sum()
        if num_pass < self.top_budget or num_pass < self.budget:
            print(
                f"Only {num_pass} designs pass filters. We highly recommend relaxing the thresholds."
            )
        print("\n")

    def absolute_metrics(self):
        norm_path = Path("src/boltzgen/resources/metrics_normalization.json")
        if not norm_path.exists():
            return

        with norm_path.open("r") as f:
            norm_stats = json.load(f)

        for col, stats in norm_stats.items():
            mean = stats["mean"]
            std = stats["std"]
            if col in self.df.columns:
                self.df[col + "_z"] = (self.df[col] - mean) / std

        importances = {
            "affinity_probability_binary1": 1.5,
            "design_iiptm": 1.0,
            "design_ptm": 0.5,
            "min_design_to_target_pae": -1.0,  # lower is better
            "design_hydrophobicity": -0.125,  # lower is better
            "design_largest_hydrophobic_patch_refolded": -0.15,  # lower is better
            "delta_sasa_refolded": 0.25,
            "plip_saltbridge_refolded": 0.25,
            "plip_hbonds_refolded": 0.25,
        }

        self.df["absolute_score"] = 0.0
        for base_col, weight in importances.items():
            if base_col in self.df.columns:
                norm_col = base_col + "_z"
                self.df["absolute_score"] += weight * self.df[norm_col]
        total_importance = sum(abs(w) for w in importances.values())
        self.df["absolute_score"] /= total_importance

        self.df["structure_confidence"] = 0.0
        weight_sum = 0
        for col in ["design_iiptm", "design_ptm", "min_design_to_target_pae"]:
            weight = importances[col]
            norm_col = col + "_z"
            weight_sum += abs(weight)
            self.df["structure_confidence"] += weight * self.df[norm_col]
        self.df["structure_confidence"] /= weight_sum

        for flt in self.filters:
            feat = flt["feature"]
            filter_col = f"pass_{feat}_filter"
            if "fraction" in feat:
                # If this is a "fraction" feature, meaning a res_type fraction filter, only apply the penalty if num_design > 8
                mask_fail = (self.df["num_design"] > 8) & (self.df[filter_col] == False)
            else:
                mask_fail = self.df[filter_col] == False

            self.df.loc[mask_fail, "absolute_score"] *= 0.1


    def sort_df(self):
        rank_df = pd.DataFrame(index=self.df.index)

        # 1. For each row, compute its rank according to each metric
        # Scale the ranks by importance (divide by inverse_importance)
        # Use the feature that we are ranking by AND the number of filters that a design passes. Thus the things that do not pass ass many filters (e.g. they dont pass all the filters) are at the end
        for col, inverse_importance in self.metrics.items():
            if self.metrics[col] is None:
                continue

            rank_df[f"rank_{col}"] = (
                self.df[["num_filters_passed", col]]
                .apply(tuple, axis=1)
                .rank(method="min", ascending=False)
                .astype(int)
                / inverse_importance
            )
        self.df = pd.concat([self.df, rank_df], axis=1)

        # 2. For each row, find the max (worst) rank across the metrics.
        # This single value determines its final rank group.
        self.df["max_rank"] = rank_df.max(axis=1)

        # 3. Sort by this new max_rank and then create the final dense rank,
        # which is equivalent to the original 'rank_counter'.
        self.df = self.df.sort_values("max_rank")
        self.df["secondary_rank"] = self.df["max_rank"].rank(method="dense").astype(int)

        # sort by ranking and resolve ties via design_iptm
        self.df = self.df.sort_values(
            by=[
                "secondary_rank",
                "design_to_target_iptm"
                if "design_to_target_iptm" in self.df
                else "design_iptm",
            ],
            ascending=[True, False],
        )

        self.df["final_rank"] = np.arange(1, len(self.df) + 1)
        self.df["quality_score"] = 1 - (self.df["final_rank"] - 1) / (len(self.df) - 1)

        # Reorder columns
        priority_col_candidates = [
            "id",
            "final_rank",
            "designed_sequence",
            "designed_chain_sequence",
            "num_design",
            "affinity_probability_binary1",
            "design_to_target_iptm"
            if "design_to_target_iptm" in self.df
            else "design_iptm",
            "min_design_to_target_pae"
            if "min_design_to_target_pae" in self.df
            else "min_interaction_pae",
            "design_ptm",
            "filter_rmsd",
            "designfolding-filter_rmsd"
            if "designfolding-filter_rmsd" in self.df
            else "filter_rmsd"
            "plip_saltbridge" + ("_refolded" if self.from_inverse_folded else ""),
            "plip_hbonds" + ("_refolded" if self.from_inverse_folded else ""),
            "delta_sasa_refolded",
            "design_largest_hydrophobic_patch_refolded",
            "design_chain_hydrophobicity",
            "design_hydrophobicity",
            "loop",
            "helix",
            "sheet",
        ]
        priority_cols = [c for c in priority_col_candidates if c in self.df.columns]

        other_cols = [col for col in self.df.columns if col not in priority_cols]
        new_column_order = priority_cols + other_cols
        self.df = self.df[new_column_order]

    def write_outdir(self):
        num_digits = len(str(len(self.df)))

        top_dir2 = self.top_dir / "before_refolding"
        top_dir2.mkdir(parents=True, exist_ok=True)
        for i, (idx, row) in tqdm(
            enumerate(self.df[: self.top_budget].iterrows()),
            desc="copy top design files",
        ):
            filename = row["file_name"]
            new_filename = f"rank{i:0{num_digits}d}_{filename}"
            src = self.design_dir / filename
            dst = top_dir2 / new_filename
            shutil.copy2(src, dst)

            src = self.design_dir / "refold_cif" / filename
            dst = self.top_dir / new_filename
            shutil.copy2(src, dst)

        # save to output/diverse_* directory
        self.div_dir.mkdir(parents=True, exist_ok=True)
        div_dir2 = self.div_dir / "before_refolding"
        div_dir2.mkdir(parents=True, exist_ok=True)
        for i in tqdm(self.diverse_selection, desc="copy diversity files"):
            src = self.design_dir / self.df_m.loc[i, "file_name"]
            qualityrank = self.df_m.loc[i, "final_rank"]
            filename = src.name
            new_filename = f"rank{qualityrank:0{num_digits}d}_{filename}"
            shutil.copy2(src, div_dir2 / new_filename)

            src = self.design_dir / "refold_cif" / self.df_m.loc[i, "file_name"]
            shutil.copy2(src, self.div_dir / new_filename)
        self.df_div.to_csv(
            self.outdir / f"final_designs_metrics_{self.budget}.csv", index=False
        )
        print("Files + CSV saved to", self.outdir)

        self.df.to_csv(self.outdir / f"all_designs_metrics.csv", index=False)

    def optimize_diversity(self):
        # Load structures and sequences to compute similarities
        seq_path = self.design_dir / "ca_coords_sequences.pkl.gz"
        if not seq_path.exists():
            raise FileNotFoundError(f"Expected {seq_path} to exist")
        df_seq = pd.read_pickle(seq_path)[["id", "sequence"]]

        self.df_m = pd.merge(self.df, df_seq, on="id", how="inner").reset_index(
            drop=True
        )
        seqs = self.df_m["sequence"].tolist()
        quality = self.df_m["quality_score"].to_numpy()

        # sequence-only similarity
        aligner = Align.PairwiseAligner()
        pid_cache = {}

        def sim_fn(i, j):
            if i == j:
                return 1.0
            key = tuple(sorted((i, j)))
            if key not in pid_cache:
                seq1 = seqs[i]
                seq2 = seqs[j]
                aln = aligner.align(seq1, seq2)[0]
                pid_cache[key] = aln.score / max(len(seqs[i]), len(seqs[j]))
            return pid_cache[key]

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        diverse_selection = self.select_lazy_greedy(self.budget, quality, sim_fn)

        self.diverse_selection = diverse_selection
        self.df_div = self.df_m.iloc[diverse_selection].reset_index(drop=True)

    def select_lazy_greedy(self, k, quality, sim_fn):
        # Handle edge case where we have fewer items than requested
        if len(quality) <= k:
            return list(range(len(quality)))

        selected = [int(np.argmax(quality))]
        remaining = set(range(len(quality))) - set(selected)

        heap = []
        for i in remaining:
            div = 1 - sim_fn(i, selected[0])
            gain = (1 - self.alpha) * quality[i] + self.alpha * div
            heapq.heappush(heap, (-gain, i))

        buckets = np.zeros(len(self.size_buckets) + 1)
        for _ in tqdm(
            range(k - 1), desc="Performing lazy greedy diversity optimization."
        ):
            while True:
                neg_gain, cand = heapq.heappop(heap)
                num_design = len(self.df_m["sequence"][cand])
                bucket_idx = None
                for idx, bucket_size in enumerate(self.size_buckets):
                    if (
                        num_design < bucket_size["max"]
                        and num_design >= bucket_size["min"]
                    ):
                        bucket_idx = idx

                if bucket_idx is None:
                    bucket_idx = len(self.size_buckets)

                bucket_full = (
                    not bucket_idx == len(self.size_buckets)
                    and buckets[bucket_idx]
                    == self.size_buckets[bucket_idx]["num_designs"]
                )
                if bucket_full:
                    continue

                true_div = 1 - max(sim_fn(cand, j) for j in selected)
                true_g = (1 - self.alpha) * quality[cand] + self.alpha * true_div
                heapq.heappush(heap, (-true_g, cand))

                if heap[0][1] == cand:
                    heapq.heappop(heap)
                    selected.append(cand)
                    remaining.remove(cand)
                    buckets[bucket_idx] += 1
                    break
        return sorted(selected)

    def prepare_visualization(self):
        summary_metrics = [
            "num_design",
            "filter_rmsd",
            "designfolding-filter_rmsd"
            if "designfolding-filter_rmsd" in self.df
            else "filter_rmsddesign_ptm",
            "design_iptm",
            "design_to_target_iptm"
            if "design_to_target_iptm" in self.df
            else "design_iptm",
            "min_design_to_target_pae"
            if "min_design_to_target_pae" in self.df
            else "min_interaction_pae",
            "delta_sasa_refolded"
            if self.from_inverse_folded
            else "delta_sasa_original",
            "plip_saltbridge" + ("_refolded" if self.from_inverse_folded else ""),
            "plip_hbonds" + ("_refolded" if self.from_inverse_folded else ""),
        ]

        # Scatter pairs (each will be one page)
        extra_pairs = [
            ("num_design", "rank"),
            (
                "num_design",
                "plip_saltbridge" + ("_refolded" if self.from_inverse_folded else ""),
            ),
            (
                "num_design",
                "plip_hbonds" + ("_refolded" if self.from_inverse_folded else ""),
            ),
            (
                "num_design",
                "delta_sasa" + ("_refolded" if self.from_inverse_folded else ""),
            ),
            (
                "num_design",
                "design_ptm",
            ),
            (
                "num_design",
                "min_design_to_target_pae"
                if "min_design_to_target_pae" in self.df
                else "min_interaction_pae",
            ),
            (
                "num_design",
                "design_to_target_iptm"
                if "design_to_target_iptm" in self.df
                else "design_iptm",
            ),
            (
                "num_design",
                "design_iptm",
            ),
            (
                "num_design",
                "design_iiptm" if "design_iiptm" in self.df else "design_iptm",
            ),
            (
                "num_design",
                "design_largest_hydrophobic_patch_refolded"
                if "design_largest_hydrophobic_patch_refolded" in self.df
                else "design_iptm",
            ),
            (
                "num_design",
                "design_hydrophobicity"
                if "design_hydrophobicity" in self.df
                else "design_iptm",
            ),
        ]
        if not self.from_inverse_folded:
            extra_pairs.append(("delta_sasa_refolded", "delta_sasa_original"))
            extra_pairs.append(("plip_saltbridge", "delta_sasa_original"))

        # Histograms with selected overlay
        hist_metrics = [
            "num_design",
            "filter_rmsd",
            "designfolding-filter_rmsd"
            if "designfolding-filter_rmsd" in self.df
            else "filter_rmsddesign_ptm",
            "design_to_target_iptm"
            if "design_to_target_iptm" in self.df
            else "design_iptm",
            "design_iptm",
            "min_design_to_target_pae"
            if "min_design_to_target_pae" in self.df
            else "min_interaction_pae",
            "plip_saltbridge" + ("_refolded" if self.from_inverse_folded else ""),
            "plip_hbonds" + ("_refolded" if self.from_inverse_folded else ""),
            "plip_hydrophobic" + ("_refolded" if self.from_inverse_folded else ""),
            "delta_sasa_refolded",
            "design_largest_hydrophobic_patch_refolded"
            if "design_largest_hydrophobic_patch_refolded" in self.df
            else "delta_sasa_refolded",
            "design_hydrophobicity"
            if "design_hydrophobicity" in self.df
            else "delta_sasa_refolded",
        ]
        if not self.from_inverse_folded:
            hist_metrics.append("delta_sasa_original")

        hist_metrics = list(dict.fromkeys(hist_metrics))
        extra_pairs = list(dict.fromkeys(extra_pairs))

        if self.use_affinity:
            summary_metrics.insert(2, "affinity_probability_binary1")
            hist_metrics.insert(2, "affinity_probability_binary1")

        avail = [m for m in summary_metrics if m in self.df.columns]
        base_rows = [
            ["Num designs", len(self.df), "-"],
        ]

        extra_mean = (
            [
                m,
                f"{self.df[m].mean():.3f}",  # mean of ALL
                f"{self.df[: self.top_budget][m].mean():.3f}",  # mean of red set
                f"{self.df_div[m].mean():.3f}",  # mean of BLUE set
            ]
            for m in avail
        )

        for row in base_rows:
            row.append("-")

        rows = base_rows + list(extra_mean)

        row_headers = [
            "Metric",
            f"Mean",
            f"Mean top {self.top_budget}",
            f"Mean top {self.budget} diverse",
        ]

        metric_rows = [[k, v] for k, v in self.metrics.items()]

        n_total = len(self.df)
        n_pass_filters = (
            int(self.df["pass_filters"].sum()) if "pass_filters" in self.df else n_total
        )

        text = f"""
        • Designs generated: {n_total}
        • Designs passing all filters: {n_pass_filters}

        • MMCIF files of final designs are in:  
          {self.div_dir}  

        • Metrics and sequences of {self.budget} final designs are in:  
          {self.outdir}/final_designs_metrics_{self.budget}.csv
          
        • Metrics of all designs are in:  
          {self.outdir}/all_designs_metrics.csv

        You can rerun filtering (very quick), using this command with changed parameters:
            -- boltzgen run input_spec.yaml --steps filtering --config filtering budget=60 alpha=0.05
        You can also rerun filtering in a jupyter notebook if you want using `filter.ipynb`

         What was run to produce this in the Filter task:
         1. Filtering: each design is evaluated against mandatory thresholds. 
         2. Ranking: for every metric we compute its rank, then scale it by the metric’s inverse-importance weight. Designs with fewer passed filters are automatically penalised because the ranking key is the pair (num_filters_passed, metric). The overall quality score is the worst (maximum) of these scaled ranks. The {self.top_budget} best designs form the Top set.
         3. Diversity: a lazy-greedy algorithm selects {self.budget} designs that jointly maximise quality and
            minimise sequence similarity (sequence-identity distance). The trade-off is controlled by α = {self.alpha}:
              • α = 0   → 100 % quality focus (same as Top set)
              • α = 1   → 100 % diversity focus (ignores quality)
              • Quality – composite of metrics such as iPTM, salt-bridges, ΔSASA, etc. Each metric has an "inverse importance" weight (see "Sorting Criteria" table). A larger weight divides the rank by a bigger number and therefore down-weights that metric.
              • Diversity – 1 − sequence identity between designs.
            We use α = {self.alpha}, meaning {round((1 - self.alpha) * 100)} % emphasis on quality, {round(self.alpha * 100)} % on diversity.
 
          """

        csv_expl_rows = [
            ["id", "filename to retrieve the file"],
            ["design_sequence", "designed amino acids (may be subset of chain)"],
            [
                "designed_chain_sequence",
                "full sequence of the chain containing designed residues (recommended for synthesis)",
            ],
            ["num_design", "number of designed residues"],
            ["secondary_rank", "intermediate rank from the sorting procedure"],
            [
                "design_ptm",
                "predicted TM score for intra-design contacts (higher = better)",
            ],
            [
                "design_iptm",
                "predicted TM score for design–target contacts (higher = better)",
            ],
            [
                "design_to_target_iptm",
                "same as design_iptm but for multi-chain designs",
            ],
            [
                "min_design_to_target_pae",
                "minimum PAE between design & target (lower = better)",
            ],
            ["plip_saltbridge", "number of salt-bridge interactions"],
            ["plip_hbonds", "number of hydrogen-bond interactions"],
            ["plip_hydrophobic", "number of hydrophobic interactions"],
            ["delta_sasa_original", "ΔSASA when binder present vs absent"],
            ["delta_sasa_refolded", "same as above but on the refolded structure"],
        ]

        intro_text = text

        return (
            hist_metrics,
            extra_pairs,
            row_headers,
            rows,
            metric_rows,
            intro_text,
            csv_expl_rows,
        )

    def make_visualization(
        self,
        hist_metrics,
        extra_pairs,
        row_headers,
        rows,
        metric_rows,
        intro_text,
        csv_expl_rows,
        jupyter_nb=False,
    ):
        pdf_path = self.outdir / f"results_overview.pdf"
        pdf = PdfPages(pdf_path)

        def _ensure_width(fig, target_w=8.5):
            w, h = fig.get_size_inches()
            if abs(w - target_w) > 0.01:
                scale = target_w / w
                fig.set_size_inches(target_w, h * scale, forward=True)

        def show(fig):
            _ensure_width(fig)
            plt.tight_layout()
            pdf.savefig(fig)
            if jupyter_nb:
                plt.show(fig)
            plt.close(fig)

        def _wrap(txt: str, width: int = 60):
            return "\n".join(fill(line, width) for line in txt.splitlines())

        def _format_body(txt: str, width: int | None = 95):
            txt = dedent(txt).strip("\n")
            if width is None:
                return txt
            out_lines = []
            for line in txt.splitlines():
                if not line.strip():
                    out_lines.append("")
                    continue
                m = re.match(r"^(\s*)([•\d]+[.)]?)(\s+)(.*)", line)
                if m:
                    lead_ws, bullet, spacer, rest = m.groups()
                    bullet_str = f"{lead_ws}{bullet}{spacer}"
                    wrapped = fill(
                        rest,
                        width=max(20, width - len(bullet_str)),
                        subsequent_indent=" " * len(bullet_str),
                    )
                    wrapped_lines = wrapped.split("\n")
                    wrapped_lines[0] = bullet_str + wrapped_lines[0]
                    out_lines.extend(wrapped_lines)
                else:
                    out_lines.extend(fill(line, width).split("\n"))
            return "\n".join(out_lines)

        def section_page(title: str, body: str = ""):
            is_overview = title.startswith("Results Overview")
            sec_height = 11 if is_overview else 4.5
            fig, ax = plt.subplots(figsize=(8.5, sec_height))
            ax.axis("off")
            main, sub = (
                (title.split("–", 1) + [None])[:2] if "–" in title else (title, None)
            )
            base_fs = 20
            fs_main = max(14, base_fs - int(len(main) / 40) * 2)
            fs_sub = fs_main - 2
            ax.text(
                0.6 if is_overview else 0.5,
                0.9 if is_overview else 0.75,
                _wrap(main.strip(), 40),
                fontsize=fs_main,
                ha="center",
                va="top",
                weight="bold",
            )
            if sub:
                ax.text(
                    0.5,
                    0.66,
                    _wrap(sub.strip(), 50),
                    fontsize=fs_sub,
                    ha="center",
                    va="top",
                    style="italic",
                )

            def _render_line(text, y, is_path=False):
                ax.text(
                    0.12,
                    y,
                    text,
                    fontsize=10,
                    ha="left",
                    va="top",
                    wrap=False,
                    fontfamily="monospace" if is_path else None,
                    color="blue" if is_path else "black",
                )

            if body:
                if is_overview:
                    y_pos = 0.78
                    for para in re.split(r"\n\s*\n", body.strip()):
                        lines = para.strip().split("\n")
                        header = lines[0].strip()
                        rest = (
                            "\n".join(l.strip() for l in lines[1:])
                            if len(lines) > 1
                            else ""
                        )
                        ax.text(
                            0.12,
                            y_pos,
                            header,
                            fontsize=12,
                            fontweight="bold",
                            ha="left",
                            va="top",
                            wrap=True,
                        )
                        y_pos -= 0.05

                        if rest:
                            for line in _format_body(rest, width=100).split("\n"):
                                _render_line(
                                    line, y_pos, is_path=line.lstrip().startswith("--")
                                )
                                y_pos -= 0.03
                            y_pos -= 0.05
                else:
                    ax.text(
                        0.12,
                        0.46,
                        _format_body(body, width=90),
                        fontsize=10,
                        ha="left",
                        va="top",
                        wrap=True,
                    )
            show(fig)

        section_page("Results Overview", body=_format_body(intro_text, width=110))

        csv_expl_rows_wrapped = [[col, _wrap(desc, 100)] for col, desc in csv_expl_rows]
        section_page("CSV Column Reference")
        fig_height = 0.35 * len(csv_expl_rows_wrapped) + 1.5
        fig, ax = plt.subplots(figsize=(8.5, fig_height))
        ax.axis("off")
        table = ax.table(
            cellText=csv_expl_rows_wrapped,
            colLabels=["Column", "Description"],
            colLoc="left",
            colWidths=[0.25, 0.70],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.2)
        show(fig)

        section_page(
            "Filtering & Sorting Criteria",
            body=(
                "This section first lists threshold filters that every design must pass (Filtering Criteria table)"
                "then explains how designs are ranked (Sorting Criteria table) weighted by their inverse importance."
            ),
        )

        filters_df = pd.DataFrame(self.filters)
        filters_df["Pass"] = 0
        for i, filter in enumerate(self.filters):
            filters_df.at[i, "Pass"] = self.df[f"pass_{filter['feature']}_filter"].sum()

        fig_height = 0.4 * len(filters_df) + 2
        fig, ax = plt.subplots(figsize=(8.5, fig_height))
        ax.axis("off")
        ax.text(
            0.5,
            1.0,
            "Filtering Criteria",
            fontsize=14,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )
        table = ax.table(
            cellText=filters_df.values,
            colLabels=filters_df.columns.tolist(),
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        show(fig)
        fig_height = 0.4 * len(metric_rows) + 1
        fig, ax = plt.subplots(figsize=(8.5, fig_height))
        ax.axis("off")
        ax.text(
            0.5,
            1.0,
            "Sorting Criteria",
            fontsize=14,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )
        table = ax.table(
            cellText=metric_rows,
            colLabels=["Metric", "Inverse Importance"],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        show(fig)

        section_page(
            "Results Summary – Aggregate statistics",
            body=(
                f"Quick numeric overview comparing (i) ALL incoming designs, (ii) the top-{self.top_budget} highest-quality "
                f"designs, and (iii) the {self.budget} quality+diversity designs produced by the lazy-greedy "
                "selection."
            ),
        )
        if rows:
            fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(rows) + 1))
            ax.axis("off")
            table = ax.table(
                cellText=rows,
                colLabels=row_headers,
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            show(fig)

        if self.plot_seq_logos:
            section_page(
                "Sequence Logos & AA Composition – Motifs & biases",
                body=(
                    "Sequence logos display the per-position amino-acid preferences after multiple-sequence alignment, "
                    "highlighting motifs that emerge in the design sets.  The accompanying pies summarise overall "
                    "hydrophobicity and charge composition comparing ALL, Top-quality (red), and Diversity-optimised subsets. "
                ),
            )
        vis = (
            "designed_sequence"
            if (
                self.df["designed_chain_sequence"].str.len().mean()
                > 1.5 * self.df["designed_sequence"].str.len().mean()
            )
            else "designed_chain_sequence"
        )
        seq_sets = [
            (
                f"All {len(self.df)} {vis}",
                self.df[vis].tolist(),
            ),
            (
                f"Top {self.top_budget} {vis}",
                self.df[vis].tolist()[: self.top_budget],
            ),
            (
                f"Diverse {self.budget} {vis}",
                self.df_div[vis].tolist(),
            ),
        ]
        if self.plot_seq_logos:
            for name, sequences in seq_sets:
                show(create_alignment_logo(sequences, name))
            for name, sequences in seq_sets:
                show(aa_composition_pie(sequences, name))

            if self.modality == "antibody":
                for name, sequences in seq_sets:
                    show(cdr_logo(sequences, name))

        section_page(
            "Scatter Plots – Metric relationships",
            body="""
                Each scatter page contains two panels:
                • Left – all designs (grey) with overlays of Top (red) and Diverse (blue).
                • Right – same but limited to designs passing the RMSD filter.
                """,
        )
        for x, y in extra_pairs:
            if x in self.df.columns and y in self.df.columns:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.5))
                self._scatter_plus(ax1, self.df, x, y, "All samples")
                if self.df["pass_filter_rmsd_filter"].sum() > 0:
                    self._scatter_plus(
                        ax2,
                        self.df[self.df["pass_filter_rmsd_filter"]],
                        x,
                        y,
                        "(Designs passing RMSD threshold)",
                    )
                show(fig)

        section_page(
            "Metric Distributions – Histograms",
            body=(
                "Distribution of each metric across all designs (grey) with overlays for Top-quality (red outline) "
                "and Diversity-optimised (blue dashed) subsets. The right panel repeats the histogram but only for designs "
                "that pass the RMSD threshold."
            ),
        )
        for m in hist_metrics:
            if m not in self.df.columns:
                continue
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.5))
            self._hist_plus(
                ax1, self.df[m], self.df[: self.top_budget][m], self.df_div[m], m, ""
            )
            if self.df["pass_filter_rmsd_filter"].sum() > 0:
                self._hist_plus(
                    ax2,
                    self.df[self.df["pass_filter_rmsd_filter"]][m],
                    self.df[: self.top_budget][m],
                    self.df_div[m],
                    m,
                    " (Designs passing RMSD threshold)",
                )
            show(fig)

        if self.num_liability_plots:
            section_page(
                "Liability Analysis – Developability flags",
                body=(
                    "For the first few top designs we compute biochemical liability scores (deamidation, oxidation, etc.) "
                    "and visualise them along the sequence with a heat-map. Dark red indicates positions needing "
                    "attention during optimisation."
                ),
            )
        # Plot Liability Heatmaps for Top-budget subset
        for idx, row in tqdm(
            enumerate(self.df[: self.num_liability_plots].itertuples(index=False)),
            desc=f"Making liability plots for top {self.num_liability_plots} sequences.",
        ):
            seq = row.designed_sequence
            try:
                res = compute_liability_scores(
                    [seq], modality=self.modality, peptide_type=self.peptide_type
                )
                liab = res.get(seq, {"score": None, "violations": []})
                fig = plot_seq_liabilities(
                    seq,
                    f"Qualityrank {idx} {row.id}",
                    liab["violations"],
                    total_score=liab["score"],
                )
                show(fig)
            except Exception as e:
                print(f"  Error processing  {seq[:20]}: {e}")
                plt.close("all")
                continue

        pdf.close()

        print(
            "A description of metrics and summarizing plots was written to:", pdf_path
        )

    def _scatter_plus(self, ax, df, x, y, title=""):
        ax.scatter(
            df[x],
            df[y],
            color="lightgray",
            alpha=0.4,
            s=14,
            zorder=1,
        )

        ax.scatter(
            self.df[: self.top_budget][x],
            self.df[: self.top_budget][y],
            facecolors="none",
            edgecolors="red",
            linewidth=1.5,
            s=30,
            zorder=2,
            alpha=0.5,
            label="top-quality",
        )

        if not self.df_div.empty:
            ax.scatter(
                self.df_div[x],
                self.df_div[y],
                facecolors="none",
                edgecolors="blue",
                linewidth=1.5,
                s=50,  #  ← slightly larger
                zorder=3,
                alpha=0.5,
                label="quality+diversity",
            )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.legend()

    def _hist_plus(self, ax, data_all, data_red, data_blue, metric, suffix):
        # Check if data contains valid values for histogram
        if data_all is None or len(data_all) == 0 or data_all.isna().all():
            ax.text(
                0.5,
                0.5,
                f"No valid data for {metric}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel(metric)
            ax.set_ylabel("count")
            ax.set_title(f"{metric}{suffix} (No data)")
            return

        # Filter out NaN values for histogram plotting
        valid_data_all = data_all.dropna()
        valid_data_red = (
            data_red.dropna() if data_red is not None else pd.Series(dtype=float)
        )
        valid_data_blue = (
            data_blue.dropna() if data_blue is not None else pd.Series(dtype=float)
        )

        if len(valid_data_all) == 0:
            ax.text(
                0.5,
                0.5,
                f"No valid data for {metric}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel(metric)
            ax.set_ylabel("count")
            ax.set_title(f"{metric}{suffix} (No data)")
            return

        # Create histogram with valid data
        ax.hist(valid_data_all, bins=30, color="lightgray", alpha=0.6, label="all")

        if len(valid_data_red) > 0:
            ax.hist(
                valid_data_red,
                bins=30,
                histtype="step",
                linewidth=1.7,
                color="red",
                label="top-quality",
            )
        if len(data_blue) and len(valid_data_blue) > 0:
            ax.hist(
                valid_data_blue,
                bins=30,
                histtype="step",
                linewidth=1.7,
                color="blue",
                linestyle="--",
                label="quality+diversity",
            )

        ax.set_xlabel(metric)
        ax.set_ylabel("count")
        ax.set_title(f"{metric}{suffix}")
        ax.legend()
