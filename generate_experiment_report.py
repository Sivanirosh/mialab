#!/usr/bin/env python3
"""
Generate visual analysis artifacts for a single segmentation experiment run.

Inputs:
    --experiment-dir path/to/experiment/run
    [--output-dir path/to/report/output]
    [--max-screenshots N]  (default: 5)
    [--labels 1 2 3 4 5]   (default: all labels)

Outputs:
    - Aggregated CSV summaries (per-label performance, improvements).
    - Plots comparing pre/post metrics.
    - 3D screenshots (PNG) of selected subjects' segmentations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mia_experiments.core.segmentation_visualizer import SegmentationVisualizer


METRICS_PRE = "metrics_preprocessed.csv"
METRICS_POST = "metrics_postprocessed.csv"


def load_metrics(experiment_dir: Path) -> pd.DataFrame:
    """Load the pre/post metrics CSV files and return a combined dataframe."""

    pre_path = experiment_dir / METRICS_PRE
    post_path = experiment_dir / METRICS_POST

    if not pre_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {pre_path}")
    if not post_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {post_path}")

    pre_df = pd.read_csv(pre_path)
    pre_df["stage"] = "pre"
    post_df = pd.read_csv(post_path)
    post_df["stage"] = "post"

    combined = pd.concat([pre_df, post_df], ignore_index=True)
    return combined


def plot_metric_distributions(metrics: pd.DataFrame, output_dir: Path, value_col: str = "DICE") -> None:
    """Create comparative plots for the given metric (Dice or Hausdorff)."""

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=metrics, x="LABEL", y=value_col, hue="stage", palette="Set2")
    plt.title(f"{value_col} distribution per label")
    plt.tight_layout()
    plt.savefig(output_dir / f"{value_col.lower()}_boxplot.png", dpi=300)
    plt.close()


def compute_improvements(metrics: pd.DataFrame, value_col: str = "DICE") -> pd.DataFrame:
    """Compute mean improvements per label between post and pre metrics."""

    pivot = (
        metrics.pivot_table(
            index=["SUBJECT", "LABEL"],
            columns="stage",
            values=value_col,
            aggfunc="mean",
        )
        .reset_index()
        .dropna()
    )

    if "post" not in pivot or "pre" not in pivot:
        return pd.DataFrame()

    pivot["improvement"] = pivot["post"] - pivot["pre"]
    return pivot


def plot_improvements(improvements: pd.DataFrame, output_dir: Path, value_col: str = "DICE") -> None:
    """Plot average improvements per label."""

    if improvements.empty:
        return

    label_stats = (
        improvements.groupby("LABEL")["improvement"]
        .agg(["mean", "median", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    label_stats.to_csv(output_dir / f"{value_col.lower()}_improvement_summary.csv", index=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=label_stats, x="LABEL", y="mean", palette="viridis")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"Mean {value_col} improvement (post - pre)")
    plt.ylabel("Mean improvement")
    plt.tight_layout()
    plt.savefig(output_dir / f"{value_col.lower()}_improvement_bar.png", dpi=300)
    plt.close()


def find_segmentation_pairs(experiment_dir: Path) -> List[tuple[str, Path, Optional[Path]]]:
    """Return a list of (subject_id, pre_path, post_path) tuples."""

    pairs = []
    for seg_path in sorted(experiment_dir.glob("*_SEG.mha")):
        subject = seg_path.name.replace("_SEG.mha", "")
        seg_pp_path = experiment_dir / f"{subject}_SEG-PP.mha"
        pairs.append((subject, seg_path, seg_pp_path if seg_pp_path.exists() else None))
    return pairs


def capture_screenshots(
    pairs: Iterable[tuple[str, Path, Optional[Path]]],
    output_dir: Path,
    labels: Optional[List[int]],
    max_subjects: int,
    per_label: bool = True,
) -> None:
    """Render PNG screenshots of the provided segmentation pairs."""

    for idx, (subject, pre_path, post_path) in enumerate(pairs):
        if idx >= max_subjects:
            break

        if post_path is None:
            print(f"[screenshot] Skipping subject {subject} (missing post-processed file)")
            continue

        print(f"[screenshot] Rendering subject {subject} comparison")
        base_visualizer = SegmentationVisualizer()
        prediction_array = base_visualizer.load_segmentation(str(pre_path))
        post_visualizer = SegmentationVisualizer()
        prediction_pp_array = post_visualizer.load_segmentation(str(post_path))

        # Render combined comparison
        comp_visualizer = SegmentationVisualizer(spacing=base_visualizer.spacing)
        comp_visualizer.visualize_comparison(
            prediction_array,
            prediction_pp_array=prediction_pp_array,
            labels_to_show=labels,
            title=f"{subject} - Pre vs Post",
            save_screenshot=str(output_dir / f"{subject}_comparison.png"),
            start_interactive=False,
        )

        if per_label:
            label_ids = labels if labels else sorted({1, 2, 3, 4, 5})
            for label_id in label_ids:
                label_visualizer = SegmentationVisualizer(spacing=base_visualizer.spacing)
                label_visualizer.visualize_comparison(
                    prediction_array,
                    prediction_pp_array=prediction_pp_array,
                    labels_to_show=[label_id],
                    title=f"{subject} - Label {label_id}",
                    save_screenshot=str(output_dir / f"{subject}_label{label_id}_comparison.png"),
                    start_interactive=False,
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate plots and 3D screenshots from a segmentation experiment run.",
    )
    parser.add_argument("--experiment-dir", required=True, help="Path to the experiment run directory.")
    parser.add_argument(
        "--output-dir",
        help="Directory to store the generated report (default: <experiment_dir>/report).",
    )
    parser.add_argument(
        "--max-screenshots",
        type=int,
        default=1,
        help="Maximum number of subjects to capture screenshots for (default: 5).",
    )
    parser.add_argument(
        "--per-label-screenshots",
        action="store_true",
        help="Additionally render per-label comparison screenshots for each subject.",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        help="Subset of labels to include in visualisations (default: all labels present).",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else experiment_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(experiment_dir)
    metrics.to_csv(output_dir / "metrics_combined.csv", index=False)

    for value_col in ("DICE", "HDRFDST"):
        if value_col in metrics.columns:
            plot_metric_distributions(metrics, output_dir, value_col=value_col)
            improvements = compute_improvements(metrics.dropna(subset=[value_col]), value_col=value_col)
            improvements.to_csv(output_dir / f"{value_col.lower()}_improvements.csv", index=False)
            plot_improvements(improvements, output_dir, value_col)

    pairs = find_segmentation_pairs(experiment_dir)
    if pairs:
        capture_screenshots(
            pairs,
            output_dir,
            labels=args.labels,
            max_subjects=args.max_screenshots,
            per_label=args.per_label_screenshots,
        )
    else:
        print("No segmentation .mha files found; skipping screenshot capture.")

    print(f"\nReport artifacts saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

