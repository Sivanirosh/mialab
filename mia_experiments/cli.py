#!/usr/bin/env python3
"""Command-line interface for the MIA experiment workflow."""

import argparse
import os
import sys
import textwrap

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mia_experiments import ExperimentManager


def _validate_paths(*paths: str) -> bool:
    """Ensure all provided directories exist."""

    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        print("✗ Missing required paths:")
        for path in missing:
            print(f"  - {path}")
        return False
    return True


def run_command(args: argparse.Namespace) -> int:
    """Execute the all-preprocessing experiment once."""

    if not _validate_paths(args.data_atlas, args.data_train, args.data_test):
        return 1

    manager = ExperimentManager(args.output_dir)
    try:
        result = manager.run_all_preprocessing_experiment(
            data_atlas_dir=args.data_atlas,
            data_train_dir=args.data_train,
            data_test_dir=args.data_test,
            include_postprocessing=not args.disable_postprocessing,
            experiment_name=args.experiment_name,
        )
    except Exception as exc:  # pragma: no cover - CLI level reporting
        print(f"\n✗ Experiment failed: {exc}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    print("\n✓ Experiment completed successfully.\n")
    for key in ["experiment_id", "pipeline_run_dir", "metrics_pre_csv", "metrics_post_csv"]:
        if result.get(key):
            print(f"  {key.replace('_', ' '):20s}: {result[key]}")

    return 0


def list_command(args: argparse.Namespace) -> int:
    """Render the experiment log."""

    manager = ExperimentManager(args.output_dir)
    df = manager.list_experiments()
    if df.empty:
        print("No experiments recorded yet.")
        return 0

    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.to_string(index=False))
    return 0


def metrics_command(args: argparse.Namespace) -> int:
    """Show a brief summary of the pre/post metrics for an experiment."""

    manager = ExperimentManager(args.output_dir)
    try:
        metrics = manager.load_metrics(args.experiment_id)
    except ValueError as exc:
        print(str(exc))
        return 1

    for view, df in metrics.items():
        print(f"\n=== {view.upper()} METRICS ({len(df)} rows) ===")
        preview = df.head(args.rows)
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(preview.to_string(index=False))
        if len(df) > args.rows:
            print(f"... ({len(df) - args.rows} more rows omitted)")
    return 0


def main() -> int:
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Run and inspect the single-pass MIA segmentation experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
Example usage:
  # Run the experiment once and store artefacts under ./experiments
  python -m mia_experiments.cli run \\
      --data-atlas ./data/atlas \\
      --data-train ./data/train \\
      --data-test ./data/test

  # List recorded runs
  python -m mia_experiments.cli list

  # Preview metrics for the most recent run
  python -m mia_experiments.cli metrics --experiment-id all_preprocessing_20250101_120000
"""
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Execute the pipeline once with all preprocessing enabled.")
    run_parser.add_argument("--data-atlas", required=True, help="Path to atlas data directory.")
    run_parser.add_argument("--data-train", required=True, help="Path to training data directory.")
    run_parser.add_argument("--data-test", required=True, help="Path to test data directory.")
    run_parser.add_argument("--output-dir", default="./experiments", help="Directory to store experiment artefacts.")
    run_parser.add_argument(
        "--experiment-name",
        default="all_preprocessing",
        help="Custom experiment name prefix (default: all_preprocessing).",
    )
    run_parser.add_argument(
        "--disable-postprocessing",
        action="store_true",
        help="Skip post-processing (only preprocessed metrics will be generated).",
    )
    run_parser.add_argument("--verbose", action="store_true", help="Print full tracebacks on failure.")

    list_parser = subparsers.add_parser("list", help="Show the experiment log.")
    list_parser.add_argument("--output-dir", default="./experiments", help="Directory containing experiment log.")

    metrics_parser = subparsers.add_parser("metrics", help="Preview pre/post metrics for a recorded experiment.")
    metrics_parser.add_argument("experiment_id", help="Experiment identifier as shown in the log.")
    metrics_parser.add_argument("--output-dir", default="./experiments", help="Directory containing experiment log.")
    metrics_parser.add_argument("--rows", type=int, default=5, help="Number of rows to display per view (default: 5).")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    if args.command == "run":
        return run_command(args)
    if args.command == "list":
        return list_command(args)
    if args.command == "metrics":
        return metrics_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())