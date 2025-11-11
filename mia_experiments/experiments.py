"""Experiment orchestration utilities for the simplified single-run workflow."""

from __future__ import annotations

import datetime
import json
import os
from typing import Dict, Optional, Tuple

import pandas as pd

from .core.config import (
    ExperimentConfig,
    RandomForestConfig,
    create_all_preprocessing_config,
)


class ExperimentManager:
    """Execute and catalogue single-run experiments."""

    LOG_COLUMNS = [
        "experiment_id",
        "name",
        "description",
        "timestamp",
        "status",
        "duration_seconds",
        "config_file",
        "experiment_root",
        "pipeline_run_dir",
        "metrics_pre_csv",
        "metrics_post_csv",
    ]

    def __init__(self, base_experiment_dir: str = "./experiments") -> None:
        self.base_experiment_dir = base_experiment_dir
        os.makedirs(self.base_experiment_dir, exist_ok=True)
        self.log_file = os.path.join(self.base_experiment_dir, "experiment_log.csv")
        self._ensure_log_schema()

    def _ensure_log_schema(self) -> None:
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=self.LOG_COLUMNS).to_csv(self.log_file, index=False)

    # --------------------------------------------------------------------- API
    def run_all_preprocessing_experiment(
        self,
        data_atlas_dir: str,
        data_train_dir: str,
        data_test_dir: str,
        include_postprocessing: bool = True,
        forest_config: Optional[RandomForestConfig] = None,
        experiment_name: str = "all_preprocessing",
        description: Optional[str] = None,
    ) -> Dict[str, str]:
        """Convenience wrapper that runs the canonical configuration."""

        config = create_all_preprocessing_config(
            name=experiment_name,
            include_postprocessing=include_postprocessing,
            forest_config=forest_config,
            description=description,
        )

        return self.run_experiment(config, data_atlas_dir, data_train_dir, data_test_dir)

    def run_experiment(
        self,
        config: ExperimentConfig,
        data_atlas_dir: str,
        data_train_dir: str,
        data_test_dir: str,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Run the pipeline once and split metrics into pre/post CSV files."""

        if experiment_id is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{config.name}_{timestamp}"

        experiment_root = os.path.join(self.base_experiment_dir, experiment_id)
        os.makedirs(experiment_root, exist_ok=True)

        config_file = os.path.join(experiment_root, "config.json")
        config.save(config_file)

        print(f"Running experiment: {experiment_id}")
        print(f"Configuration stored at: {config_file}")

        start_time = datetime.datetime.now()
        self._upsert_log(
            experiment_id,
            config,
            start_time,
            status="running",
            duration_seconds=0.0,
            config_file=config_file,
            experiment_root=experiment_root,
        )

        try:
            run_dir = self._execute_pipeline(config, experiment_root, data_atlas_dir, data_train_dir, data_test_dir)
            metrics_pre, metrics_post = self._split_results_csv(run_dir)

            duration = (datetime.datetime.now() - start_time).total_seconds()

            self._upsert_log(
                experiment_id,
                config,
                start_time,
                status="completed",
                duration_seconds=duration,
                config_file=config_file,
                experiment_root=experiment_root,
                pipeline_run_dir=run_dir,
                metrics_pre_csv=metrics_pre,
                metrics_post_csv=metrics_post,
            )

            print(f"Experiment {experiment_id} completed in {duration/60:.1f} minutes")
            return {
                "experiment_id": experiment_id,
                "experiment_root": experiment_root,
                "pipeline_run_dir": run_dir,
                "config_file": config_file,
                "metrics_pre_csv": metrics_pre,
                "metrics_post_csv": metrics_post,
            }

        except Exception as exc:  # pragma: no cover - propagated upwards but logged
            duration = (datetime.datetime.now() - start_time).total_seconds()
            self._upsert_log(
                experiment_id,
                config,
                start_time,
                status=f"failed: {exc}",
                duration_seconds=duration,
                config_file=config_file,
                experiment_root=experiment_root,
            )
            raise

    # -------------------------------------------------------------- Internals
    def _execute_pipeline(
        self,
        config: ExperimentConfig,
        experiment_root: str,
        data_atlas_dir: str,
        data_train_dir: str,
        data_test_dir: str,
    ) -> str:
        """Run `pipeline.main` and return the path of the generated run directory."""

        try:
            from pipeline import main as run_pipeline  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - import side-effect
            raise ImportError("Could not import pipeline module. Make sure pipeline.py is available.") from exc

        existing_children = set(os.listdir(experiment_root))
        run_pipeline(
            experiment_root,
            data_atlas_dir,
            data_train_dir,
            data_test_dir,
            config.to_pipeline_dict(),
        )

        new_children = [child for child in os.listdir(experiment_root) if child not in existing_children]
        if not new_children:
            raise RuntimeError("Pipeline completed but did not create a timestamped results directory.")

        latest_child = sorted(new_children)[-1]
        run_dir = os.path.join(experiment_root, latest_child)
        if not os.path.isdir(run_dir):
            raise RuntimeError(f"Expected results directory at {run_dir}, but it is missing.")

        return run_dir

    def _split_results_csv(self, run_dir: str) -> Tuple[str, str]:
        """Split pipeline metrics into pre- and post-processed CSV files."""

        results_csv = os.path.join(run_dir, "results.csv")
        if not os.path.exists(results_csv):
            raise FileNotFoundError(f"results.csv not found in pipeline run directory: {run_dir}")

        df = pd.read_csv(results_csv)
        if "SUBJECT" not in df.columns:
            raise ValueError("results.csv does not carry a SUBJECT column. Cannot split metrics.")

        df["SUBJECT"] = df["SUBJECT"].astype(str)
        post_mask = df["SUBJECT"].str.endswith("-PP")

        pre_df = df.loc[~post_mask].copy()
        post_df = df.loc[post_mask].copy()
        post_df["SUBJECT"] = post_df["SUBJECT"].str.replace("-PP$", "", regex=True)

        pre_path = os.path.join(run_dir, "metrics_preprocessed.csv")
        post_path = os.path.join(run_dir, "metrics_postprocessed.csv")

        pre_df.to_csv(pre_path, index=False)
        post_df.to_csv(post_path, index=False)

        # Optionally provide per-view summaries for downstream scripts.
        self._write_summary(pre_df, os.path.join(run_dir, "metrics_preprocessed_summary.csv"))
        self._write_summary(post_df, os.path.join(run_dir, "metrics_postprocessed_summary.csv"))

        return pre_path, post_path

    @staticmethod
    def _write_summary(df: pd.DataFrame, target_path: str) -> None:
        """Persist mean/std summaries if Dice metrics are available."""

        if df.empty or "DICE" not in df.columns:
            return

        summary_rows = []
        for label, group in df.groupby("LABEL"):
            if group["DICE"].empty:
                continue
            summary_rows.append(
                {
                    "LABEL": label,
                    "DICE_MEAN": group["DICE"].mean(),
                    "DICE_STD": group["DICE"].std(),
                    "HDRFDST_MEAN": group["HDRFDST"].mean() if "HDRFDST" in group else None,
                    "HDRFDST_STD": group["HDRFDST"].std() if "HDRFDST" in group else None,
                    "N": len(group),
                }
            )

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(target_path, index=False)

    def _upsert_log(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        start_time: datetime.datetime,
        status: str,
        duration_seconds: float,
        config_file: str,
        experiment_root: str,
        pipeline_run_dir: Optional[str] = None,
        metrics_pre_csv: Optional[str] = None,
        metrics_post_csv: Optional[str] = None,
    ) -> None:
        """Update the experiment log with the latest run information."""

        log_df = pd.read_csv(self.log_file)
        record = {
            "experiment_id": experiment_id,
            "name": config.name,
            "description": config.description,
            "timestamp": start_time.isoformat(),
            "status": status,
            "duration_seconds": float(duration_seconds),
            "config_file": config_file,
            "experiment_root": experiment_root,
            "pipeline_run_dir": pipeline_run_dir or "",
            "metrics_pre_csv": metrics_pre_csv or "",
            "metrics_post_csv": metrics_post_csv or "",
        }

        mask = log_df["experiment_id"] == experiment_id
        if mask.any():
            log_df.loc[mask, self.LOG_COLUMNS] = [record[column] for column in self.LOG_COLUMNS]
        else:
            log_df = pd.concat([log_df, pd.DataFrame([record])], ignore_index=True)

        log_df.to_csv(self.log_file, index=False)

    # ----------------------------------------------------------------- Helpers
    def list_experiments(self) -> pd.DataFrame:
        """Return the experiment log dataframe."""

        if not os.path.exists(self.log_file):
            return pd.DataFrame(columns=self.LOG_COLUMNS)
        return pd.read_csv(self.log_file)

    def load_metrics(self, experiment_id: str) -> Dict[str, pd.DataFrame]:
        """Load pre/post metrics for a recorded experiment."""

        log_df = self.list_experiments()
        entries = log_df[log_df["experiment_id"] == experiment_id]
        if entries.empty:
            raise ValueError(f"No experiment with id '{experiment_id}' found in log.")

        entry = entries.iloc[0]
        metrics = {}

        if entry["metrics_pre_csv"]:
            metrics["pre"] = pd.read_csv(entry["metrics_pre_csv"])
        if entry["metrics_post_csv"]:
            metrics["post"] = pd.read_csv(entry["metrics_post_csv"])

        return metrics