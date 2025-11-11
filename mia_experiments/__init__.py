"""Simplified entry points for running the single all-preprocessing experiment."""

from __future__ import annotations

from .core.analysis import ComponentAnalyzer, PerformanceAnalyzer, ResultsExporter, StatisticalAnalyzer
from .core.config import (
    ExperimentConfig,
    OptimizationLevel,
    PostprocessingConfig,
    PreprocessingConfig,
    RandomForestConfig,
    RandomForestOptimizer,
    create_all_preprocessing_config,
    create_default_config,
)
from .core.data import DataLoader, ExperimentCollection, ExperimentData
from .core.visualization import ExperimentVisualizer, PlotStyle
from .experiments import ExperimentManager

__version__ = "2.0.0"
__author__ = "MIA Lab"


def run_all_preprocessing(
    data_atlas_dir: str,
    data_train_dir: str,
    data_test_dir: str,
    *,
    output_dir: str = "./experiments",
    include_postprocessing: bool = True,
) -> dict:
    """Execute the canonical experiment and return key artefact paths."""

    manager = ExperimentManager(output_dir)
    return manager.run_all_preprocessing_experiment(
        data_atlas_dir=data_atlas_dir,
        data_train_dir=data_train_dir,
        data_test_dir=data_test_dir,
        include_postprocessing=include_postprocessing,
    )


__all__ = [
    "ExperimentData",
    "DataLoader",
    "ExperimentCollection",
    "ComponentAnalyzer",
    "StatisticalAnalyzer",
    "PerformanceAnalyzer",
    "ResultsExporter",
    "ExperimentVisualizer",
    "PlotStyle",
    "ExperimentConfig",
    "PreprocessingConfig",
    "PostprocessingConfig",
    "RandomForestConfig",
    "OptimizationLevel",
    "RandomForestOptimizer",
    "ExperimentManager",
    "create_all_preprocessing_config",
    "create_default_config",
    "run_all_preprocessing",
]
