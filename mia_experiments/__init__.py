"""MIA Experiments Package - Brain Tissue Segmentation Experiments Framework

A comprehensive framework for running ablation studies and experiments on brain 
tissue segmentation pipelines with systematic analysis and visualization.
"""

import os

from .core.data import ExperimentData, DataLoader, ExperimentCollection
from .core.analysis import ComponentAnalyzer, StatisticalAnalyzer, PerformanceAnalyzer, ResultsExporter
from .core.visualization import ExperimentVisualizer, PlotStyle
from .core.config import (
    ExperimentConfig, PreprocessingConfig, PostprocessingConfig, RandomForestConfig,
    OptimizationLevel, AblationStudyConfigurator, RandomForestOptimizer, create_default_config
)
from .experiments import ExperimentManager, AblationStudyManager, ExperimentComparator

__version__ = "1.0.0"
__author__ = "MIA Lab"

# Main entry points
def run_ablation_study(data_atlas_dir: str, data_train_dir: str, data_test_dir: str,
                      optimization_level: str = "quick", output_dir: str = "./ablation_experiments",
                      study_type: str = "preprocessing"):
    """
    Run ablation study with configurable study type.
    
    Args:
        data_atlas_dir: Path to atlas data directory
        data_train_dir: Path to training data directory
        data_test_dir: Path to test data directory
        optimization_level: 'none', 'quick', or 'full'
        output_dir: Output directory for results
        study_type: Type of ablation study ('preprocessing', 'postprocessing', or 'combined')
    
    Returns:
        Dictionary with experiment results
    """
    level = OptimizationLevel(optimization_level)
    manager = AblationStudyManager(output_dir)
    return manager.run_ablation_study(data_atlas_dir, data_train_dir, data_test_dir, level, study_type)


def analyze_results(experiment_dir: str, optimization_level: str = "quick"):
    """
    Analyze ablation study results.
    
    Args:
        experiment_dir: Directory containing experiment results
        optimization_level: Optimization level used in experiments
    """
    level = OptimizationLevel(optimization_level)
    manager = AblationStudyManager(experiment_dir)
    manager.analyze_ablation_results(level)


def load_and_visualize(experiment_dir: str, output_dir: str = None):
    """
    Load experiment results and create visualizations.
    
    Args:
        experiment_dir: Directory containing experiment results
        output_dir: Output directory for plots (default: experiment_dir/plots)
    """
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, "plots")
    
    experiments = DataLoader.load_ablation_experiments(experiment_dir)
    if experiments:
        collection = ExperimentCollection(experiments)
        visualizer = ExperimentVisualizer(collection)
        return visualizer.create_comprehensive_report(output_dir)
    else:
        print("No experiments found in directory")
        return {}

# Make commonly used classes easily accessible
__all__ = [
    'ExperimentData', 'DataLoader', 'ExperimentCollection',
    'ComponentAnalyzer', 'StatisticalAnalyzer', 'PerformanceAnalyzer', 'ResultsExporter',
    'ExperimentVisualizer', 'PlotStyle',
    'ExperimentConfig', 'PreprocessingConfig', 'PostprocessingConfig', 'RandomForestConfig',
    'OptimizationLevel', 'AblationStudyConfigurator', 'RandomForestOptimizer',
    'ExperimentManager', 'AblationStudyManager', 'ExperimentComparator',
    'run_ablation_study', 'analyze_results', 'load_and_visualize', 'create_default_config'
]