"""Experiment management module for running and tracking experiments.

This module handles experiment execution, logging, and result management.
"""

import os
import json
import datetime
import shutil
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

from .core.config import ExperimentConfig, OptimizationLevel, RandomForestConfig, RandomForestOptimizer
from .core.data import DataLoader, ExperimentCollection


class ExperimentManager:
    """Manages experiment execution and tracking."""
    
    def __init__(self, base_experiment_dir: str = "./experiments"):
        self.base_experiment_dir = base_experiment_dir
        os.makedirs(base_experiment_dir, exist_ok=True)
        
        # Initialize experiment log
        self.log_file = os.path.join(base_experiment_dir, "experiment_log.csv")
        self._initialize_log()
    
    def _initialize_log(self):
        """Initialize experiment log file."""
        if not os.path.exists(self.log_file):
            log_df = pd.DataFrame(columns=[
                'experiment_id', 'name', 'description', 'timestamp', 
                'status', 'duration_seconds', 'config_file', 'results_dir'
            ])
            log_df.to_csv(self.log_file, index=False)
    
    def run_single_experiment(self, config: ExperimentConfig, 
                            data_atlas_dir: str, data_train_dir: str, data_test_dir: str,
                            experiment_id: Optional[str] = None) -> Tuple[str, bool]:
        """Run a single experiment.
        
        Returns:
            Tuple of (experiment_directory, success_flag)
        """
        if experiment_id is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_id = f"{config.name}_{timestamp}"
        
        # Create experiment directory
        exp_dir = os.path.join(self.base_experiment_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save configuration
        config_file = os.path.join(exp_dir, "config.json")
        config.save(config_file)
        
        print(f"Running experiment: {experiment_id}")
        print(f"Description: {config.description}")
        
        start_time = datetime.datetime.now()
        
        try:
            # Log start
            self._log_experiment(experiment_id, config.name, config.description, 
                               start_time.isoformat(), "running", 0, config_file, exp_dir)
            
            # Run the pipeline
            self._run_pipeline_with_config(config, exp_dir, data_atlas_dir, 
                                         data_train_dir, data_test_dir)
            
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log completion
            self._log_experiment(experiment_id, config.name, config.description, 
                               start_time.isoformat(), "completed", duration, config_file, exp_dir)
            
            print(f"Experiment {experiment_id} completed in {duration:.1f} seconds")
            return exp_dir, True
            
        except Exception as e:
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log failure
            self._log_experiment(experiment_id, config.name, config.description, 
                               start_time.isoformat(), f"failed: {str(e)}", duration, config_file, exp_dir)
            
            print(f"Experiment {experiment_id} failed: {str(e)}")
            return exp_dir, False
    
    def _run_pipeline_with_config(self, config: ExperimentConfig, exp_dir: str,
                                data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
        """Run the pipeline with specific configuration."""
        # Import the pipeline function
        try:
            from pipeline import main as run_pipeline
        except ImportError:
            raise ImportError("Could not import pipeline module. Make sure pipeline.py is available.")
        
        # Convert config to pipeline format
        pipeline_config = config.to_pipeline_dict()
        
        # Run the pipeline
        run_pipeline(exp_dir, data_atlas_dir, data_train_dir, data_test_dir, pipeline_config)
    
    def _log_experiment(self, exp_id: str, name: str, description: str, 
                       timestamp: str, status: str, duration: float, 
                       config_file: str, results_dir: str):
        """Log experiment details."""
        if os.path.exists(self.log_file):
            log_df = pd.read_csv(self.log_file)
            
            # Update existing entry or add new one
            mask = log_df['experiment_id'] == exp_id
            if mask.any():
                log_df['duration_seconds'] = log_df['duration_seconds'].astype(float)
                log_df.loc[mask, 'status'] = status
                log_df.loc[mask, 'duration_seconds'] = float(duration)

            else:
                new_row = pd.DataFrame({
                    'experiment_id': [exp_id],
                    'name': [name],
                    'description': [description], 
                    'timestamp': [timestamp],
                    'status': [status],
                    'duration_seconds': [duration],
                    'config_file': [config_file],
                    'results_dir': [results_dir]
                })
                log_df = pd.concat([log_df, new_row], ignore_index=True)
        else:
            log_df = pd.DataFrame({
                'experiment_id': [exp_id],
                'name': [name],
                'description': [description],
                'timestamp': [timestamp], 
                'status': [status],
                'duration_seconds': [duration],
                'config_file': [config_file],
                'results_dir': [results_dir]
            })
        
        log_df.to_csv(self.log_file, index=False)
    
    def list_experiments(self) -> pd.DataFrame:
        """List all experiments."""
        if os.path.exists(self.log_file):
            return pd.read_csv(self.log_file)
        else:
            return pd.DataFrame()
    
    def get_experiment_results(self, experiment_id: str) -> Optional[pd.DataFrame]:
        """Get results for a specific experiment."""
        log_df = self.list_experiments()
        
        if experiment_id in log_df['experiment_id'].values:
            exp_row = log_df[log_df['experiment_id'] == experiment_id].iloc[0]
            results_dir = exp_row['results_dir']
            
            # Look for results CSV
            import glob
            results_files = glob.glob(os.path.join(results_dir, "**/results.csv"), recursive=True)
            
            if results_files:
                return pd.read_csv(results_files[0])
        
        return None


class AblationStudyManager:
    """Manages comprehensive ablation studies."""
    
    def __init__(self, base_experiment_dir: str = "./ablation_experiments"):
        self.base_experiment_dir = base_experiment_dir
        self.exp_manager = ExperimentManager(base_experiment_dir)
        os.makedirs(base_experiment_dir, exist_ok=True)
        
        # Ablation-specific logs
        self.ablation_log = os.path.join(base_experiment_dir, "ablation_log.json")
    
    def optimize_random_forest(self, data_atlas_dir: str, data_train_dir: str,
                             optimization_level: OptimizationLevel) -> RandomForestConfig:
        """Optimize Random Forest hyperparameters."""
        
        if optimization_level == OptimizationLevel.NONE:
            print("Using default Random Forest parameters (no optimization)")
            return RandomForestOptimizer.get_default_parameters()
        
        print(f"Optimizing Random Forest hyperparameters (level: {optimization_level.value})")
        
        # Import required modules for feature extraction
        try:
            import sys
            sys.path.append('.')
            import mialab.utilities.pipeline_utilities as putil
            import mialab.utilities.file_access_utilities as futil
            import mialab.data.structure as structure
        except ImportError:
            raise ImportError("Could not import mialab modules. Make sure they are available.")
        
        # Load atlas
        putil.load_atlas_images(data_atlas_dir)
        
        # Load training data for optimization
        LOADING_KEYS = [
            structure.BrainImageTypes.T1w,
            structure.BrainImageTypes.T2w,
            structure.BrainImageTypes.GroundTruth,
            structure.BrainImageTypes.BrainMask,
            structure.BrainImageTypes.RegistrationTransform
        ]
        
        crawler = futil.FileSystemDataCrawler(
            data_train_dir, LOADING_KEYS, 
            futil.BrainImageFilePathGenerator(),
            futil.DataDirectoryFilter()
        )
        
        # Use full preprocessing for optimization (experiment 7 configuration)
        opt_preprocessing = {
            'skullstrip_pre': True,
            'normalization_pre': True,
            'registration_pre': True,
            'coordinates_feature': True,
            'intensity_feature': True,
            'gradient_intensity_feature': True
        }
        
        # Process images
        images = putil.pre_process_batch(crawler.data, opt_preprocessing, multi_process=True)
        
        # Extract features
        data_train = np.concatenate([img.feature_matrix[0] for img in images])
        labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()
        
        # Sample data for faster optimization if dataset is large
        if len(data_train) > 10000:
            print(f"Sampling {10000} samples from {len(data_train)} for optimization")
            indices = np.random.choice(len(data_train), 10000, replace=False)
            data_train = data_train[indices]
            labels_train = labels_train[indices]
        
        # Get parameter grid
        param_grid = RandomForestOptimizer.get_parameter_grid(optimization_level)
        n_combinations = np.prod([len(v) for v in param_grid.values()])
        
        print(f"Testing {n_combinations} parameter combinations with 3-fold cross-validation")
        
        # Create base classifier
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(data_train, labels_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Create optimized config
        optimized_config = RandomForestConfig(**grid_search.best_params_)
        optimized_config.random_state = 42  # Ensure reproducibility
        optimized_config.n_jobs = -1  # Use all cores
        
        # Save optimization results
        opt_results = {
            'optimization_level': optimization_level.value,
            'best_parameters': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'n_combinations_tested': int(n_combinations),
            'optimization_timestamp': datetime.datetime.now().isoformat()
        }
        
        opt_file = os.path.join(self.base_experiment_dir, f"optimization_{optimization_level.value}.json")
        with open(opt_file, 'w') as f:
            json.dump(opt_results, f, indent=2)
        
        return optimized_config
    
    def run_ablation_study(self, data_atlas_dir: str, data_train_dir: str, data_test_dir: str,
                          optimization_level: OptimizationLevel = OptimizationLevel.QUICK,
                          study_type: str = 'preprocessing') -> Dict[int, Dict]:
        """Run ablation study with configurable study type.
        
        Args:
            data_atlas_dir: Path to atlas data
            data_train_dir: Path to training data
            data_test_dir: Path to test data
            optimization_level: RF optimization level
            study_type: Type of ablation study to run:
                - 'preprocessing': Original 9-experiment preprocessing ablation (default)
                - 'postprocessing': 9-experiment postprocessing ablation
                - 'combined': 18-experiment combined preprocessing + postprocessing ablation
        
        Returns:
            Dictionary of experiment results
        """
        
        print("Starting Comprehensive Ablation Study")
        print("=" * 60)
        print(f"Study type: {study_type}")
        print(f"Optimization level: {optimization_level.value}")
        
        # Step 1: Optimize Random Forest parameters
        forest_config = self.optimize_random_forest(data_atlas_dir, data_train_dir, optimization_level)
        
        # Step 2: Create ablation configurations based on study type
        from .core.config import AblationStudyConfigurator
        
        if study_type == 'preprocessing':
            ablation_configs = AblationStudyConfigurator.create_ablation_configs(forest_config)
            n_experiments = 8
            get_summary = AblationStudyConfigurator.get_experiment_summary
        elif study_type == 'postprocessing':
            ablation_configs = AblationStudyConfigurator.create_postprocessing_ablation_configs(forest_config)
            n_experiments = 9
            get_summary = AblationStudyConfigurator.get_postprocessing_experiment_summary
        elif study_type == 'combined':
            ablation_configs = AblationStudyConfigurator.create_combined_ablation_configs(forest_config)
            n_experiments = 16
            get_summary = AblationStudyConfigurator.get_combined_experiment_summary
        else:
            raise ValueError(f"Unknown study_type: {study_type}. Must be 'preprocessing', 'postprocessing', or 'combined'")
        
        print(f"\nRunning {n_experiments} experiments")
        print("\nExperiment Plan:")
        for exp_id, desc in get_summary().items():
            print(f"  {exp_id}: {desc}")
        
        # Step 3: Run all experiments
        experiment_results = {}
        
        for exp_id in range(n_experiments):
            config = ablation_configs[exp_id]
            
            print(f"\n{'='*60}")
            print(f"Running Experiment {exp_id}/{n_experiments-1}: {config.name}")
            print(f"Description: {config.description}")
            print(f"{'='*60}")
            
            exp_dir, success = self.exp_manager.run_single_experiment(
                config, data_atlas_dir, data_train_dir, data_test_dir,
                experiment_id=f"exp_{exp_id:02d}_{config.name}"
            )
            
            experiment_results[exp_id] = {
                'config': config,
                'directory': exp_dir,
                'success': success,
                'experiment_id': f"exp_{exp_id:02d}_{config.name}"
            }
            
            if success:
                print(f"✅ Experiment {exp_id} completed successfully")
            else:
                print(f"❌ Experiment {exp_id} failed")
        
        # Step 4: Log ablation study summary
        ablation_summary = {
            'start_time': datetime.datetime.now().isoformat(),
            'study_type': study_type,
            'optimization_level': optimization_level.value,
            'forest_config': forest_config.__dict__,
            'experiments': {
                exp_id: {
                    'name': result['config'].name,
                    'success': result['success'],
                    'directory': result['directory']
                }
                for exp_id, result in experiment_results.items()
            },
            'total_experiments': len(experiment_results),
            'successful_experiments': sum(1 for r in experiment_results.values() if r['success']),
            'failed_experiments': sum(1 for r in experiment_results.values() if not r['success'])
        }
        
        with open(self.ablation_log, 'w') as f:
            json.dump(ablation_summary, f, indent=2)
        
        # Step 5: Generate analysis if we have results
        if ablation_summary['successful_experiments'] > 0:
            print(f"\n{'='*60}")
            print("ANALYZING RESULTS")
            print(f"{'='*60}")
            
            try:
                self.analyze_ablation_results(optimization_level)
            except Exception as e:
                print(f"Analysis failed: {e}")
                print("You can run analysis separately using: mia-analyze")
        
        # Summary
        print(f"\n🎉 ABLATION STUDY COMPLETED!")
        print("=" * 60)
        print(f"Study type: {study_type}")
        print(f"Total experiments: {ablation_summary['total_experiments']}")
        print(f"Successful: {ablation_summary['successful_experiments']}")
        print(f"Failed: {ablation_summary['failed_experiments']}")
        print(f"Results directory: {self.base_experiment_dir}")
        
        return experiment_results
    
    def analyze_ablation_results(self, optimization_level: OptimizationLevel):
        """Analyze ablation study results."""
        from .core.analysis import ResultsExporter
        from .core.visualization import ExperimentVisualizer
        
        # Load experiment data
        experiments = DataLoader.load_ablation_experiments(self.base_experiment_dir)
        
        if len(experiments) == 0:
            print("No experiment results found for analysis")
            return
        
        experiment_collection = ExperimentCollection(experiments)
        
        print(f"Loaded {len(experiments)} experiments for analysis")
        
        # Create analysis directory
        analysis_dir = os.path.join(self.base_experiment_dir, f"analysis_{optimization_level.value}")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Export comprehensive analysis
        exporter = ResultsExporter(experiment_collection)
        exported_files = exporter.export_comprehensive_report(analysis_dir)
        
        # Create visualizations
        visualizer = ExperimentVisualizer(experiment_collection)
        plot_files = visualizer.create_comprehensive_report(analysis_dir)
        
        print(f"\nAnalysis complete! Results saved to: {analysis_dir}")
        print("\nKey files generated:")
        for file_type, file_path in {**exported_files, **plot_files}.items():
            print(f"  📄 {file_type}: {os.path.basename(file_path)}")
    
    def get_ablation_summary(self) -> Optional[Dict]:
        """Get summary of the last ablation study."""
        if os.path.exists(self.ablation_log):
            with open(self.ablation_log, 'r') as f:
                return json.load(f)
        return None


class ExperimentComparator:
    """Compares multiple experiments or ablation studies."""
    
    def __init__(self):
        pass
    
    def compare_ablation_studies(self, study_dirs: List[str], output_dir: str):
        """Compare multiple ablation studies."""
        from .core.visualization import ExperimentVisualizer
        
        all_experiments = []
        study_names = []
        
        for i, study_dir in enumerate(study_dirs):
            experiments = DataLoader.load_ablation_experiments(study_dir)
            
            # Add study identifier to experiment names
            study_name = os.path.basename(study_dir)
            study_names.append(study_name)
            
            for exp in experiments:
                exp.name = f"{study_name}_{exp.name}"
            
            all_experiments.extend(experiments)
        
        if not all_experiments:
            print("No experiments found in provided directories")
            return
        
        experiment_collection = ExperimentCollection(all_experiments)
        
        # Create comparison visualizations
        os.makedirs(output_dir, exist_ok=True)
        
        visualizer = ExperimentVisualizer(experiment_collection)
        
        # Overall comparison
        fig = visualizer.plot_overall_performance(
            save_path=os.path.join(output_dir, "study_comparison.png"),
            title_suffix=f" - Comparing {len(study_names)} Studies"
        )
        plt.close(fig)
        
        # Component effects comparison
        fig = visualizer.plot_component_effects(
            save_path=os.path.join(output_dir, "component_comparison.png"),
            title_suffix=f" - Comparing {len(study_names)} Studies"
        )
        plt.close(fig)
        
        print(f"Comparison complete! Results saved to: {output_dir}")