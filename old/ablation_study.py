"""
Comprehensive Ablation Study Framework for Brain Tissue Segmentation

This module implements a systematic ablation study to understand the contribution
of each preprocessing component and includes random forest hyperparameter optimization.
"""

import os
import json
import datetime
import itertools
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

from pipeline import main as run_pipeline
from experiment_framework import ExperimentManager, ExperimentConfig


class AblationStudyConfig:
    """Configuration for ablation study experiments."""
    
    def __init__(self):
        # Define the 9 experimental configurations
        self.experiments = {
            0: {
                'name': 'baseline_none',
                'description': 'Baseline without any preprocessing',
                'preprocessing': {
                    'skullstrip_pre': False,
                    'normalization_pre': False,
                    'registration_pre': False,
                    'coordinates_feature': True,
                    'intensity_feature': True,
                    'gradient_intensity_feature': True
                },
                'postprocessing': {'simple_post': False}
            },
            1: {
                'name': 'normalization_only',
                'description': 'With normalization only',
                'preprocessing': {
                    'skullstrip_pre': False,
                    'normalization_pre': True,
                    'registration_pre': False,
                    'coordinates_feature': True,
                    'intensity_feature': True,
                    'gradient_intensity_feature': True
                },
                'postprocessing': {'simple_post': False}
            },
            2: {
                'name': 'skullstrip_only',
                'description': 'With skull stripping only',
                'preprocessing': {
                    'skullstrip_pre': True,
                    'normalization_pre': False,
                    'registration_pre': False,
                    'coordinates_feature': True,
                    'intensity_feature': True,
                    'gradient_intensity_feature': True
                },
                'postprocessing': {'simple_post': False}
            },
            3: {
                'name': 'registration_only',
                'description': 'With registration only',
                'preprocessing': {
                    'skullstrip_pre': False,
                    'normalization_pre': False,
                    'registration_pre': True,
                    'coordinates_feature': True,
                    'intensity_feature': True,
                    'gradient_intensity_feature': True
                },
                'postprocessing': {'simple_post': False}
            },
            4: {
                'name': 'norm_skull',
                'description': 'With normalization + skull stripping',
                'preprocessing': {
                    'skullstrip_pre': True,
                    'normalization_pre': True,
                    'registration_pre': False,
                    'coordinates_feature': True,
                    'intensity_feature': True,
                    'gradient_intensity_feature': True
                },
                'postprocessing': {'simple_post': False}
            },
            5: {
                'name': 'norm_reg',
                'description': 'With normalization + registration',
                'preprocessing': {
                    'skullstrip_pre': False,
                    'normalization_pre': True,
                    'registration_pre': True,
                    'coordinates_feature': True,
                    'intensity_feature': True,
                    'gradient_intensity_feature': True
                },
                'postprocessing': {'simple_post': False}
            },
            6: {
                'name': 'reg_skull',
                'description': 'With registration + skull stripping',
                'preprocessing': {
                    'skullstrip_pre': True,
                    'normalization_pre': False,
                    'registration_pre': True,
                    'coordinates_feature': True,
                    'intensity_feature': True,
                    'gradient_intensity_feature': True
                },
                'postprocessing': {'simple_post': False}
            },
            7: {
                'name': 'all_preprocessing',
                'description': 'With normalization + skull stripping + registration',
                'preprocessing': {
                    'skullstrip_pre': True,
                    'normalization_pre': True,
                    'registration_pre': True,
                    'coordinates_feature': True,
                    'intensity_feature': True,
                    'gradient_intensity_feature': True
                },
                'postprocessing': {'simple_post': False}
            },
            8: {
                'name': 'all_preprocessing_postprocessing',
                'description': 'With all preprocessing + postprocessing',
                'preprocessing': {
                    'skullstrip_pre': True,
                    'normalization_pre': True,
                    'registration_pre': True,
                    'coordinates_feature': True,
                    'intensity_feature': True,
                    'gradient_intensity_feature': True
                },
                'postprocessing': {'simple_post': True}
            }
        }
    
    def get_experiment_config(self, exp_id: int, forest_params: dict = None) -> dict:
        """Get complete experiment configuration."""
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment ID {exp_id} not found")
        
        config = self.experiments[exp_id].copy()
        
        # Add default forest parameters if not provided
        if forest_params is None:
            forest_params = {
                'n_estimators': 50,
                'max_depth': 15,
                'max_features': 'sqrt',
                'min_samples_split': 10,
                'min_samples_leaf': 5
            }
        
        config['forest'] = forest_params
        return config


class RandomForestOptimizer:
    """Hyperparameter optimization for Random Forest classifier."""
    
    def __init__(self, data_train: np.ndarray, labels_train: np.ndarray, 
                 cv_folds: int = 3, n_jobs: int = -1):
        """Initialize the optimizer.
        
        Args:
            data_train: Training features
            labels_train: Training labels
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs
        """
        self.data_train = data_train
        self.labels_train = labels_train
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.best_params = None
        self.best_score = None
        
    def get_default_parameters(self) -> dict:
        """Get default Random Forest parameters (no optimization)."""
        return {
            'n_estimators': 100,        # sklearn default (changed from 10 in recent versions)
            'max_depth': None,          # unlimited depth (sklearn default)
            'max_features': 'sqrt',     # sqrt(n_features) (sklearn default for classification)
            'min_samples_split': 2,     # sklearn default
            'min_samples_leaf': 1,      # sklearn default
            'bootstrap': True,          # sklearn default
            'random_state': 42
        }
        
    def optimize_hyperparameters(self, optimization_level: str = 'quick') -> dict:
        """Optimize Random Forest hyperparameters.
        
        Args:
            optimization_level: 'none', 'quick', or 'full'
            
        Returns:
            Dictionary with best parameters
        """
        if optimization_level == 'none':
            print("Using default Random Forest parameters (no optimization)")
            self.best_params = self.get_default_parameters()
            self.best_score = None  # No cross-validation performed
            return self.best_params
        
        # Define parameter grids
        if optimization_level == 'quick':
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [10, 15, 20],
                'max_features': ['sqrt', 'log2'],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 5]
            }
        elif optimization_level == 'full':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, 25],
                'max_features': ['sqrt', 'log2', None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4, 5],
                'bootstrap': [True, False]
            }
        else:
            raise ValueError("optimization_level must be 'none', 'quick', or 'full'")
        
        print(f"Optimizing Random Forest hyperparameters (level: {optimization_level})...")
        print(f"Parameter grid: {param_grid}")
        print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
        print(f"Using {self.cv_folds}-fold cross-validation")
        
        # Create base classifier
        rf = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring='f1_weighted',  # Use weighted F1 for multi-class
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(self.data_train, self.labels_train)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {self.best_score:.4f}")
        
        return self.best_params


class AblationStudyRunner:
    """Runs comprehensive ablation study experiments."""
    
    def __init__(self, base_experiment_dir: str = "./ablation_experiments"):
        self.base_experiment_dir = base_experiment_dir
        self.config = AblationStudyConfig()
        os.makedirs(base_experiment_dir, exist_ok=True)
        
        # Results tracking
        self.results_log = os.path.join(base_experiment_dir, "ablation_results.csv")
        self.timing_log = os.path.join(base_experiment_dir, "timing_results.csv")
        
    def optimize_forest_for_baseline(self, data_atlas_dir: str, data_train_dir: str, 
                                   optimization_level: str = 'quick') -> dict:
        """Optimize Random Forest parameters using baseline configuration.
        
        Args:
            optimization_level: 'none', 'quick', or 'full'
        """
        if optimization_level == 'none':
            print("Step 1: Using default Random Forest parameters (no optimization)...")
            optimizer = RandomForestOptimizer(np.array([[1]]), np.array([1]))  # Dummy data
            return optimizer.get_default_parameters()
        
        print(f"Step 1: Optimizing Random Forest hyperparameters (level: {optimization_level})...")
        
        # Use experiment 7 (all preprocessing) for optimization
        opt_config = self.config.get_experiment_config(7)
        
        # Create temporary directory for optimization
        temp_dir = os.path.join(self.base_experiment_dir, "temp_optimization")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Import necessary modules for feature extraction
        import sys
        sys.path.append('.')
        import mialab.utilities.pipeline_utilities as putil
        import mialab.utilities.file_access_utilities as futil
        import mialab.data.structure as structure
        
        # Load atlas
        putil.load_atlas_images(data_atlas_dir)
        
        # Load training data
        LOADING_KEYS = [structure.BrainImageTypes.T1w,
                       structure.BrainImageTypes.T2w,
                       structure.BrainImageTypes.GroundTruth,
                       structure.BrainImageTypes.BrainMask,
                       structure.BrainImageTypes.RegistrationTransform]
        
        crawler = futil.FileSystemDataCrawler(data_train_dir,
                                             LOADING_KEYS,
                                             futil.BrainImageFilePathGenerator(),
                                             futil.DataDirectoryFilter())
        
        # Process images
        images = putil.pre_process_batch(crawler.data, opt_config['preprocessing'], multi_process=False)
        
        # Extract features for optimization
        data_train = np.concatenate([img.feature_matrix[0] for img in images])
        labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()
        
        # Sample data for faster optimization if dataset is large
        if len(data_train) > 10000:
            indices = np.random.choice(len(data_train), 10000, replace=False)
            data_train = data_train[indices]
            labels_train = labels_train[indices]
        
        # Optimize hyperparameters
        optimizer = RandomForestOptimizer(data_train, labels_train)
        best_params = optimizer.optimize_hyperparameters(optimization_level=optimization_level)
        
        # Save optimized parameters
        with open(os.path.join(self.base_experiment_dir, f"forest_params_{optimization_level}.json"), 'w') as f:
            json.dump(best_params, f, indent=2)
        
        return best_params
    
    def run_ablation_study(self, data_atlas_dir: str, data_train_dir: str, 
                          data_test_dir: str, optimization_level: str = 'quick'):
        """Run complete ablation study.
        
        Args:
            optimization_level: 'none', 'quick', or 'full'
        """
        print("Starting Comprehensive Ablation Study")
        print("=" * 60)
        
        # Validate optimization level
        valid_levels = ['none', 'quick', 'full']
        if optimization_level not in valid_levels:
            raise ValueError(f"optimization_level must be one of {valid_levels}")
        
        # Step 1: Get Random Forest parameters
        print(f"Optimization level: {optimization_level}")
        forest_params = self.optimize_forest_for_baseline(
            data_atlas_dir, data_train_dir, optimization_level
        )
        
        print(f"\nUsing Random Forest parameters: {forest_params}")
        
        # Step 2: Run all experiments
        experiment_results = {}
        
        for exp_id in range(9):
            exp_config = self.config.get_experiment_config(exp_id, forest_params)
            exp_name = exp_config['name']
            
            print(f"\n{'='*60}")
            print(f"Running Experiment {exp_id}: {exp_name}")
            print(f"Description: {exp_config['description']}")
            print(f"{'='*60}")
            
            # Create experiment directory
            exp_dir = os.path.join(self.base_experiment_dir, f"exp_{exp_id:02d}_{exp_name}")
            os.makedirs(exp_dir, exist_ok=True)
            
            # Save configuration
            config_file = os.path.join(exp_dir, "config.json")
            with open(config_file, 'w') as f:
                json.dump(exp_config, f, indent=2)
            
            # Run experiment
            try:
                start_time = datetime.datetime.now()
                run_pipeline(exp_dir, data_atlas_dir, data_train_dir, data_test_dir, exp_config)
                end_time = datetime.datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                
                # Log results
                self._log_experiment_result(exp_id, exp_name, exp_config, 
                                          exp_dir, duration, "completed", optimization_level)
                
                experiment_results[exp_id] = {
                    'config': exp_config,
                    'directory': exp_dir,
                    'duration': duration,
                    'status': 'completed'
                }
                
                print(f"✅ Experiment {exp_id} completed in {duration:.1f} seconds")
                
            except Exception as e:
                print(f"❌ Experiment {exp_id} failed: {str(e)}")
                self._log_experiment_result(exp_id, exp_name, exp_config, 
                                          exp_dir, 0, f"failed: {str(e)}", optimization_level)
                experiment_results[exp_id] = {
                    'config': exp_config,
                    'directory': exp_dir,
                    'duration': 0,
                    'status': f'failed: {str(e)}'
                }
        
        # Step 3: Analyze results
        print(f"\n{'='*60}")
        print("ANALYZING RESULTS")
        print(f"{'='*60}")
        
        self.analyze_ablation_results(experiment_results, optimization_level)
        
        return experiment_results
    
    def _log_experiment_result(self, exp_id: int, exp_name: str, config: dict,
                             exp_dir: str, duration: float, status: str, optimization_level: str):
        """Log experiment result to CSV."""
        log_data = {
            'experiment_id': exp_id,
            'experiment_name': exp_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'duration_seconds': duration,
            'status': status,
            'directory': exp_dir,
            'optimization_level': optimization_level,
            'normalization': config['preprocessing']['normalization_pre'],
            'skull_stripping': config['preprocessing']['skullstrip_pre'],
            'registration': config['preprocessing']['registration_pre'],
            'postprocessing': config['postprocessing']['simple_post']
        }
        
        # Check if log file exists
        if os.path.exists(self.results_log):
            df = pd.read_csv(self.results_log)
            df = pd.concat([df, pd.DataFrame([log_data])], ignore_index=True)
        else:
            df = pd.DataFrame([log_data])
        
        df.to_csv(self.results_log, index=False)
    
    def analyze_ablation_results(self, experiment_results: dict, optimization_level: str):
        """Analyze and visualize ablation study results."""
        # Collect results from all experiments
        all_results = []
        
        for exp_id, exp_data in experiment_results.items():
            if exp_data['status'] != 'completed':
                continue
                
            exp_dir = exp_data['directory']
            config = exp_data['config']
            
            # Find results file
            results_files = []
            for root, dirs, files in os.walk(exp_dir):
                for file in files:
                    if file == 'results.csv':
                        results_files.append(os.path.join(root, file))
            
            if not results_files:
                continue
                
            # Load results
            try:
                results_df = pd.read_csv(results_files[0])
                
                # Add experiment metadata
                results_df['experiment_id'] = exp_id
                results_df['experiment_name'] = config['name']
                results_df['optimization_level'] = optimization_level
                results_df['normalization'] = config['preprocessing']['normalization_pre']
                results_df['skull_stripping'] = config['preprocessing']['skullstrip_pre']
                results_df['registration'] = config['preprocessing']['registration_pre']
                results_df['postprocessing'] = config['postprocessing']['simple_post']
                
                all_results.append(results_df)
                
            except Exception as e:
                print(f"Error loading results for experiment {exp_id}: {e}")
        
        if not all_results:
            print("No results to analyze")
            return
        
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        combined_results.to_csv(os.path.join(self.base_experiment_dir, f"combined_results_{optimization_level}.csv"), index=False)
        
        # Generate analysis
        self._generate_ablation_analysis(combined_results, optimization_level)
    
    def _generate_ablation_analysis(self, combined_results: pd.DataFrame, optimization_level: str):
        """Generate comprehensive analysis of ablation study."""
        # Create analysis directory
        analysis_dir = os.path.join(self.base_experiment_dir, f"analysis_{optimization_level}")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 1. Summary statistics
        self._create_summary_statistics(combined_results, analysis_dir, optimization_level)
        
        # 2. Component contribution analysis
        self._analyze_component_contributions(combined_results, analysis_dir, optimization_level)
        
        # 3. Visualization
        self._create_ablation_visualizations(combined_results, analysis_dir, optimization_level)
        
        # 4. Statistical tests
        self._perform_statistical_tests(combined_results, analysis_dir)
        
        print(f"Analysis complete. Results saved to: {analysis_dir}")
    
    def _create_summary_statistics(self, combined_results: pd.DataFrame, analysis_dir: str, optimization_level: str):
        """Create summary statistics table."""
        # Get Dice coefficient columns
        dice_columns = [col for col in combined_results.columns if 'Dice' in col and col != 'ID']
        
        if not dice_columns:
            return
        
        # Calculate summary by experiment
        summary_data = []
        
        for exp_id in sorted(combined_results['experiment_id'].unique()):
            exp_data = combined_results[combined_results['experiment_id'] == exp_id]
            exp_name = exp_data['experiment_name'].iloc[0]
            
            row = {
                'experiment_id': exp_id,
                'experiment_name': exp_name,
                'optimization_level': optimization_level,
                'normalization': exp_data['normalization'].iloc[0],
                'skull_stripping': exp_data['skull_stripping'].iloc[0],
                'registration': exp_data['registration'].iloc[0],
                'postprocessing': exp_data['postprocessing'].iloc[0]
            }
            
            # Calculate mean Dice across all tissues
            dice_values = []
            for col in dice_columns:
                if col in exp_data.columns:
                    values = exp_data[col].dropna().values
                    dice_values.extend(values)
            
            if dice_values:
                row['mean_dice'] = np.mean(dice_values)
                row['std_dice'] = np.std(dice_values)
                row['min_dice'] = np.min(dice_values)
                row['max_dice'] = np.max(dice_values)
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(analysis_dir, f"summary_statistics_{optimization_level}.csv"), index=False)
        
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False))
    
    def _analyze_component_contributions(self, combined_results: pd.DataFrame, analysis_dir: str, optimization_level: str):
        """Analyze the contribution of each preprocessing component."""
        # Get mean Dice for each experiment
        exp_performance = {}
        dice_columns = [col for col in combined_results.columns if 'Dice' in col and col != 'ID']
        
        for exp_id in sorted(combined_results['experiment_id'].unique()):
            exp_data = combined_results[combined_results['experiment_id'] == exp_id]
            
            dice_values = []
            for col in dice_columns:
                if col in exp_data.columns:
                    values = exp_data[col].dropna().values
                    dice_values.extend(values)
            
            if dice_values:
                exp_performance[exp_id] = {
                    'mean_dice': np.mean(dice_values),
                    'name': exp_data['experiment_name'].iloc[0],
                    'normalization': exp_data['normalization'].iloc[0],
                    'skull_stripping': exp_data['skull_stripping'].iloc[0],
                    'registration': exp_data['registration'].iloc[0],
                    'postprocessing': exp_data['postprocessing'].iloc[0]
                }
        
        if len(exp_performance) == 0:
            print("No performance data available for component analysis")
            return
        
        # Analyze component contributions
        baseline_score = exp_performance.get(0, {}).get('mean_dice', 0)  # Experiment 0 is baseline
        
        contributions = {
            'optimization_level': optimization_level,
            'baseline_performance': baseline_score,
        }
        
        # Individual component effects (if we have the experiments)
        if 1 in exp_performance:
            contributions['normalization_only'] = exp_performance[1]['mean_dice'] - baseline_score
        if 2 in exp_performance:
            contributions['skull_stripping_only'] = exp_performance[2]['mean_dice'] - baseline_score
        if 3 in exp_performance:
            contributions['registration_only'] = exp_performance[3]['mean_dice'] - baseline_score
            
        # Component effects in full preprocessing context
        if 7 in exp_performance and 6 in exp_performance:
            contributions['normalization_effect_in_context'] = exp_performance[7]['mean_dice'] - exp_performance[6]['mean_dice']
        if 7 in exp_performance and 5 in exp_performance:
            contributions['skull_stripping_effect_in_context'] = exp_performance[7]['mean_dice'] - exp_performance[5]['mean_dice']
        if 7 in exp_performance and 4 in exp_performance:
            contributions['registration_effect_in_context'] = exp_performance[7]['mean_dice'] - exp_performance[4]['mean_dice']
        if 8 in exp_performance and 7 in exp_performance:
            contributions['postprocessing_effect'] = exp_performance[8]['mean_dice'] - exp_performance[7]['mean_dice']
        
        # Save contributions analysis
        with open(os.path.join(analysis_dir, f"component_contributions_{optimization_level}.json"), 'w') as f:
            json.dump(contributions, f, indent=2)
        
        print(f"\nComponent Contributions (optimization: {optimization_level}):")
        for component, contribution in contributions.items():
            if isinstance(contribution, (int, float)):
                print(f"{component}: {contribution:+.4f}")
            else:
                print(f"{component}: {contribution}")
    
    def _create_ablation_visualizations(self, combined_results: pd.DataFrame, analysis_dir: str, optimization_level: str):
        """Create visualizations for ablation study."""
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Get Dice columns
        dice_columns = [col for col in combined_results.columns if 'Dice' in col and col != 'ID']
        
        if not dice_columns:
            return
        
        # 1. Overall performance comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Calculate mean Dice for each experiment
        exp_means = []
        exp_names = []
        exp_ids = []
        
        for exp_id in sorted(combined_results['experiment_id'].unique()):
            exp_data = combined_results[combined_results['experiment_id'] == exp_id]
            exp_name = exp_data['experiment_name'].iloc[0]
            
            dice_values = []
            for col in dice_columns:
                if col in exp_data.columns:
                    values = exp_data[col].dropna().values
                    dice_values.extend(values)
            
            if dice_values:
                exp_means.append(np.mean(dice_values))
                exp_names.append(f"{exp_id}: {exp_name}")
                exp_ids.append(exp_id)
        
        bars = ax.bar(exp_names, exp_means)
        ax.set_ylabel('Mean Dice Coefficient')
        ax.set_title(f'Ablation Study: Overall Performance Comparison (Optimization: {optimization_level})')
        ax.tick_params(axis='x', rotation=45)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if exp_means[i] == max(exp_means):
                bar.set_color('gold')
            elif exp_means[i] == min(exp_means):
                bar.set_color('lightcoral')
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"overall_performance_{optimization_level}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Component effect analysis
        self._plot_component_effects(combined_results, analysis_dir, optimization_level)
        
        # 3. Per-tissue analysis
        self._plot_per_tissue_analysis(combined_results, analysis_dir, optimization_level)
    
    def _plot_component_effects(self, combined_results: pd.DataFrame, analysis_dir: str, optimization_level: str):
        """Plot the effects of individual components."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        components = ['normalization', 'skull_stripping', 'registration', 'postprocessing']
        
        dice_columns = [col for col in combined_results.columns if 'Dice' in col and col != 'ID']
        
        for i, component in enumerate(components):
            ax = axes[i]
            
            # Compare experiments with and without this component
            with_component = []
            without_component = []
            
            for exp_id in sorted(combined_results['experiment_id'].unique()):
                exp_data = combined_results[combined_results['experiment_id'] == exp_id]
                
                dice_values = []
                for col in dice_columns:
                    if col in exp_data.columns:
                        values = exp_data[col].dropna().values
                        dice_values.extend(values)
                
                if dice_values:
                    mean_dice = np.mean(dice_values)
                    if exp_data[component].iloc[0]:
                        with_component.append(mean_dice)
                    else:
                        without_component.append(mean_dice)
            
            # Box plot
            data_to_plot = [without_component, with_component]
            labels = [f'Without {component}', f'With {component}']
            
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_title(f'Effect of {component.replace("_", " ").title()}')
            ax.set_ylabel('Mean Dice Coefficient')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Component Effects (Optimization: {optimization_level})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"component_effects_{optimization_level}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_tissue_analysis(self, combined_results: pd.DataFrame, analysis_dir: str, optimization_level: str):
        """Plot per-tissue performance analysis."""
        dice_columns = [col for col in combined_results.columns if 'Dice' in col and col != 'ID']
        
        if len(dice_columns) == 0:
            return
        
        # Create subplot for each tissue
        n_tissues = len(dice_columns)
        n_cols = 3
        n_rows = (n_tissues + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, dice_col in enumerate(dice_columns):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row][col]
            
            tissue_name = dice_col.replace('Dice', '').replace('_', '').strip()
            if not tissue_name:
                tissue_name = dice_col
            
            # Get performance for each experiment
            exp_means = []
            exp_names = []
            
            for exp_id in sorted(combined_results['experiment_id'].unique()):
                exp_data = combined_results[combined_results['experiment_id'] == exp_id]
                
                if dice_col in exp_data.columns:
                    values = exp_data[dice_col].dropna().values
                    if len(values) > 0:
                        exp_means.append(np.mean(values))
                        exp_names.append(f"{exp_id}")
            
            ax.bar(exp_names, exp_means)
            ax.set_title(f'{tissue_name}')
            ax.set_ylabel('Mean Dice Coefficient')
            ax.set_xlabel('Experiment ID')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_tissues, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row][col].set_visible(False)
        
        plt.suptitle(f'Per-Tissue Analysis (Optimization: {optimization_level})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f"per_tissue_analysis_{optimization_level}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _perform_statistical_tests(self, combined_results: pd.DataFrame, analysis_dir: str):
        """Perform statistical significance tests."""
        from scipy import stats
        
        dice_columns = [col for col in combined_results.columns if 'Dice' in col and col != 'ID']
        
        if not dice_columns:
            return
        
        # Collect all experiments' Dice scores
        experiment_scores = {}
        
        for exp_id in sorted(combined_results['experiment_id'].unique()):
            exp_data = combined_results[combined_results['experiment_id'] == exp_id]
            
            dice_values = []
            for col in dice_columns:
                if col in exp_data.columns:
                    values = exp_data[col].dropna().values
                    dice_values.extend(values)
            
            if dice_values:
                experiment_scores[exp_id] = dice_values
        
        # Perform pairwise t-tests
        test_results = []
        
        for i, exp1 in enumerate(experiment_scores.keys()):
            for exp2 in list(experiment_scores.keys())[i+1:]:
                statistic, p_value = stats.ttest_ind(
                    experiment_scores[exp1], 
                    experiment_scores[exp2]
                )
                
                test_results.append({
                    'experiment_1': exp1,
                    'experiment_2': exp2,
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        
        # Save statistical test results
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(os.path.join(analysis_dir, "statistical_tests.csv"), index=False)
        
        print(f"\nStatistical Tests (p < 0.05):")
        significant_tests = test_df[test_df['significant']]
        for _, row in significant_tests.iterrows():
            print(f"Exp {row['experiment_1']} vs Exp {row['experiment_2']}: p = {row['p_value']:.4f}")


def main():
    """Main function to run ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive ablation study')
    parser.add_argument('--data-atlas-dir', type=str, default='../data/atlas',
                       help='Atlas data directory')
    parser.add_argument('--data-train-dir', type=str, default='../data/train',
                       help='Training data directory')
    parser.add_argument('--data-test-dir', type=str, default='../data/test',
                       help='Testing data directory')
    parser.add_argument('--optimization-level', type=str, default='quick',
                       choices=['none', 'quick', 'full'],
                       help='Random Forest optimization level: none (default params), quick (small grid), full (large grid)')
    parser.add_argument('--experiment-dir', type=str, default='./ablation_experiments',
                       help='Base directory for experiments')
    
    args = parser.parse_args()
    
    print(f"Optimization level: {args.optimization_level}")
    if args.optimization_level == 'none':
        print("  -> Using default sklearn Random Forest parameters")
    elif args.optimization_level == 'quick':
        print("  -> Using small hyperparameter grid (~48 combinations)")
    elif args.optimization_level == 'full':
        print("  -> Using large hyperparameter grid (~864 combinations)")
    
    # Create and run ablation study
    runner = AblationStudyRunner(args.experiment_dir)
    
    results = runner.run_ablation_study(
        args.data_atlas_dir,
        args.data_train_dir, 
        args.data_test_dir,
        optimization_level=args.optimization_level
    )
    
    print("\nAblation study completed!")
    print(f"Results saved to: {args.experiment_dir}")
    print(f"Analysis directory: {args.experiment_dir}/analysis_{args.optimization_level}/")


if __name__ == '__main__':
    main()