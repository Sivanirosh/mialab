"""Experimental framework for brain tissue segmentation experiments.

This module provides tools to run experiments, track results, and compare different configurations.
"""

import os
import json
import datetime
import shutil
import argparse
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import the main pipeline function
from pipeline import main as run_pipeline


class ExperimentConfig:
    """Configuration class for experiments."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.preprocessing_params = {
            'skullstrip_pre': True,
            'normalization_pre': True,
            'registration_pre': True,
            'coordinates_feature': True,
            'intensity_feature': True,
            'gradient_intensity_feature': True
        }
        self.postprocessing_params = {
            'simple_post': True
        }
        self.forest_params = {
            'n_estimators': 10,
            'max_depth': 10,
            'max_features': None  # Will be set based on feature count
        }
        
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = {
            'name': self.name,
            'description': self.description,
            'preprocessing_params': self.preprocessing_params,
            'postprocessing_params': self.postprocessing_params,
            'forest_params': self.forest_params
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls(config_dict['name'], config_dict['description'])
        config.preprocessing_params = config_dict['preprocessing_params']
        config.postprocessing_params = config_dict['postprocessing_params']
        config.forest_params = config_dict['forest_params']
        return config


class ExperimentManager:
    """Manages and tracks experiments."""
    
    def __init__(self, base_experiment_dir: str = "./experiments"):
        self.base_experiment_dir = base_experiment_dir
        os.makedirs(base_experiment_dir, exist_ok=True)
        
        # Create experiments log file
        self.log_file = os.path.join(base_experiment_dir, "experiments_log.csv")
        if not os.path.exists(self.log_file):
            self._initialize_log()
    
    def _initialize_log(self):
        """Initialize the experiments log file."""
        log_data = pd.DataFrame(columns=[
            'experiment_id', 'name', 'description', 'timestamp', 
            'config_file', 'results_dir', 'status'
        ])
        log_data.to_csv(self.log_file, index=False)
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment and return its ID."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f"{config.name}_{timestamp}"
        
        # Create experiment directory
        exp_dir = os.path.join(self.base_experiment_dir, experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save configuration
        config_file = os.path.join(exp_dir, "config.json")
        config.save(config_file)
        
        # Log the experiment
        self._log_experiment(experiment_id, config.name, config.description, 
                           timestamp, config_file, exp_dir, "created")
        
        return experiment_id
    
    def _log_experiment(self, exp_id: str, name: str, description: str, 
                       timestamp: str, config_file: str, results_dir: str, status: str):
        """Log experiment details."""
        log_data = pd.read_csv(self.log_file)
        new_row = pd.DataFrame({
            'experiment_id': [exp_id],
            'name': [name],
            'description': [description],
            'timestamp': [timestamp],
            'config_file': [config_file],
            'results_dir': [results_dir],
            'status': [status]
        })
        log_data = pd.concat([log_data, new_row], ignore_index=True)
        log_data.to_csv(self.log_file, index=False)
    
    def run_experiment(self, experiment_id: str, data_atlas_dir: str, 
                      data_train_dir: str, data_test_dir: str):
        """Run an experiment."""
        exp_dir = os.path.join(self.base_experiment_dir, experiment_id)
        if not os.path.exists(exp_dir):
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Load configuration
        config_file = os.path.join(exp_dir, "config.json")
        config = ExperimentConfig.load(config_file)
        
        # Create results directory
        results_dir = os.path.join(exp_dir, "results")
        
        print(f"Running experiment: {experiment_id}")
        print(f"Configuration: {config.name}")
        print(f"Description: {config.description}")
        
        try:
            # Run the pipeline with modified parameters
            self._run_pipeline_with_config(config, results_dir, data_atlas_dir, 
                                         data_train_dir, data_test_dir)
            
            # Update status
            self._update_experiment_status(experiment_id, "completed")
            
            # Generate analysis plots
            self._generate_experiment_plots(exp_dir, results_dir)
            
            print(f"Experiment {experiment_id} completed successfully!")
            
        except Exception as e:
            self._update_experiment_status(experiment_id, f"failed: {str(e)}")
            print(f"Experiment {experiment_id} failed: {str(e)}")
            raise
    
    def _run_pipeline_with_config(self, config: ExperimentConfig, results_dir: str,
                                 data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
        """Run the pipeline with specific configuration."""
        # For now, we'll call the original pipeline
        # In a more advanced version, we would modify the pipeline to accept these parameters
        run_pipeline(results_dir, data_atlas_dir, data_train_dir, data_test_dir)
    
    def _update_experiment_status(self, experiment_id: str, status: str):
        """Update experiment status in log."""
        log_data = pd.read_csv(self.log_file)
        log_data.loc[log_data['experiment_id'] == experiment_id, 'status'] = status
        log_data.to_csv(self.log_file, index=False)
    
    def _generate_experiment_plots(self, exp_dir: str, results_dir: str):
        """Generate analysis plots for the experiment."""
        # Find the results CSV file
        import glob
        result_files = glob.glob(os.path.join(results_dir, "**/results.csv"), recursive=True)
        
        if not result_files:
            print("No results.csv found")
            return
        
        results_csv = result_files[0]
        
        # Load results
        data = pd.read_csv(results_csv)
        
        # Generate plots
        self._plot_dice_coefficients(data, exp_dir)
        self._plot_hausdorff_distances(data, exp_dir)
        
    def _plot_dice_coefficients(self, data: pd.DataFrame, exp_dir: str):
        """Plot Dice coefficients."""
        dice_columns = [col for col in data.columns if 'Dice' in col and col != 'ID']
        
        if not dice_columns:
            return
        
        dice_data = []
        tissue_labels = []
        
        for col in dice_columns:
            tissue_name = col.replace('Dice', '').replace('_', '').strip()
            if not tissue_name:
                tissue_name = col
            
            values = data[col].dropna().values
            dice_data.append(values)
            tissue_labels.append(tissue_name)
        
        plt.figure(figsize=(12, 8))
        box_plot = plt.boxplot(dice_data, labels=tissue_labels, patch_artist=True)
        
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        for patch, color in zip(box_plot['boxes'], colors[:len(dice_data)]):
            patch.set_facecolor(color)
        
        plt.title('Dice Coefficients per Tissue Type', fontsize=16, fontweight='bold')
        plt.xlabel('Tissue Type', fontsize=12)
        plt.ylabel('Dice Coefficient', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        # Add mean values
        for i, (data_vals, label) in enumerate(zip(dice_data, tissue_labels)):
            mean_val = np.mean(data_vals)
            plt.text(i+1, mean_val + 0.02, f'μ={mean_val:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'dice_coefficients.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_hausdorff_distances(self, data: pd.DataFrame, exp_dir: str):
        """Plot Hausdorff distances."""
        hausdorff_columns = [col for col in data.columns if 'Hausdorff' in col and col != 'ID']
        
        if not hausdorff_columns:
            return
        
        hausdorff_data = []
        tissue_labels = []
        
        for col in hausdorff_columns:
            tissue_name = col.replace('HausdorffDistance', '').replace('_', '').strip()
            if not tissue_name:
                tissue_name = col
            
            values = data[col].dropna().values
            hausdorff_data.append(values)
            tissue_labels.append(tissue_name)
        
        plt.figure(figsize=(12, 8))
        box_plot = plt.boxplot(hausdorff_data, labels=tissue_labels, patch_artist=True)
        
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        for patch, color in zip(box_plot['boxes'], colors[:len(hausdorff_data)]):
            patch.set_facecolor(color)
        
        plt.title('Hausdorff Distances (95th percentile) per Tissue Type', fontsize=16, fontweight='bold')
        plt.xlabel('Tissue Type', fontsize=12)
        plt.ylabel('Hausdorff Distance (mm)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'hausdorff_distances.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_experiments(self, experiment_ids: List[str], output_dir: str = None):
        """Compare multiple experiments."""
        if output_dir is None:
            output_dir = os.path.join(self.base_experiment_dir, "comparisons")
        os.makedirs(output_dir, exist_ok=True)
        
        comparison_data = {}
        
        for exp_id in experiment_ids:
            exp_dir = os.path.join(self.base_experiment_dir, exp_id)
            if not os.path.exists(exp_dir):
                print(f"Warning: Experiment {exp_id} not found")
                continue
            
            # Find results file
            import glob
            result_files = glob.glob(os.path.join(exp_dir, "**/results.csv"), recursive=True)
            
            if result_files:
                data = pd.read_csv(result_files[0])
                comparison_data[exp_id] = data
        
        if len(comparison_data) < 2:
            print("Need at least 2 experiments for comparison")
            return
        
        # Generate comparison plots
        self._plot_experiment_comparison(comparison_data, output_dir)
        
        # Generate summary statistics
        self._generate_comparison_summary(comparison_data, output_dir)
    
    def _plot_experiment_comparison(self, comparison_data: Dict[str, pd.DataFrame], output_dir: str):
        """Plot comparison between experiments."""
        # Compare Dice coefficients
        dice_columns = None
        for exp_id, data in comparison_data.items():
            cols = [col for col in data.columns if 'Dice' in col and col != 'ID']
            if dice_columns is None:
                dice_columns = cols
            break
        
        if not dice_columns:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(dice_columns[:6]):  # Limit to 6 tissues
            tissue_name = col.replace('Dice', '').replace('_', '').strip()
            
            exp_data = []
            exp_labels = []
            
            for exp_id, data in comparison_data.items():
                if col in data.columns:
                    values = data[col].dropna().values
                    exp_data.append(values)
                    exp_labels.append(exp_id)
            
            if exp_data:
                axes[i].boxplot(exp_data, labels=exp_labels)
                axes[i].set_title(f'{tissue_name} - Dice Coefficient')
                axes[i].set_ylabel('Dice Coefficient')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(dice_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'experiment_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_comparison_summary(self, comparison_data: Dict[str, pd.DataFrame], output_dir: str):
        """Generate summary statistics for comparison."""
        summary_stats = {}
        
        # Get all metric columns
        all_columns = set()
        for data in comparison_data.values():
            all_columns.update([col for col in data.columns if col != 'ID'])
        
        for exp_id, data in comparison_data.items():
            stats = {}
            for col in all_columns:
                if col in data.columns:
                    values = data[col].dropna().values
                    if len(values) > 0:
                        stats[col] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values))
                        }
            summary_stats[exp_id] = stats
        
        # Save summary
        with open(os.path.join(output_dir, 'comparison_summary.json'), 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Comparison results saved to {output_dir}")
    
    def list_experiments(self):
        """List all experiments."""
        if os.path.exists(self.log_file):
            log_data = pd.read_csv(self.log_file)
            print("Experiments:")
            print(log_data[['experiment_id', 'name', 'status', 'timestamp']].to_string(index=False))
        else:
            print("No experiments found.")


def create_baseline_config() -> ExperimentConfig:
    """Create baseline configuration."""
    config = ExperimentConfig("baseline", "Baseline configuration with basic preprocessing and no post-processing")
    config.postprocessing_params['simple_post'] = False  # No post-processing for baseline
    return config


def create_postprocessing_config() -> ExperimentConfig:
    """Create configuration with post-processing."""
    config = ExperimentConfig("with_postprocessing", "Configuration with spatial regularization post-processing")
    config.postprocessing_params['simple_post'] = True  # Enable post-processing
    return config


def main():
    """Command line interface for experiment management."""
    parser = argparse.ArgumentParser(description='Experiment Management for Brain Tissue Segmentation')
    
    parser.add_argument('--action', type=str, required=True,
                       choices=['create', 'run', 'list', 'compare'],
                       help='Action to perform')
    
    parser.add_argument('--config', type=str, choices=['baseline', 'postprocessing'],
                       help='Predefined configuration to use')
    
    parser.add_argument('--experiment-id', type=str,
                       help='Experiment ID for running or comparison')
    
    parser.add_argument('--experiment-ids', type=str, nargs='+',
                       help='Multiple experiment IDs for comparison')
    
    parser.add_argument('--data-atlas-dir', type=str, default='../data/atlas',
                       help='Atlas data directory')
    
    parser.add_argument('--data-train-dir', type=str, default='../data/train',
                       help='Training data directory')
    
    parser.add_argument('--data-test-dir', type=str, default='../data/test',
                       help='Testing data directory')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.action == 'create':
        if args.config == 'baseline':
            config = create_baseline_config()
        elif args.config == 'postprocessing':
            config = create_postprocessing_config()
        else:
            raise ValueError("Please specify a valid config (baseline or postprocessing)")
        
        exp_id = manager.create_experiment(config)
        print(f"Created experiment: {exp_id}")
        
    elif args.action == 'run':
        if not args.experiment_id:
            raise ValueError("Please specify experiment ID to run")
        
        manager.run_experiment(args.experiment_id, args.data_atlas_dir,
                             args.data_train_dir, args.data_test_dir)
        
    elif args.action == 'list':
        manager.list_experiments()
        
    elif args.action == 'compare':
        if not args.experiment_ids or len(args.experiment_ids) < 2:
            raise ValueError("Please specify at least 2 experiment IDs for comparison")
        
        manager.compare_experiments(args.experiment_ids)


if __name__ == '__main__':
    main()