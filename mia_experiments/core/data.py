"""Core data handling module for brain tissue segmentation experiments.

This module provides utilities for loading, parsing, and managing experiment data.
"""

import os
import pandas as pd
import numpy as np
import glob
from typing import Dict, List, Optional, Tuple
import json


class ExperimentData:
    """Container for experiment data with metadata."""
    
    def __init__(self, experiment_id: int, name: str, data: pd.DataFrame):
        self.experiment_id = experiment_id
        self.name = name
        self.data = data
        self.config = {}
        
    @property
    def dice_columns(self) -> List[str]:
        """Get columns containing Dice coefficient data."""
        return [col for col in self.data.columns if 'DICE' in col.upper()]
    
    @property
    def hausdorff_columns(self) -> List[str]:
        """Get columns containing Hausdorff distance data."""
        return [col for col in self.data.columns if 'HAUSDORFF' in col.upper() or 'HDRFDST' in col.upper()]
    
    @property
    def tissue_labels(self) -> List[str]:
        """Get unique tissue labels in the data."""
        if 'LABEL' in self.data.columns:
            return sorted(self.data['LABEL'].unique())
        return []
    
    @property
    def subjects(self) -> List[str]:
        """Get unique subjects in the data."""
        if 'SUBJECT' in self.data.columns:
            return sorted(self.data['SUBJECT'].unique())
        return []
    
    def get_dice_scores(self, tissue_label: Optional[str] = None) -> np.ndarray:
        """Get Dice scores, optionally filtered by tissue label."""
        dice_col = self.dice_columns[0] if self.dice_columns else 'DICE'
        
        if tissue_label is not None and 'LABEL' in self.data.columns:
            filtered_data = self.data[self.data['LABEL'] == tissue_label]
            return filtered_data[dice_col].dropna().values
        else:
            return self.data[dice_col].dropna().values
    
    def get_hausdorff_distances(self, tissue_label: Optional[str] = None) -> np.ndarray:
        """Get Hausdorff distances, optionally filtered by tissue label."""
        hausdorff_col = self.hausdorff_columns[0] if self.hausdorff_columns else 'HDRFDST'
        
        if tissue_label is not None and 'LABEL' in self.data.columns:
            filtered_data = self.data[self.data['LABEL'] == tissue_label]
            return filtered_data[hausdorff_col].dropna().values
        else:
            return self.data[hausdorff_col].dropna().values
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for this experiment."""
        dice_scores = self.get_dice_scores()
        hausdorff_scores = self.get_hausdorff_distances()
        
        stats = {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'n_measurements': len(self.data),
            'n_subjects': len(self.subjects),
            'n_tissues': len(self.tissue_labels)
        }
        
        if len(dice_scores) > 0:
            stats.update({
                'mean_dice': float(np.mean(dice_scores)),
                'std_dice': float(np.std(dice_scores)),
                'min_dice': float(np.min(dice_scores)),
                'max_dice': float(np.max(dice_scores))
            })
        
        if len(hausdorff_scores) > 0:
            stats.update({
                'mean_hausdorff': float(np.mean(hausdorff_scores)),
                'std_hausdorff': float(np.std(hausdorff_scores))
            })
        
        return stats


class DataLoader:
    """Loads experiment data from various sources."""
    
    @staticmethod
    def load_from_csv(file_path: str, experiment_id: int = 0, name: str = None) -> ExperimentData:
        """Load experiment data from a single CSV file."""
        try:
            # Try standard comma-separated format first
            data = pd.read_csv(file_path)
        except:
            # Fallback to semicolon-separated format
            data = pd.read_csv(file_path, sep=';')
        
        # Clean up data
        data = DataLoader._clean_data(data)
        
        if name is None:
            name = os.path.basename(file_path).replace('.csv', '')
        
        return ExperimentData(experiment_id, name, data)
    
    @staticmethod
    def load_ablation_experiments(base_dir: str) -> List[ExperimentData]:
        """Load all experiments from an ablation study directory."""
        experiments = []
        
        # Find experiment directories
        exp_dirs = glob.glob(os.path.join(base_dir, "exp_*"))
        exp_dirs.sort()
        
        for exp_dir in exp_dirs:
            exp_name = os.path.basename(exp_dir)
            
            # Extract experiment ID
            try:
                exp_id = int(exp_name.split('_')[1])
            except:
                continue
            
            # Find results CSV files
            results_files = glob.glob(os.path.join(exp_dir, "**/results.csv"), recursive=True)
            
            if results_files:
                try:
                    exp_data = DataLoader.load_from_csv(results_files[0], exp_id, exp_name)
                    
                    # Try to load configuration
                    config_file = os.path.join(exp_dir, "config.json")
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            exp_data.config = json.load(f)
                    
                    experiments.append(exp_data)
                except Exception as e:
                    print(f"Warning: Could not load experiment {exp_name}: {e}")
        
        return experiments
    
    @staticmethod
    def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data format."""
        # Remove header rows that may have been included as data
        if 'DICE' in data.columns:
            data = data[data['DICE'] != 'DICE']
        
        # Convert numeric columns
        numeric_columns = ['DICE', 'HDRFDST']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove rows with missing critical data
        if 'DICE' in data.columns:
            data = data.dropna(subset=['DICE'])
        
        return data.reset_index(drop=True)


class ExperimentCollection:
    """Collection of experiments with analysis capabilities."""
    
    def __init__(self, experiments: List[ExperimentData]):
        self.experiments = {exp.experiment_id: exp for exp in experiments}
    
    def get_experiment(self, experiment_id: int) -> Optional[ExperimentData]:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)
    
    def get_experiment_ids(self) -> List[int]:
        """Get all experiment IDs."""
        return sorted(self.experiments.keys())
    
    def get_combined_data(self) -> pd.DataFrame:
        """Get combined data from all experiments."""
        all_data = []
        
        for exp in self.experiments.values():
            exp_data = exp.data.copy()
            exp_data['experiment_id'] = exp.experiment_id
            exp_data['experiment_name'] = exp.name
            all_data.append(exp_data)
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    def get_summary_table(self) -> pd.DataFrame:
        """Get summary statistics table for all experiments."""
        summaries = []
        for exp in self.experiments.values():
            summaries.append(exp.get_summary_stats())
        
        return pd.DataFrame(summaries)
    
    def filter_experiments(self, experiment_ids: List[int]) -> 'ExperimentCollection':
        """Create new collection with filtered experiments."""
        filtered_experiments = [self.experiments[exp_id] for exp_id in experiment_ids 
                              if exp_id in self.experiments]
        return ExperimentCollection(filtered_experiments)