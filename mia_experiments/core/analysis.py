"""Core analysis module for ablation studies and component analysis.

This module provides tools for analyzing experiment results, computing component
contributions, and performing statistical tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import json

from .data import ExperimentCollection, ExperimentData


class ComponentAnalyzer:
    """Analyzes the contribution of individual preprocessing components."""
    
    def __init__(self, experiments: ExperimentCollection):
        self.experiments = experiments
        
        # Define experiment configurations for ablation study
        self.ablation_configs = {
            0: {'normalization': False, 'skull_stripping': False, 'registration': False, 'postprocessing': False},
            1: {'normalization': True, 'skull_stripping': False, 'registration': False, 'postprocessing': False},
            2: {'normalization': False, 'skull_stripping': True, 'registration': False, 'postprocessing': False},
            3: {'normalization': False, 'skull_stripping': False, 'registration': True, 'postprocessing': False},
            4: {'normalization': True, 'skull_stripping': True, 'registration': False, 'postprocessing': False},
            5: {'normalization': True, 'skull_stripping': False, 'registration': True, 'postprocessing': False},
            6: {'normalization': False, 'skull_stripping': True, 'registration': True, 'postprocessing': False},
            7: {'normalization': True, 'skull_stripping': True, 'registration': True, 'postprocessing': False},
            8: {'normalization': True, 'skull_stripping': True, 'registration': True, 'postprocessing': True},
        }
    
    def analyze_component_contributions(self) -> Dict:
        """Analyze the contribution of each preprocessing component."""
        # Get mean performance for each experiment
        exp_performance = {}
        
        for exp_id in self.experiments.get_experiment_ids():
            exp_data = self.experiments.get_experiment(exp_id)
            if exp_data:
                dice_scores = exp_data.get_dice_scores()
                if len(dice_scores) > 0:
                    exp_performance[exp_id] = np.mean(dice_scores)
        
        if len(exp_performance) == 0:
            return {'error': 'No performance data available'}
        
        # Calculate component contributions
        baseline_score = exp_performance.get(0, 0)
        contributions = {
            'baseline_performance': baseline_score,
            'experiment_performance': exp_performance.copy()
        }
        
        # Individual component effects (isolated)
        if 1 in exp_performance:
            contributions['normalization_isolated'] = exp_performance[1] - baseline_score
        if 2 in exp_performance:
            contributions['skull_stripping_isolated'] = exp_performance[2] - baseline_score
        if 3 in exp_performance:
            contributions['registration_isolated'] = exp_performance[3] - baseline_score
        
        # Component effects in context (marginal contributions)
        if 7 in exp_performance and 6 in exp_performance:
            contributions['normalization_marginal'] = exp_performance[7] - exp_performance[6]
        if 7 in exp_performance and 5 in exp_performance:
            contributions['skull_stripping_marginal'] = exp_performance[7] - exp_performance[5]
        if 7 in exp_performance and 4 in exp_performance:
            contributions['registration_marginal'] = exp_performance[7] - exp_performance[4]
        if 8 in exp_performance and 7 in exp_performance:
            contributions['postprocessing_effect'] = exp_performance[8] - exp_performance[7]
        
        # Interaction effects
        contributions['interactions'] = self._analyze_interactions(exp_performance)
        
        return contributions
    
    def _analyze_interactions(self, exp_performance: Dict[int, float]) -> Dict:
        """Analyze interaction effects between components."""
        interactions = {}
        
        # Two-way interactions
        if all(exp_id in exp_performance for exp_id in [0, 1, 2, 4]):
            # Normalization × Skull stripping interaction
            expected_additive = exp_performance[0] + (exp_performance[1] - exp_performance[0]) + (exp_performance[2] - exp_performance[0])
            actual = exp_performance[4]
            interactions['norm_skull_interaction'] = actual - expected_additive
        
        if all(exp_id in exp_performance for exp_id in [0, 1, 3, 5]):
            # Normalization × Registration interaction
            expected_additive = exp_performance[0] + (exp_performance[1] - exp_performance[0]) + (exp_performance[3] - exp_performance[0])
            actual = exp_performance[5]
            interactions['norm_reg_interaction'] = actual - expected_additive
        
        if all(exp_id in exp_performance for exp_id in [0, 2, 3, 6]):
            # Skull stripping × Registration interaction
            expected_additive = exp_performance[0] + (exp_performance[2] - exp_performance[0]) + (exp_performance[3] - exp_performance[0])
            actual = exp_performance[6]
            interactions['skull_reg_interaction'] = actual - expected_additive
        
        return interactions
    
    def get_component_rankings(self) -> List[Tuple[str, float]]:
        """Get components ranked by their isolated contribution."""
        contributions = self.analyze_component_contributions()
        
        component_effects = []
        if 'normalization_isolated' in contributions:
            component_effects.append(('normalization', contributions['normalization_isolated']))
        if 'skull_stripping_isolated' in contributions:
            component_effects.append(('skull_stripping', contributions['skull_stripping_isolated']))
        if 'registration_isolated' in contributions:
            component_effects.append(('registration', contributions['registration_isolated']))
        if 'postprocessing_effect' in contributions:
            component_effects.append(('postprocessing', contributions['postprocessing_effect']))
        
        # Sort by effect size (descending)
        component_effects.sort(key=lambda x: x[1], reverse=True)
        
        return component_effects


class StatisticalAnalyzer:
    """Performs statistical analysis on experiment results."""
    
    def __init__(self, experiments: ExperimentCollection):
        self.experiments = experiments
    
    def perform_pairwise_tests(self, metric: str = 'dice') -> pd.DataFrame:
        """Perform pairwise statistical tests between experiments."""
        test_results = []
        exp_ids = self.experiments.get_experiment_ids()
        
        # Get data for all experiments
        exp_data = {}
        for exp_id in exp_ids:
            exp = self.experiments.get_experiment(exp_id)
            if exp:
                if metric == 'dice':
                    scores = exp.get_dice_scores()
                elif metric == 'hausdorff':
                    scores = exp.get_hausdorff_distances()
                else:
                    continue
                
                if len(scores) > 0:
                    exp_data[exp_id] = scores
        
        # Perform pairwise t-tests
        for i, exp1 in enumerate(exp_data.keys()):
            for exp2 in list(exp_data.keys())[i+1:]:
                try:
                    statistic, p_value = stats.ttest_ind(exp_data[exp1], exp_data[exp2])
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(exp_data[exp1]) - 1) * np.var(exp_data[exp1], ddof=1) + 
                                         (len(exp_data[exp2]) - 1) * np.var(exp_data[exp2], ddof=1)) / 
                                        (len(exp_data[exp1]) + len(exp_data[exp2]) - 2))
                    
                    cohens_d = (np.mean(exp_data[exp1]) - np.mean(exp_data[exp2])) / pooled_std
                    
                    test_results.append({
                        'experiment_1': exp1,
                        'experiment_2': exp2,
                        'metric': metric,
                        'statistic': statistic,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant_005': p_value < 0.05,
                        'significant_001': p_value < 0.01,
                        'effect_size': self._interpret_effect_size(abs(cohens_d))
                    })
                except Exception as e:
                    print(f"Error in t-test between {exp1} and {exp2}: {e}")
        
        return pd.DataFrame(test_results)
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return 'negligible'
        elif cohens_d < 0.5:
            return 'small'
        elif cohens_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def perform_anova(self, metric: str = 'dice') -> Dict:
        """Perform one-way ANOVA across all experiments."""
        exp_data = []
        exp_labels = []
        
        for exp_id in self.experiments.get_experiment_ids():
            exp = self.experiments.get_experiment(exp_id)
            if exp:
                if metric == 'dice':
                    scores = exp.get_dice_scores()
                elif metric == 'hausdorff':
                    scores = exp.get_hausdorff_distances()
                else:
                    continue
                
                if len(scores) > 0:
                    exp_data.append(scores)
                    exp_labels.extend([exp_id] * len(scores))
        
        if len(exp_data) < 2:
            return {'error': 'Need at least 2 experiments for ANOVA'}
        
        # Perform one-way ANOVA
        f_statistic, p_value = stats.f_oneway(*exp_data)
        
        # Calculate eta-squared (effect size)
        all_scores = np.concatenate(exp_data)
        ss_total = np.sum((all_scores - np.mean(all_scores)) ** 2)
        ss_between = sum(len(group) * (np.mean(group) - np.mean(all_scores)) ** 2 for group in exp_data)
        eta_squared = ss_between / ss_total
        
        return {
            'metric': metric,
            'f_statistic': f_statistic,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < 0.05,
            'n_experiments': len(exp_data),
            'total_observations': len(all_scores)
        }
    
    def compute_confidence_intervals(self, confidence: float = 0.95) -> pd.DataFrame:
        """Compute confidence intervals for experiment means."""
        results = []
        
        for exp_id in self.experiments.get_experiment_ids():
            exp = self.experiments.get_experiment(exp_id)
            if exp:
                for metric in ['dice', 'hausdorff']:
                    if metric == 'dice':
                        scores = exp.get_dice_scores()
                    else:
                        scores = exp.get_hausdorff_distances()
                    
                    if len(scores) > 1:
                        mean_score = np.mean(scores)
                        sem = stats.sem(scores)  # Standard error of mean
                        
                        # Calculate confidence interval
                        confidence_interval = stats.t.interval(
                            confidence, len(scores) - 1, loc=mean_score, scale=sem
                        )
                        
                        results.append({
                            'experiment_id': exp_id,
                            'experiment_name': exp.name,
                            'metric': metric,
                            'mean': mean_score,
                            'std': np.std(scores, ddof=1),
                            'n_observations': len(scores),
                            'confidence_level': confidence,
                            'ci_lower': confidence_interval[0],
                            'ci_upper': confidence_interval[1],
                            'margin_of_error': confidence_interval[1] - mean_score
                        })
        
        return pd.DataFrame(results)


class PerformanceAnalyzer:
    """Analyzes overall performance patterns and trends."""
    
    def __init__(self, experiments: ExperimentCollection):
        self.experiments = experiments
    
    def analyze_performance_trends(self) -> Dict:
        """Analyze performance trends across experiments."""
        summary_table = self.experiments.get_summary_table()
        
        if len(summary_table) == 0:
            return {'error': 'No data available'}
        
        trends = {
            'overall_performance': {
                'best_experiment': int(summary_table.loc[summary_table['mean_dice'].idxmax(), 'experiment_id']),
                'worst_experiment': int(summary_table.loc[summary_table['mean_dice'].idxmin(), 'experiment_id']),
                'performance_range': float(summary_table['mean_dice'].max() - summary_table['mean_dice'].min()),
                'mean_performance': float(summary_table['mean_dice'].mean()),
                'std_performance': float(summary_table['mean_dice'].std())
            }
        }
        
        # Analyze variability trends
        if 'std_dice' in summary_table.columns:
            trends['variability'] = {
                'most_consistent': int(summary_table.loc[summary_table['std_dice'].idxmin(), 'experiment_id']),
                'least_consistent': int(summary_table.loc[summary_table['std_dice'].idxmax(), 'experiment_id']),
                'mean_variability': float(summary_table['std_dice'].mean())
            }
        
        return trends
    
    def analyze_tissue_specific_performance(self) -> Dict:
        """Analyze performance by tissue type."""
        tissue_analysis = {}
        
        # Get all tissue labels across experiments
        all_tissues = set()
        for exp in self.experiments.experiments.values():
            all_tissues.update(exp.tissue_labels)
        
        for tissue in all_tissues:
            tissue_performance = []
            
            for exp_id in self.experiments.get_experiment_ids():
                exp = self.experiments.get_experiment(exp_id)
                if exp and tissue in exp.tissue_labels:
                    scores = exp.get_dice_scores(tissue)
                    if len(scores) > 0:
                        tissue_performance.append({
                            'experiment_id': exp_id,
                            'mean_dice': np.mean(scores),
                            'std_dice': np.std(scores),
                            'n_samples': len(scores)
                        })
            
            if tissue_performance:
                tissue_df = pd.DataFrame(tissue_performance)
                
                tissue_analysis[tissue] = {
                    'best_experiment': int(tissue_df.loc[tissue_df['mean_dice'].idxmax(), 'experiment_id']),
                    'worst_experiment': int(tissue_df.loc[tissue_df['mean_dice'].idxmin(), 'experiment_id']),
                    'performance_range': float(tissue_df['mean_dice'].max() - tissue_df['mean_dice'].min()),
                    'mean_performance': float(tissue_df['mean_dice'].mean()),
                    'experiments_tested': len(tissue_performance)
                }
        
        return tissue_analysis


class ResultsExporter:
    """Exports analysis results to various formats."""
    
    def __init__(self, experiments: ExperimentCollection):
        self.experiments = experiments
    
    def export_comprehensive_report(self, output_dir: str) -> Dict[str, str]:
        """Export a comprehensive analysis report."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # Component analysis
        component_analyzer = ComponentAnalyzer(self.experiments)
        contributions = component_analyzer.analyze_component_contributions()
        
        contrib_file = os.path.join(output_dir, 'component_contributions.json')
        with open(contrib_file, 'w') as f:
            json.dump(contributions, f, indent=2)
        exported_files['component_contributions'] = contrib_file
        
        # Statistical analysis
        stat_analyzer = StatisticalAnalyzer(self.experiments)
        
        # Pairwise tests
        pairwise_tests = stat_analyzer.perform_pairwise_tests('dice')
        pairwise_file = os.path.join(output_dir, 'pairwise_tests.csv')
        pairwise_tests.to_csv(pairwise_file, index=False)
        exported_files['pairwise_tests'] = pairwise_file
        
        # Confidence intervals
        ci_results = stat_analyzer.compute_confidence_intervals()
        ci_file = os.path.join(output_dir, 'confidence_intervals.csv')
        ci_results.to_csv(ci_file, index=False)
        exported_files['confidence_intervals'] = ci_file
        
        # Summary statistics
        summary_table = self.experiments.get_summary_table()
        summary_file = os.path.join(output_dir, 'summary_statistics.csv')
        summary_table.to_csv(summary_file, index=False)
        exported_files['summary_statistics'] = summary_file
        
        # Performance analysis
        perf_analyzer = PerformanceAnalyzer(self.experiments)
        trends = perf_analyzer.analyze_performance_trends()
        tissue_analysis = perf_analyzer.analyze_tissue_specific_performance()
        
        analysis_file = os.path.join(output_dir, 'performance_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump({
                'performance_trends': trends,
                'tissue_specific_analysis': tissue_analysis
            }, f, indent=2)
        exported_files['performance_analysis'] = analysis_file
        
        return exported_files