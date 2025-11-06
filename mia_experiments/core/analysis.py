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
        
        # Define experiment configurations for ablation study (8 experiments with decoupled normalization and bias correction)
        self.ablation_configs = {
            0: {'normalization': False, 'bias_correction': False, 'skull_stripping': False, 'registration': False, 'postprocessing': False},
            1: {'normalization': True, 'bias_correction': False, 'skull_stripping': False, 'registration': False, 'postprocessing': False},
            2: {'normalization': False, 'bias_correction': True, 'skull_stripping': False, 'registration': False, 'postprocessing': False},
            3: {'normalization': True, 'bias_correction': True, 'skull_stripping': False, 'registration': False, 'postprocessing': False},
            4: {'normalization': False, 'bias_correction': False, 'skull_stripping': True, 'registration': False, 'postprocessing': False},
            5: {'normalization': False, 'bias_correction': False, 'skull_stripping': False, 'registration': True, 'postprocessing': False},
            6: {'normalization': True, 'bias_correction': True, 'skull_stripping': True, 'registration': True, 'postprocessing': False},
            7: {'normalization': True, 'bias_correction': True, 'skull_stripping': True, 'registration': True, 'postprocessing': True},
        }
    
    def answer_question_1_component_importance(self) -> Dict:
        """
        Question 1: What is the importance of each preprocessing component?
        
        Key experiments to analyze:
        - Baseline (exp 0) vs Single components (exp 1, 2, 3)
        - Best combination (exp 6) vs all (exp 7)
        """
        results = {}
        
        # Get performance for each experiment
        exp_performance = {}
        for exp_id in self.experiments.get_experiment_ids():
            exp = self.experiments.get_experiment(exp_id)
            if exp:
                dice_scores = exp.get_dice_scores()
                if len(dice_scores) > 0:
                    exp_performance[exp_id] = np.mean(dice_scores)
        
        if not exp_performance:
            return {'error': 'No performance data available'}
        
        baseline_dice = exp_performance.get(0, 0)
        
        # Individual contributions (decoupled normalization and bias correction)
        if 1 in exp_performance:
            results['normalization_only'] = exp_performance[1] - baseline_dice
        if 2 in exp_performance:
            results['bias_correction_only'] = exp_performance[2] - baseline_dice
        if 3 in exp_performance:
            results['normalization_bias_combined'] = exp_performance[3] - baseline_dice
        if 4 in exp_performance:
            results['skull_stripping_only'] = exp_performance[4] - baseline_dice
        if 5 in exp_performance:
            results['registration_only'] = exp_performance[5] - baseline_dice
        
        # All preprocessing
        if 6 in exp_performance:
            results['all_preprocessing_dice'] = exp_performance[6]
        
        results['baseline_dice'] = baseline_dice
        
        return results
    
    def answer_question_2_postprocessing_necessity(self) -> Dict:
        """
        Question 2: Is postprocessing necessary and when?
        
        Compare: same preprocessing WITH vs WITHOUT postprocessing
        """
        results = {
            'comparisons': [],
            'average_effect': 0,
            'beneficial_count': 0
        }
        
        # Compare experiments 0-7 (no post) vs 8-15 (with post) for combined study
        # or single comparison 0 vs 7 for preprocessing study
        max_exp_id = max(self.experiments.get_experiment_ids()) if self.experiments.get_experiment_ids() else 0
        
        if max_exp_id >= 15:  # Combined study (16 experiments: 0-7 vs 8-15)
            for exp_id in range(8):
                no_post_exp = self.experiments.get_experiment(exp_id)
                with_post_exp = self.experiments.get_experiment(exp_id + 8)
                if no_post_exp and with_post_exp:
                    no_post_dice = np.mean(no_post_exp.get_dice_scores())
                    with_post_dice = np.mean(with_post_exp.get_dice_scores())
                    delta = with_post_dice - no_post_dice
                    
                    results['comparisons'].append({
                        'experiment_id': exp_id,
                        'name': no_post_exp.name,
                        'no_postprocessing': no_post_dice,
                        'with_postprocessing': with_post_dice,
                        'difference': delta,
                        'improves': delta > 0
                    })
        elif 7 in self.experiments.get_experiment_ids():  # Preprocessing study (8 experiments: 0-7)
            for exp_id in range(8):
                no_post_exp = self.experiments.get_experiment(exp_id)
                if no_post_exp:
                    no_post_dice = np.mean(no_post_exp.get_dice_scores())
                    # For preprocessing study, only exp 7 has postprocessing
                    if exp_id == 7:
                        with_post_dice = no_post_dice  # Already has post
                    else:
                        with_post_exp = self.experiments.get_experiment(7)
                        with_post_dice = np.mean(with_post_exp.get_dice_scores()) if with_post_exp else no_post_dice
                        # Only compare the difference if we're comparing to the "all preprocessing" baseline
                        continue
                    
            # Just compare baseline vs all+post for simplicity in preprocessing study
            baseline_exp = self.experiments.get_experiment(0)
            all_post_exp = self.experiments.get_experiment(7)
            if baseline_exp and all_post_exp:
                no_post_dice = np.mean(baseline_exp.get_dice_scores())
                with_post_dice = np.mean(all_post_exp.get_dice_scores())
                delta = with_post_dice - no_post_dice
                
                results['comparisons'].append({
                    'experiment_id': 0,
                    'name': 'Baseline vs All+Post',
                    'no_postprocessing': no_post_dice,
                    'with_postprocessing': with_post_dice,
                    'difference': delta,
                    'improves': delta > 0
                })
        
        if results['comparisons']:
            results['average_effect'] = np.mean([c['difference'] for c in results['comparisons']])
            results['beneficial_count'] = sum(1 for c in results['comparisons'] if c['improves'])
        
        return results
    
    def answer_question_3_label_dependency(self) -> Dict:
        """
        Question 3: Is postprocessing label-dependent?
        
        Analyze tissue-specific performance with vs without postprocessing
        """
        results = {
            'tissue_effects': []
        }
        
        # Get all tissues
        combined_data = self.experiments.get_combined_data()
        if 'LABEL' not in combined_data.columns:
            return results
        
        tissues = sorted(combined_data['LABEL'].unique())
        
        max_exp_id = max(self.experiments.get_experiment_ids()) if self.experiments.get_experiment_ids() else 0
        
        for tissue in tissues:
            no_post_dice = []
            with_post_dice = []
            
            if max_exp_id >= 15:  # Combined study (16 experiments: 0-7 vs 8-15)
                # Collect data from experiments 0-7 (no post)
                for exp_id in range(8):
                    exp = self.experiments.get_experiment(exp_id)
                    if exp:
                        tissue_scores = exp.get_dice_scores(tissue)
                        no_post_dice.extend(tissue_scores.tolist())
                
                # Collect data from experiments 8-15 (with post)
                for exp_id in range(8, 16):
                    exp = self.experiments.get_experiment(exp_id)
                    if exp:
                        tissue_scores = exp.get_dice_scores(tissue)
                        with_post_dice.extend(tissue_scores.tolist())
            elif 7 in self.experiments.get_experiment_ids():  # Preprocessing study (8 experiments)
                # Collect data from experiments 0-6 (no post)
                for exp_id in range(7):
                    exp = self.experiments.get_experiment(exp_id)
                    if exp:
                        tissue_scores = exp.get_dice_scores(tissue)
                        no_post_dice.extend(tissue_scores.tolist())
                
                # Collect data from experiment 7 (with post)
                exp = self.experiments.get_experiment(7)
                if exp:
                    tissue_scores = exp.get_dice_scores(tissue)
                    with_post_dice.extend(tissue_scores.tolist())
            
            if no_post_dice and with_post_dice:
                delta = np.mean(with_post_dice) - np.mean(no_post_dice)
                results['tissue_effects'].append({
                    'tissue': tissue,
                    'no_post_mean': np.mean(no_post_dice),
                    'with_post_mean': np.mean(with_post_dice),
                    'difference': delta
                })
        
        return results
    
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
        
        # Individual component effects (isolated) - decoupled normalization and bias correction
        if 1 in exp_performance:
            contributions['normalization_isolated'] = exp_performance[1] - baseline_score
        if 2 in exp_performance:
            contributions['bias_correction_isolated'] = exp_performance[2] - baseline_score
        if 4 in exp_performance:
            contributions['skull_stripping_isolated'] = exp_performance[4] - baseline_score
        if 5 in exp_performance:
            contributions['registration_isolated'] = exp_performance[5] - baseline_score
        
        # Component effects in context (marginal contributions)
        if 7 in exp_performance and 6 in exp_performance:
            contributions['postprocessing_effect'] = exp_performance[7] - exp_performance[6]
        
        # Interaction effects
        contributions['interactions'] = self._analyze_interactions(exp_performance)
        
        return contributions
    
    def _analyze_interactions(self, exp_performance: Dict[int, float]) -> Dict:
        """Analyze interaction effects between components."""
        interactions = {}
        
        # Normalization × Bias correction interaction
        if all(exp_id in exp_performance for exp_id in [0, 1, 2, 3]):
            expected_additive = exp_performance[0] + (exp_performance[1] - exp_performance[0]) + (exp_performance[2] - exp_performance[0])
            actual = exp_performance[3]
            interactions['norm_bias_interaction'] = actual - expected_additive
        
        # Normalization × Skull stripping interaction (using exp 1, 4, 6)
        if all(exp_id in exp_performance for exp_id in [0, 1, 4, 6]):
            expected_additive = exp_performance[0] + (exp_performance[1] - exp_performance[0]) + (exp_performance[4] - exp_performance[0])
            actual = exp_performance[6]
            interactions['norm_skull_interaction'] = actual - expected_additive
        
        # Bias correction × Skull stripping interaction (using exp 2, 4, 6)
        if all(exp_id in exp_performance for exp_id in [0, 2, 4, 6]):
            expected_additive = exp_performance[0] + (exp_performance[2] - exp_performance[0]) + (exp_performance[4] - exp_performance[0])
            actual = exp_performance[6]
            interactions['bias_skull_interaction'] = actual - expected_additive
        
        # Normalization × Registration interaction (using exp 1, 5, 6)
        if all(exp_id in exp_performance for exp_id in [0, 1, 5, 6]):
            expected_additive = exp_performance[0] + (exp_performance[1] - exp_performance[0]) + (exp_performance[5] - exp_performance[0])
            actual = exp_performance[6]
            interactions['norm_reg_interaction'] = actual - expected_additive
        
        return interactions
    
    def get_component_rankings(self) -> List[Tuple[str, float]]:
        """Get components ranked by their isolated contribution."""
        contributions = self.analyze_component_contributions()
        
        component_effects = []
        if 'normalization_isolated' in contributions:
            component_effects.append(('normalization', contributions['normalization_isolated']))
        if 'bias_correction_isolated' in contributions:
            component_effects.append(('bias_correction', contributions['bias_correction_isolated']))
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