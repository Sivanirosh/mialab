"""Visualization module for experiment results and analysis.

This module provides plotting functions for ablation studies, component analysis,
and statistical comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from matplotlib.patches import Rectangle

from .data import ExperimentCollection, ExperimentData
from .analysis import ComponentAnalyzer, StatisticalAnalyzer


class PlotStyle:
    """Consistent plotting style configuration."""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """Setup matplotlib and seaborn styles."""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Custom color palette for experiments
        self.experiment_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Performance colors
        self.performance_colors = {
            'best': 'gold',
            'worst': 'lightcoral',
            'significant': 'darkgreen',
            'not_significant': 'lightgray'
        }
    
    def get_experiment_color(self, exp_id: int) -> str:
        """Get color for specific experiment."""
        return self.experiment_colors[exp_id % len(self.experiment_colors)]


class ExperimentVisualizer:
    """Main visualization class for experiment results."""
    
    def __init__(self, experiments: ExperimentCollection):
        self.experiments = experiments
        self.style = PlotStyle()
    
    def plot_overall_performance(self, save_path: str = None, 
                               metric: str = 'dice', 
                               title_suffix: str = '') -> plt.Figure:
        """Plot overall performance comparison across experiments."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        exp_ids = self.experiments.get_experiment_ids()
        exp_names = []
        exp_scores = []
        
        for exp_id in exp_ids:
            exp = self.experiments.get_experiment(exp_id)
            if exp:
                if metric == 'dice':
                    scores = exp.get_dice_scores()
                    ylabel = 'Mean Dice Coefficient'
                elif metric == 'hausdorff':
                    scores = exp.get_hausdorff_distances()
                    ylabel = 'Mean Hausdorff Distance (mm)'
                else:
                    continue
                
                if len(scores) > 0:
                    exp_names.append(f"{exp_id}: {exp.name.replace('exp_', '').replace('_', ' ')}")
                    exp_scores.append(np.mean(scores))
        
        if not exp_scores:
            return fig
        
        # Create bar plot
        bars = ax.bar(exp_names, exp_scores, 
                     color=[self.style.get_experiment_color(i) for i in range(len(exp_scores))])
        
        # Highlight best and worst performance
        if len(exp_scores) > 1:
            best_idx = exp_scores.index(max(exp_scores))
            worst_idx = exp_scores.index(min(exp_scores))
            bars[best_idx].set_color(self.style.performance_colors['best'])
            bars[worst_idx].set_color(self.style.performance_colors['worst'])
            
            # Add performance annotations
            ax.annotate('Best', xy=(best_idx, exp_scores[best_idx]), 
                       xytext=(best_idx, exp_scores[best_idx] + 0.02),
                       ha='center', fontweight='bold', color='darkgoldenrod')
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Ablation Study: Overall Performance Comparison{title_suffix}', 
                    fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, exp_scores)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_component_effects(self, save_path: str = None,
                              title_suffix: str = '') -> plt.Figure:
        """Plot the effects of individual preprocessing components."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        components = ['normalization', 'bias_correction', 'skull_stripping', 'registration', 'postprocessing']
        component_mapping = {
            'normalization': [1, 3, 6, 7],  # Experiments with normalization
            'bias_correction': [2, 3, 6, 7],  # Experiments with bias correction
            'skull_stripping': [4, 6, 7],  # Experiments with skull stripping
            'registration': [5, 6, 7],  # Experiments with registration
            'postprocessing': [7]  # Experiments with postprocessing
        }
        
        combined_data = self.experiments.get_combined_data()
        
        for i, component in enumerate(components):
            ax = axes[i]
            
            with_component = []
            without_component = []
            
            for exp_id in self.experiments.get_experiment_ids():
                exp_data = combined_data[combined_data['experiment_id'] == exp_id]
                
                if len(exp_data) > 0 and 'DICE' in exp_data.columns:
                    mean_dice = exp_data['DICE'].mean()
                    
                    if exp_id in component_mapping[component]:
                        with_component.append(mean_dice)
                    else:
                        without_component.append(mean_dice)
            
            if with_component and without_component:
                data_to_plot = [without_component, with_component]
                labels = [f'Without {component.replace("_", " ")}', 
                         f'With {component.replace("_", " ")}']
                
                box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # Color the boxes
                box_plot['boxes'][0].set_facecolor('lightcoral')
                box_plot['boxes'][1].set_facecolor('lightblue')
                
                ax.set_title(f'Effect of {component.replace("_", " ").title()}', fontweight='bold')
                ax.set_ylabel('Mean Dice Coefficient')
                ax.grid(True, alpha=0.3)
                
                # Add statistical annotation
                if len(with_component) > 1 and len(without_component) > 1:
                    from scipy import stats
                    _, p_value = stats.ttest_ind(without_component, with_component)
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    ax.text(1.5, ax.get_ylim()[1] * 0.95, f'p {significance}', 
                           ha='center', fontweight='bold')
        
        plt.suptitle(f'Component Effects Analysis{title_suffix}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_tissue_specific_analysis(self, save_path: str = None) -> plt.Figure:
        """Plot performance analysis by tissue type."""
        combined_data = self.experiments.get_combined_data()
        
        if 'LABEL' not in combined_data.columns:
            # Create empty figure if no tissue-specific data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No tissue-specific data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        tissues = sorted(combined_data['LABEL'].unique())
        n_tissues = len(tissues)
        
        if n_tissues == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No tissue labels found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Calculate subplot grid
        n_cols = min(3, n_tissues)
        n_rows = (n_tissues + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, tissue in enumerate(tissues):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows > 1:
                ax = axes[row][col]
            else:
                ax = axes[col] if n_cols > 1 else axes[0]
            
            # Get performance for this tissue across experiments
            tissue_data = combined_data[combined_data['LABEL'] == tissue]
            exp_means = []
            exp_ids = []
            
            for exp_id in sorted(tissue_data['experiment_id'].unique()):
                exp_tissue_data = tissue_data[tissue_data['experiment_id'] == exp_id]
                if len(exp_tissue_data) > 0 and 'DICE' in exp_tissue_data.columns:
                    mean_dice = exp_tissue_data['DICE'].mean()
                    exp_means.append(mean_dice)
                    exp_ids.append(str(exp_id))
            
            if exp_means:
                bars = ax.bar(exp_ids, exp_means, 
                            color=[self.style.get_experiment_color(int(eid)) for eid in exp_ids])
                
                # Highlight best performance for this tissue
                if len(exp_means) > 1:
                    best_idx = exp_means.index(max(exp_means))
                    bars[best_idx].set_color(self.style.performance_colors['best'])
                
                ax.set_title(f'Tissue: {tissue}', fontweight='bold')
                ax.set_ylabel('Mean Dice Coefficient')
                ax.set_xlabel('Experiment ID')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, score in zip(bars, exp_means):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Hide unused subplots
        for i in range(n_tissues, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row][col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_statistical_significance(self, save_path: str = None) -> plt.Figure:
        """Plot statistical significance matrix between experiments."""
        stat_analyzer = StatisticalAnalyzer(self.experiments)
        test_results = stat_analyzer.perform_pairwise_tests()
        
        if len(test_results) == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No statistical test results available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Create significance matrix
        exp_ids = sorted(self.experiments.get_experiment_ids())
        n_exp = len(exp_ids)
        
        sig_matrix = np.ones((n_exp, n_exp))  # Initialize with 1s (diagonal)
        p_matrix = np.zeros((n_exp, n_exp))
        
        for _, row in test_results.iterrows():
            exp1_idx = exp_ids.index(row['experiment_1'])
            exp2_idx = exp_ids.index(row['experiment_2'])
            
            is_sig = row['significant_005']
            p_val = row['p_value']
            
            sig_matrix[exp1_idx, exp2_idx] = 0 if is_sig else 0.5
            sig_matrix[exp2_idx, exp1_idx] = 0 if is_sig else 0.5
            
            p_matrix[exp1_idx, exp2_idx] = p_val
            p_matrix[exp2_idx, exp1_idx] = p_val
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(sig_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        
        # Add experiment labels
        ax.set_xticks(range(n_exp))
        ax.set_yticks(range(n_exp))
        ax.set_xticklabels([f'Exp {exp_id}' for exp_id in exp_ids])
        ax.set_yticklabels([f'Exp {exp_id}' for exp_id in exp_ids])
        
        # Add text annotations
        for i in range(n_exp):
            for j in range(n_exp):
                if i != j:
                    p_val = p_matrix[i, j]
                    if p_val < 0.001:
                        text = '***'
                    elif p_val < 0.01:
                        text = '**'
                    elif p_val < 0.05:
                        text = '*'
                    else:
                        text = 'ns'
                    
                    ax.text(j, i, text, ha='center', va='center', 
                           color='white' if sig_matrix[i, j] < 0.3 else 'black',
                           fontweight='bold')
        
        ax.set_title('Statistical Significance Matrix\n(* p<0.05, ** p<0.01, *** p<0.001)', 
                    fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Significance Level')
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Significant', 'Not Significant', 'Same Experiment'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_component_contributions(self, save_path: str = None) -> plt.Figure:
        """Plot component contribution analysis."""
        component_analyzer = ComponentAnalyzer(self.experiments)
        contributions = component_analyzer.analyze_component_contributions()
        
        if 'error' in contributions:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, contributions['error'], 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Extract component effects
        isolated_effects = []
        marginal_effects = []
        component_names = []
        
        for key, value in contributions.items():
            if key.endswith('_isolated') and isinstance(value, (int, float)):
                component_name = key.replace('_isolated', '').replace('_', ' ').title()
                isolated_effects.append(value)
                component_names.append(component_name)
            elif key.endswith('_marginal') and isinstance(value, (int, float)):
                marginal_effects.append(value)
        
        if len(isolated_effects) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No component contribution data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Create subplot for isolated and marginal effects
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Isolated effects
        colors = ['green' if x > 0 else 'red' for x in isolated_effects]
        bars1 = ax1.barh(component_names, isolated_effects, color=colors, alpha=0.7)
        ax1.set_xlabel('Dice Coefficient Change')
        ax1.set_title('Isolated Component Effects\n(vs Baseline)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, effect in zip(bars1, isolated_effects):
            ax1.text(bar.get_width() + (0.002 if effect > 0 else -0.002), 
                    bar.get_y() + bar.get_height()/2,
                    f'{effect:+.4f}', ha='left' if effect > 0 else 'right', 
                    va='center', fontweight='bold')
        
        # Marginal effects (if available)
        if len(marginal_effects) == len(component_names):
            colors2 = ['green' if x > 0 else 'red' for x in marginal_effects]
            bars2 = ax2.barh(component_names, marginal_effects, color=colors2, alpha=0.7)
            ax2.set_xlabel('Dice Coefficient Change')
            ax2.set_title('Marginal Component Effects\n(in context)', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels
            for bar, effect in zip(bars2, marginal_effects):
                ax2.text(bar.get_width() + (0.002 if effect > 0 else -0.002), 
                        bar.get_y() + bar.get_height()/2,
                        f'{effect:+.4f}', ha='left' if effect > 0 else 'right', 
                        va='center', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Marginal effects not available\n(need full ablation study)', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Marginal Component Effects', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_component_importance(self, save_path: str = None) -> plt.Figure:
        """Create horizontal bar chart for component importance (presentation quality)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Key experiments for component importance
        exp_map = {exp.experiment_id: exp for exp in self.experiments.experiments.values()}
        
        # Order: Baseline → add each component → best combo
        configs = []
        config_labels = [
            'Baseline\n(no preprocessing)',
            '+ Normalization',
            '+ Bias Correction',
            '+ Normalization + Bias',
            '+ Skull Stripping',
            '+ Registration',
            'All Preprocessing',
        ]
        config_ids = [0, 1, 2, 3, 4, 5, 6]
        
        for exp_id, label in zip(config_ids, config_labels):
            if exp_id in exp_map:
                dice = np.mean(exp_map[exp_id].get_dice_scores())
                configs.append((label, dice))
        
        if not configs:
            return fig
        
        config_names = [c[0] for c in configs]
        config_values = [c[1] for c in configs]
        
        # Colors: red for harmful, green for helpful
        colors = []
        baseline = configs[0][1]
        for val in config_values:
            if val >= baseline + 0.01:
                colors.append('#2E7D32')  # Dark green
            elif val >= baseline:
                colors.append('#8BC34A')  # Light green
            elif val >= baseline - 0.01:
                colors.append('#FFA726')  # Orange
            else:
                colors.append('#D32F2F')  # Red
        
        bars = ax.barh(range(len(configs)), config_values, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, config_values)):
            width = bar.get_width()
            delta = val - baseline
            label = f'{val:.3f}'
            if i > 0:
                label += f' ({delta:+.3f})'
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                    label, ha='left', va='center', fontweight='bold', fontsize=10)
        
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(config_names, fontsize=11)
        ax.set_xlabel('Mean Dice Coefficient', fontsize=12, fontweight='bold')
        ax.set_title('Preprocessing Component Importance', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add vertical line at baseline
        ax.axvline(x=baseline, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(baseline + 0.005, len(configs) - 0.5, 'Baseline', rotation=90, 
                va='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_postprocessing_comparison(self, save_path: str = None) -> plt.Figure:
        """Create postprocessing necessity comparison."""
        exp_map = {exp.experiment_id: exp for exp in self.experiments.experiments.values()}
        
        # Compare 8 configurations with vs without postprocessing (for combined study)
        # Or compare against experiment 7 (which has postprocessing)
        comparisons = []
        labels = []
        max_exp_id = max(exp_map.keys()) if exp_map else 0
        
        if max_exp_id >= 15:  # Combined study (16 experiments: 0-7 vs 8-15)
            for exp_id in range(8):
                if exp_id in exp_map and (exp_id + 8) in exp_map:
                    no_post = np.mean(exp_map[exp_id].get_dice_scores())
                    with_post = np.mean(exp_map[exp_id + 8].get_dice_scores())
                    delta = with_post - no_post
                    comparisons.append(delta)
                    
                    # Short labels
                    name = exp_map[exp_id].name.replace('exp_', '').replace('_', ' ')
                    labels.append(name)
        elif 7 in exp_map:  # Preprocessing study: compare 7 (with post) vs baseline
            baseline = np.mean(exp_map[0].get_dice_scores()) if 0 in exp_map else 0
            with_post = np.mean(exp_map[7].get_dice_scores())
            delta = with_post - baseline
            comparisons.append(delta)
            labels.append("Overall effect")
        
        if not comparisons:
            # Fallback if not combined ablation study
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Postprocessing comparison requires combined ablation study\n(16 experiments: 0-7 vs 8-15)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        
        avg_effect = np.mean(comparisons)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Individual effects
        bar_colors = ['#D32F2F' if x < 0 else '#2E7D32' for x in comparisons]
        bars1 = ax1.barh(range(len(comparisons)), comparisons, color=bar_colors, edgecolor='black', linewidth=1)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax1.set_yticks(range(len(comparisons)))
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_xlabel('Δ Dice (With Post - No Post)', fontsize=12, fontweight='bold')
        ax1.set_title('Postprocessing Impact by Configuration', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, val in zip(bars1, comparisons):
            width = bar.get_width()
            ax1.text(width - 0.001 if width < 0 else width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', ha='right' if width < 0 else 'left', va='center', fontsize=9, fontweight='bold')
        
        # Right: Average effect
        avg_color = '#D32F2F' if avg_effect < 0 else '#2E7D32'
        ax2.bar([0], [avg_effect], color=avg_color, width=0.5, edgecolor='black', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Average'], fontsize=12, fontweight='bold')
        ax2.set_ylabel('Δ Dice', fontsize=12, fontweight='bold')
        ax2.set_title(f'Overall Postprocessing Effect\n{avg_effect:+.3f}', 
                     fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value label
        ax2.text(0, avg_effect - 0.002 if avg_effect < 0 else avg_effect + 0.002, f'{avg_effect:.3f}', 
                ha='center', va='top' if avg_effect < 0 else 'bottom', fontsize=14, fontweight='bold')
        
        # Add conclusion text
        if avg_effect < -0.005:
            conclusion = "✗ HARMFUL"
        elif avg_effect < 0.005:
            conclusion = "~ NEUTRAL"
        else:
            conclusion = "✓ BENEFICIAL"
        ax2.text(0, -0.015 if avg_effect < 0 else 0.01, conclusion, ha='center', va='center', 
                fontsize=14, fontweight='bold', color='#D32F2F' if avg_effect < -0.005 else '#FFA726')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_tissue_heatmap(self, save_path: str = None) -> plt.Figure:
        """Create tissue-specific performance heatmap."""
        combined_data = self.experiments.get_combined_data()
        
        if 'LABEL' not in combined_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No tissue-specific data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        
        tissues = ['WhiteMatter', 'GreyMatter', 'Thalamus', 'Hippocampus', 'Amygdala']
        
        # Key experiments (check if combined study with 16 experiments or preprocessing with 8)
        max_exp_id = max(self.experiments.get_experiment_ids()) if self.experiments.get_experiment_ids() else 0
        if max_exp_id >= 15:  # Combined study (16 experiments)
            key_experiments = [
                (0, 'Baseline'),
                (5, 'Registration'),
                (6, 'All Pre'),
                (14, 'All Pre + Post')
            ]
        else:  # Preprocessing study (8 experiments)
            key_experiments = [
                (0, 'Baseline'),
                (5, 'Registration'),
                (6, 'All Pre'),
                (7, 'All Pre + Post')
            ]
        
        # Build heatmap data
        heatmap_data = []
        for exp_id, exp_name in key_experiments:
            exp_data = combined_data[combined_data['experiment_id'] == exp_id]
            if len(exp_data) > 0:
                row = [exp_name]
                for tissue in tissues:
                    tissue_data = exp_data[exp_data['LABEL'] == tissue]
                    if len(tissue_data) > 0 and 'DICE' in tissue_data.columns:
                        row.append(tissue_data['DICE'].mean())
                    else:
                        row.append(0.0)
                heatmap_data.append(row)
        
        if not heatmap_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No heatmap data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        
        # Extract values for plotting
        config_names = [row[0] for row in heatmap_data]
        values = np.array([row[1:] for row in heatmap_data])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap
        im = ax.imshow(values, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=0.85)
        
        # Set ticks and labels
        ax.set_xticks(range(len(tissues)))
        ax.set_yticks(range(len(config_names)))
        ax.set_xticklabels(tissues, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(config_names, fontsize=11)
        
        # Add text annotations
        for i in range(len(config_names)):
            for j in range(len(tissues)):
                text = ax.text(j, i, f'{values[i, j]:.3f}', ha='center', va='center', 
                              color='black', fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Dice Coefficient', fontsize=12, fontweight='bold')
        
        ax.set_title('Tissue-Specific Performance Across Key Configurations', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_report(self, output_dir: str) -> Dict[str, str]:
        """Create a comprehensive visual report."""
        os.makedirs(output_dir, exist_ok=True)
        
        plot_files = {}
        
        # Overall performance
        fig = self.plot_overall_performance()
        overall_path = os.path.join(output_dir, 'overall_performance.png')
        fig.savefig(overall_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['overall_performance'] = overall_path
        
        # Component importance (presentation quality)
        fig = self.plot_component_importance()
        comp_imp_path = os.path.join(output_dir, 'component_importance.png')
        fig.savefig(comp_imp_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['component_importance'] = comp_imp_path
        
        # Component effects
        fig = self.plot_component_effects()
        comp_path = os.path.join(output_dir, 'component_effects.png')
        fig.savefig(comp_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['component_effects'] = comp_path
        
        # Postprocessing comparison
        fig = self.plot_postprocessing_comparison()
        post_path = os.path.join(output_dir, 'postprocessing_necessity.png')
        fig.savefig(post_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['postprocessing_necessity'] = post_path
        
        # Tissue heatmap (presentation quality)
        fig = self.plot_tissue_heatmap()
        tissue_hm_path = os.path.join(output_dir, 'tissue_heatmap.png')
        fig.savefig(tissue_hm_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['tissue_heatmap'] = tissue_hm_path
        
        # Tissue-specific analysis
        fig = self.plot_tissue_specific_analysis()
        tissue_path = os.path.join(output_dir, 'tissue_analysis.png')
        fig.savefig(tissue_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['tissue_analysis'] = tissue_path
        
        # Statistical significance
        fig = self.plot_statistical_significance()
        stat_path = os.path.join(output_dir, 'statistical_significance.png')
        fig.savefig(stat_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['statistical_significance'] = stat_path
        
        # Component contributions
        fig = self.plot_component_contributions()
        contrib_path = os.path.join(output_dir, 'component_contributions.png')
        fig.savefig(contrib_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['component_contributions'] = contrib_path
        
        return plot_files