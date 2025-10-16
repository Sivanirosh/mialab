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
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        components = ['normalization', 'skull_stripping', 'registration', 'postprocessing']
        component_mapping = {
            'normalization': [1, 4, 5, 7, 8],
            'skull_stripping': [2, 4, 6, 7, 8],
            'registration': [3, 5, 6, 7, 8],
            'postprocessing': [8]
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
        
        # Component effects
        fig = self.plot_component_effects()
        comp_path = os.path.join(output_dir, 'component_effects.png')
        fig.savefig(comp_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['component_effects'] = comp_path
        
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