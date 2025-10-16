#!/usr/bin/env python3
"""
Standalone analysis script for ablation study results.

This script analyzes completed ablation experiments without re-running the pipeline.
It handles the actual CSV format with SUBJECT;LABEL;DICE;HDRFDST columns.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats
from typing import Dict, List
import glob

def load_ablation_results(base_dir: str = "./ablation_experiments", optimization_level: str = "none"):
    """Load results from all completed experiments."""
    print(f"Loading ablation results from {base_dir}...")
    
    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(base_dir, "exp_*"))
    exp_dirs.sort()
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    all_results = []
    
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        
        # Extract experiment ID
        try:
            exp_id = int(exp_name.split('_')[1])
        except:
            print(f"Warning: Could not extract experiment ID from {exp_name}")
            continue
        
        # Find results.csv files
        results_files = glob.glob(os.path.join(exp_dir, "**/results.csv"), recursive=True)
        
        if not results_files:
            print(f"Warning: No results.csv found in {exp_name}")
            continue
        
        # Use the first results file found
        results_file = results_files[0]
        print(f"Loading {exp_name}: {results_file}")
        
        try:
            # Load CSV - handle semicolon separator for first columns
            with open(results_file, 'r') as f:
                content = f.read()
            
            # Split lines and handle mixed separators
            lines = content.strip().split('\n')
            
            processed_data = []
            for line in lines:
                # Split on semicolon first, then handle comma-separated metadata
                if ';' in line:
                    parts = line.split(';')
                    if len(parts) >= 4:
                        # Core data: SUBJECT, LABEL, DICE, HDRFDST
                        subject = parts[0]
                        label = parts[1] 
                        dice = parts[2]
                        hdrfdst = parts[3]
                        
                        # Additional metadata (comma-separated)
                        if len(parts) > 4 and ',' in parts[4]:
                            metadata = parts[4].split(',')
                        else:
                            metadata = [''] * 7  # Default empty metadata
                        
                        processed_data.append({
                            'SUBJECT': subject,
                            'LABEL': label,
                            'DICE': dice,
                            'HDRFDST': hdrfdst,
                            'experiment_id': exp_id,
                            'experiment_name': exp_name,
                            'optimization_level': optimization_level
                        })
            
            if processed_data:
                df = pd.DataFrame(processed_data)
                
                # Convert DICE and HDRFDST to numeric, handling potential header row
                df = df[df['DICE'] != 'DICE']  # Remove header row if present
                df['DICE'] = pd.to_numeric(df['DICE'], errors='coerce')
                df['HDRFDST'] = pd.to_numeric(df['HDRFDST'], errors='coerce')
                df['experiment_id'] = exp_id
                
                # Remove any rows with NaN values
                df = df.dropna(subset=['DICE'])
                
                if len(df) > 0:
                    all_results.append(df)
                    print(f"  ✅ Loaded {len(df)} measurements from {exp_name}")
                else:
                    print(f"  ❌ No valid data in {exp_name}")
            else:
                print(f"  ❌ Could not parse data in {exp_name}")
                
        except Exception as e:
            print(f"  ❌ Error loading {exp_name}: {e}")
    
    if not all_results:
        print("❌ No results loaded!")
        return None
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"\n✅ Successfully loaded {len(combined_df)} total measurements from {len(all_results)} experiments")
    
    return combined_df

def analyze_experiment_performance(df: pd.DataFrame):
    """Analyze performance by experiment."""
    print("\n📊 EXPERIMENT PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Get unique experiments and labels
    experiments = sorted(df['experiment_id'].unique())
    labels = sorted(df['LABEL'].unique())
    
    print(f"Experiments: {experiments}")
    print(f"Tissue labels: {labels}")
    
    # Calculate mean performance per experiment
    exp_performance = {}
    
    for exp_id in experiments:
        exp_data = df[df['experiment_id'] == exp_id]
        exp_name = exp_data['experiment_name'].iloc[0] if len(exp_data) > 0 else f"exp_{exp_id}"
        
        # Overall mean across all tissues
        overall_dice = exp_data['DICE'].mean()
        overall_hausdorff = exp_data['HDRFDST'].mean()
        
        # Per-tissue performance
        tissue_performance = {}
        for label in labels:
            tissue_data = exp_data[exp_data['LABEL'] == label]
            if len(tissue_data) > 0:
                tissue_performance[label] = {
                    'dice_mean': tissue_data['DICE'].mean(),
                    'dice_std': tissue_data['DICE'].std(),
                    'hausdorff_mean': tissue_data['HDRFDST'].mean(),
                    'hausdorff_std': tissue_data['HDRFDST'].std(),
                    'n_subjects': len(tissue_data)
                }
        
        exp_performance[exp_id] = {
            'name': exp_name,
            'overall_dice': overall_dice,
            'overall_hausdorff': overall_hausdorff,
            'tissue_performance': tissue_performance,
            'n_total': len(exp_data)
        }
        
        print(f"Exp {exp_id}: {overall_dice:.4f} Dice, {overall_hausdorff:.2f} Hausdorff ({len(exp_data)} measurements)")
    
    return exp_performance

def create_summary_statistics(exp_performance: Dict, output_dir: str):
    """Create summary statistics table."""
    print("\n📋 Creating summary statistics...")
    
    summary_data = []
    
    # Define experiment configurations (you may need to adjust these)
    exp_configs = {
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
    
    for exp_id, perf in exp_performance.items():
        config = exp_configs.get(exp_id, {})
        
        row = {
            'experiment_id': exp_id,
            'experiment_name': perf['name'],
            'mean_dice': perf['overall_dice'],
            'mean_hausdorff': perf['overall_hausdorff'],
            'n_measurements': perf['n_total'],
            **config
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"Summary statistics saved to: {summary_file}")
    print(summary_df.to_string(index=False))
    
    return summary_df

def analyze_component_contributions(exp_performance: Dict, output_dir: str):
    """Analyze individual component contributions."""
    print("\n🔬 COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 50)
    
    if len(exp_performance) < 9:
        print(f"Warning: Only {len(exp_performance)} experiments found, need 9 for full analysis")
    
    contributions = {}
    
    # Get baseline performance (experiment 0)
    baseline_dice = exp_performance.get(0, {}).get('overall_dice', 0)
    contributions['baseline_performance'] = baseline_dice
    
    # Individual component effects (if experiments exist)
    if 1 in exp_performance:
        contributions['normalization_only'] = exp_performance[1]['overall_dice'] - baseline_dice
    if 2 in exp_performance:
        contributions['skull_stripping_only'] = exp_performance[2]['overall_dice'] - baseline_dice
    if 3 in exp_performance:
        contributions['registration_only'] = exp_performance[3]['overall_dice'] - baseline_dice
    
    # Component effects in context (if full set of experiments exists)
    if 7 in exp_performance and 6 in exp_performance:
        contributions['normalization_effect_in_context'] = exp_performance[7]['overall_dice'] - exp_performance[6]['overall_dice']
    if 7 in exp_performance and 5 in exp_performance:
        contributions['skull_stripping_effect_in_context'] = exp_performance[7]['overall_dice'] - exp_performance[5]['overall_dice']
    if 7 in exp_performance and 4 in exp_performance:
        contributions['registration_effect_in_context'] = exp_performance[7]['overall_dice'] - exp_performance[4]['overall_dice']
    if 8 in exp_performance and 7 in exp_performance:
        contributions['postprocessing_effect'] = exp_performance[8]['overall_dice'] - exp_performance[7]['overall_dice']
    
    # Save contributions
    contrib_file = os.path.join(output_dir, "component_contributions.json")
    with open(contrib_file, 'w') as f:
        json.dump(contributions, f, indent=2)
    
    print(f"Component contributions saved to: {contrib_file}")
    print("\nComponent Effects (Dice coefficient changes):")
    for component, effect in contributions.items():
        if isinstance(effect, (int, float)):
            print(f"  {component}: {effect:+.4f}")
        else:
            print(f"  {component}: {effect}")
    
    return contributions

def create_visualizations(df: pd.DataFrame, exp_performance: Dict, output_dir: str):
    """Create visualization plots."""
    print("\n📈 Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall performance comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    exp_ids = sorted(exp_performance.keys())
    exp_names = [f"{i}: {exp_performance[i]['name'].replace('exp_', '').replace('_', ' ')}" for i in exp_ids]
    exp_scores = [exp_performance[i]['overall_dice'] for i in exp_ids]
    
    bars = ax.bar(exp_names, exp_scores)
    ax.set_ylabel('Mean Dice Coefficient')
    ax.set_title('Ablation Study: Overall Performance Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Color best and worst performance
    if exp_scores:
        best_idx = exp_scores.index(max(exp_scores))
        worst_idx = exp_scores.index(min(exp_scores))
        bars[best_idx].set_color('gold')
        bars[worst_idx].set_color('lightcoral')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-tissue analysis
    labels = sorted(df['LABEL'].unique())
    n_tissues = len(labels)
    
    if n_tissues > 0:
        n_cols = min(3, n_tissues)
        n_rows = (n_tissues + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, label in enumerate(labels):
            row = i // n_cols
            col = i % n_cols
            
            if n_rows > 1:
                ax = axes[row][col]
            else:
                ax = axes[col] if n_cols > 1 else axes[0]
            
            # Get performance for this tissue across experiments
            tissue_scores = []
            tissue_exp_names = []
            
            for exp_id in exp_ids:
                tissue_perf = exp_performance[exp_id]['tissue_performance'].get(label, {})
                if 'dice_mean' in tissue_perf:
                    tissue_scores.append(tissue_perf['dice_mean'])
                    tissue_exp_names.append(str(exp_id))
            
            if tissue_scores:
                ax.bar(tissue_exp_names, tissue_scores)
                ax.set_title(f'Tissue: {label}')
                ax.set_ylabel('Mean Dice Coefficient')
                ax.set_xlabel('Experiment ID')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_tissues, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row][col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "per_tissue_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Box plots comparing preprocessing components
    create_component_boxplots(df, output_dir)
    
    print(f"Visualizations saved to: {output_dir}")

def create_component_boxplots(df: pd.DataFrame, output_dir: str):
    """Create box plots showing component effects."""
    print("Creating component effect box plots...")
    
    # Define which experiments have which components
    component_mapping = {
        'normalization': [1, 4, 5, 7, 8],
        'skull_stripping': [2, 4, 6, 7, 8], 
        'registration': [3, 5, 6, 7, 8],
        'postprocessing': [8]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    components = ['normalization', 'skull_stripping', 'registration', 'postprocessing']
    
    for i, component in enumerate(components):
        ax = axes[i]
        
        with_component = []
        without_component = []
        
        exp_ids = sorted(df['experiment_id'].unique())
        
        for exp_id in exp_ids:
            exp_data = df[df['experiment_id'] == exp_id]
            mean_dice = exp_data['DICE'].mean()
            
            if exp_id in component_mapping[component]:
                with_component.append(mean_dice)
            else:
                without_component.append(mean_dice)
        
        if with_component and without_component:
            data_to_plot = [without_component, with_component]
            labels = [f'Without {component}', f'With {component}']
            
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_title(f'Effect of {component.replace("_", " ").title()}')
            ax.set_ylabel('Mean Dice Coefficient')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "component_effects.png"), dpi=300, bbox_inches='tight')
    plt.close()

def perform_statistical_tests(df: pd.DataFrame, output_dir: str):
    """Perform statistical significance tests."""
    print("\n📊 Performing statistical tests...")
    
    # Get mean Dice per experiment
    exp_scores = {}
    for exp_id in sorted(df['experiment_id'].unique()):
        exp_data = df[df['experiment_id'] == exp_id]
        exp_scores[exp_id] = exp_data['DICE'].values
    
    # Pairwise t-tests
    test_results = []
    
    exp_ids = list(exp_scores.keys())
    for i, exp1 in enumerate(exp_ids):
        for exp2 in exp_ids[i+1:]:
            try:
                statistic, p_value = stats.ttest_ind(exp_scores[exp1], exp_scores[exp2])
                
                test_results.append({
                    'experiment_1': exp1,
                    'experiment_2': exp2,
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
            except Exception as e:
                print(f"Error in t-test between {exp1} and {exp2}: {e}")
    
    if test_results:
        test_df = pd.DataFrame(test_results)
        test_file = os.path.join(output_dir, "statistical_tests.csv")
        test_df.to_csv(test_file, index=False)
        
        print(f"Statistical test results saved to: {test_file}")
        
        # Show significant results
        significant_tests = test_df[test_df['significant']]
        if len(significant_tests) > 0:
            print(f"\nSignificant differences (p < 0.05):")
            for _, row in significant_tests.iterrows():
                print(f"  Exp {row['experiment_1']} vs Exp {row['experiment_2']}: p = {row['p_value']:.4f}")
        else:
            print("\nNo statistically significant differences found")

def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze completed ablation study results')
    parser.add_argument('--experiment-dir', type=str, default='./ablation_experiments',
                       help='Directory containing completed experiments')
    parser.add_argument('--optimization-level', type=str, default='none',
                       help='Optimization level used in experiments')
    
    args = parser.parse_args()
    
    print("🧠 ABLATION STUDY ANALYSIS")
    print("=" * 60)
    
    # Load results
    df = load_ablation_results(args.experiment_dir, args.optimization_level)
    
    if df is None:
        print("Failed to load results. Check experiment directory and file format.")
        return
    
    # Create analysis output directory
    analysis_dir = os.path.join(args.experiment_dir, f"analysis_{args.optimization_level}")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Perform analysis
    exp_performance = analyze_experiment_performance(df)
    summary_df = create_summary_statistics(exp_performance, analysis_dir)
    contributions = analyze_component_contributions(exp_performance, analysis_dir)
    create_visualizations(df, exp_performance, analysis_dir)
    perform_statistical_tests(df, analysis_dir)
    
    # Save combined results in corrected format
    combined_file = os.path.join(args.experiment_dir, f"combined_results_{args.optimization_level}.csv")
    df.to_csv(combined_file, index=False)
    
    print("-" * 60)
    print(f"\nANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Results saved to: {analysis_dir}")
    print(f"Key files:")
    print(f"  {analysis_dir}/overall_performance.png")
    print(f"  {analysis_dir}/component_effects.png") 
    print(f"  {analysis_dir}/summary_statistics.csv")
    print(f"  {analysis_dir}/component_contributions.json")

if __name__ == "__main__":
    main()