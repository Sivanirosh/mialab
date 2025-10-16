#!/usr/bin/env python3
"""
Main command-line interface for MIA Experiments Framework.

This provides the primary entry points for running ablation studies, analyzing results,
and managing experiments.
"""

import argparse
import sys
import os
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mia_experiments import (
    run_ablation_study, analyze_results, load_and_visualize,
    OptimizationLevel, AblationStudyConfigurator, ConfigurationValidator
)


def create_main_parser() -> argparse.ArgumentParser:
    """Create main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description='MIA Experiments Framework - Brain Tissue Segmentation Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete ablation study with quick optimization
  mia-experiments run --data-atlas ../data/atlas --data-train ../data/train --data-test ../data/test

  # Run with full hyperparameter optimization
  mia-experiments run --data-atlas ../data/atlas --data-train ../data/train --data-test ../data/test --optimization full

  # Analyze existing results
  mia-experiments analyze --experiment-dir ./ablation_experiments --optimization quick

  # Create visualizations only
  mia-experiments visualize --experiment-dir ./ablation_experiments

  # Show experiment plan
  mia-experiments plan
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run ablation study')
    run_parser.add_argument('--data-atlas', type=str, required=True,
                           help='Atlas data directory')
    run_parser.add_argument('--data-train', type=str, required=True,
                           help='Training data directory')
    run_parser.add_argument('--data-test', type=str, required=True,
                           help='Test data directory')
    run_parser.add_argument('--optimization', type=str, default='quick',
                           choices=['none', 'quick', 'full'],
                           help='Random Forest optimization level')
    run_parser.add_argument('--output-dir', type=str, default='./ablation_experiments',
                           help='Output directory for results')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze experiment results')
    analyze_parser.add_argument('--experiment-dir', type=str, required=True,
                               help='Directory containing experiment results')
    analyze_parser.add_argument('--optimization', type=str, default='quick',
                               choices=['none', 'quick', 'full'],
                               help='Optimization level used in experiments')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
    viz_parser.add_argument('--experiment-dir', type=str, required=True,
                           help='Directory containing experiment results')
    viz_parser.add_argument('--output-dir', type=str, default=None,
                           help='Output directory for plots')
    
    # Plan command
    plan_parser = subparsers.add_parser('plan', help='Show experiment plan')
    plan_parser.add_argument('--optimization', type=str, default='quick',
                            choices=['none', 'quick', 'full'],
                            help='Optimization level to show timing for')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data directories')
    validate_parser.add_argument('--data-atlas', type=str, required=True,
                                help='Atlas data directory')
    validate_parser.add_argument('--data-train', type=str, required=True,
                                help='Training data directory')
    validate_parser.add_argument('--data-test', type=str, required=True,
                                help='Test data directory')
    
    return parser


def cmd_run(args):
    """Run ablation study command."""
    print("MIA Experiments Framework - Ablation Study")
    print("=" * 50)
    
    # Validate paths
    missing_paths = ConfigurationValidator.validate_paths(
        args.data_atlas, args.data_train, args.data_test
    )
    
    if missing_paths:
        print("Missing data directories:")
        for path in missing_paths:
            print(f"  - {path}")
        return 1
    
    print(f"Data directories validated")
    print(f"Optimization level: {args.optimization}")
    
    # Show experiment plan
    show_experiment_plan(OptimizationLevel(args.optimization))
    
    # Ask for confirmation
    response = input("\nProceed with ablation study? (y/n): ").lower().strip()
    if response != 'y':
        print("Ablation study cancelled.")
        return 0
    
    # Run the study
    try:
        results = run_ablation_study(
            args.data_atlas, 
            args.data_train, 
            args.data_test,
            args.optimization,
            args.output_dir
        )
        
        completed = sum(1 for r in results.values() if r.get('success', False))
        total = len(results)
        
        print(f"\nAblation study completed!")
        print(f"Successful experiments: {completed}/{total}")
        print(f"Results directory: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error during ablation study: {e}")
        return 1


def cmd_analyze(args):
    """Analyze results command."""
    print("Analyzing experiment results...")
    
    if not os.path.exists(args.experiment_dir):
        print(f"Experiment directory not found: {args.experiment_dir}")
        return 1
    
    try:
        analyze_results(args.experiment_dir, args.optimization)
        print(f"Analysis complete! Check {args.experiment_dir}/analysis_{args.optimization}/")
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1


def cmd_visualize(args):
    """Create visualizations command."""
    print("Creating visualizations...")
    
    if not os.path.exists(args.experiment_dir):
        print(f"Experiment directory not found: {args.experiment_dir}")
        return 1
    
    try:
        plot_files = load_and_visualize(args.experiment_dir, args.output_dir)
        
        if plot_files:
            print(f"Visualizations created:")
            for plot_type, path in plot_files.items():
                print(f"  - {plot_type}: {path}")
        else:
            print("No experiment data found for visualization")
        
        return 0
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return 1


def cmd_plan(args):
    """Show experiment plan command."""
    show_experiment_plan(OptimizationLevel(args.optimization))
    return 0


def cmd_validate(args):
    """Validate data directories command."""
    print("Validating data directories...")
    
    missing_paths = ConfigurationValidator.validate_paths(
        args.data_atlas, args.data_train, args.data_test
    )
    
    if missing_paths:
        print("Missing directories:")
        for path in missing_paths:
            print(f"  ❌ {path}")
        return 1
    else:
        print("All data directories found:")
        print(f"  ✅ Atlas: {args.data_atlas}")
        print(f"  ✅ Training: {args.data_train}")
        print(f"  ✅ Test: {args.data_test}")
        return 0


def show_experiment_plan(optimization_level: OptimizationLevel):
    """Show the ablation study experiment plan."""
    print("\nEXPERIMENT PLAN:")
    print("-" * 40)
    
    experiment_summary = AblationStudyConfigurator.get_experiment_summary()
    for exp_id, description in experiment_summary.items():
        print(f"  {exp_id}: {description}")
    
    print(f"\nRANDOM FOREST OPTIMIZATION:")
    print(f"  Level: {optimization_level.value}")
    
    if optimization_level == OptimizationLevel.NONE:
        print("  Parameters: sklearn defaults")
        print("  Time: No optimization")
    elif optimization_level == OptimizationLevel.QUICK:
        print("  Parameters: Small grid search (~48 combinations)")
        print("  Time: ~30 minutes")
    elif optimization_level == OptimizationLevel.FULL:
        print("  Parameters: Large grid search (~864 combinations)")  
        print("  Time: ~3 hours")
    
    estimated_time = ConfigurationValidator.estimate_runtime(optimization_level)
    print(f"\nESTIMATED TOTAL RUNTIME: {estimated_time}")


def main():
    """Main entry point."""
    parser = create_main_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
    
    args = parser.parse_args()
    
    if args.command == 'run':
        return cmd_run(args)
    elif args.command == 'analyze':
        return cmd_analyze(args)
    elif args.command == 'visualize':
        return cmd_visualize(args)
    elif args.command == 'plan':
        return cmd_plan(args)
    elif args.command == 'validate':
        return cmd_validate(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())