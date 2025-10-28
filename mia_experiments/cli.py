#!/usr/bin/env python3
"""
Command-line interface for MIA Experiments Framework.
Enables non-interactive execution of ablation studies on HPC clusters.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mia_experiments import run_ablation_study, analyze_results, load_and_visualize


def run_command(args):
    """Run ablation study."""
    print("=" * 60)
    print("STARTING ABLATION STUDY")
    print("=" * 60)
    print(f"Atlas data: {args.data_atlas}")
    print(f"Training data: {args.data_train}")
    print(f"Test data: {args.data_test}")
    print(f"Optimization: {args.optimization}")
    print(f"Study type: {args.study_type}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Run the ablation study
    results = run_ablation_study(
        data_atlas_dir=args.data_atlas,
        data_train_dir=args.data_train,
        data_test_dir=args.data_test,
        optimization_level=args.optimization,
        output_dir=args.output_dir,
        study_type=args.study_type
    )
    
    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETED")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)
    
    return 0


def analyze_command(args):
    """Analyze results."""
    print("=" * 60)
    print("ANALYZING RESULTS")
    print("=" * 60)
    
    analyze_results(args.experiment_dir, args.optimization)
    
    print(f"\nAnalysis complete! Check {args.experiment_dir}/analysis_{args.optimization}/")
    return 0


def visualize_command(args):
    """Create visualizations."""
    print("=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_files = load_and_visualize(args.experiment_dir)
    
    if plot_files:
        print("\nVisualizations created:")
        for plot_type, path in plot_files.items():
            print(f"  {plot_type}: {path}")
    else:
        print("No experiment data found")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MIA Experiments Framework - Brain Tissue Segmentation Ablation Studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ablation study with no optimization
  python -m mia_experiments.cli run \\
    --data-atlas ./data/atlas \\
    --data-train ./data/train \\
    --data-test ./data/test \\
    --optimization none

  # Run with quick optimization
  python -m mia_experiments.cli run \\
    --data-atlas ./data/atlas \\
    --data-train ./data/train \\
    --data-test ./data/test \\
    --optimization quick \\
    --study-type preprocessing

  # Analyze results
  python -m mia_experiments.cli analyze \\
    --experiment-dir ./ablation_experiments \\
    --optimization quick

  # Create visualizations
  python -m mia_experiments.cli visualize \\
    --experiment-dir ./ablation_experiments
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run ablation study')
    run_parser.add_argument('--data-atlas', required=True,
                           help='Path to atlas data directory')
    run_parser.add_argument('--data-train', required=True,
                           help='Path to training data directory')
    run_parser.add_argument('--data-test', required=True,
                           help='Path to test data directory')
    run_parser.add_argument('--optimization', 
                           choices=['none', 'None', 'quick', 'full'],
                           default='none',
                           help='Optimization level (default: none)')
    run_parser.add_argument('--study-type',
                           choices=['preprocessing', 'postprocessing', 'combined'],
                           default='preprocessing',
                           help='Type of ablation study (default: preprocessing)')
    run_parser.add_argument('--output-dir', default='./ablation_experiments',
                           help='Output directory (default: ./ablation_experiments)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument('--experiment-dir', default='./ablation_experiments',
                               help='Experiment directory (default: ./ablation_experiments)')
    analyze_parser.add_argument('--optimization', default='quick',
                               help='Optimization level used (default: quick)')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
    viz_parser.add_argument('--experiment-dir', default='./ablation_experiments',
                           help='Experiment directory (default: ./ablation_experiments)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Normalize 'None' to 'none'
    if hasattr(args, 'optimization') and args.optimization == 'None':
        args.optimization = 'none'
    
    # Execute command
    if args.command == 'run':
        return run_command(args)
    elif args.command == 'analyze':
        return analyze_command(args)
    elif args.command == 'visualize':
        return visualize_command(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())