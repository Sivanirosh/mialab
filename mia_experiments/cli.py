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

from mia_experiments import run_ablation_study, analyze_results, load_and_visualize, OptimizationLevel


def run_command(args):
    """Run ablation study."""
    import os
    import time
    
    print("=" * 60)
    print("STARTING ABLATION STUDY")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Atlas data: {args.data_atlas}")
    print(f"Training data: {args.data_train}")
    print(f"Test data: {args.data_test}")
    print(f"Optimization: {args.optimization}")
    print(f"Study type: {args.study_type}")
    print(f"Output directory: {args.output_dir}")
    
    # Check data directories exist
    if args.verbose:
        print("\nChecking data directories...")
        for name, path in [('Atlas', args.data_atlas), ('Training', args.data_train), ('Test', args.data_test)]:
            if os.path.exists(path):
                print(f"  ✓ {name}: {path} (exists)")
            else:
                print(f"  ✗ {name}: {path} (NOT FOUND!)")
                return 1
    
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run the ablation study
        results = run_ablation_study(
            data_atlas_dir=args.data_atlas,
            data_train_dir=args.data_train,
            data_test_dir=args.data_test,
            optimization_level=args.optimization,
            output_dir=args.output_dir,
            study_type=args.study_type
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("ABLATION STUDY COMPLETED")
        print(f"Results saved to: {args.output_dir}")
        print(f"Total time: {elapsed_time/60:.1f} minutes ({elapsed_time:.1f} seconds)")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\nERROR: Ablation study failed after {elapsed_time:.1f} seconds")
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        return 1


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


def report_command(args):
    """Create comprehensive analysis report."""
    print("=" * 60)
    print("CREATING COMPREHENSIVE REPORT")
    print("=" * 60)
    
    try:
        # Import the unified analysis script functionality
        import os
        import sys
        script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, script_path)
        
        from analyze_ablation import (
            print_header, question_1_analysis, question_2_analysis, question_3_analysis,
            create_visualizations, export_analysis
        )
        from mia_experiments.core.data import DataLoader, ExperimentCollection
        from mia_experiments.core.analysis import ComponentAnalyzer
        
        # Load experiments
        print("\nLoading experiment data...")
        experiments = DataLoader.load_ablation_experiments(args.experiment_dir)
        
        if not experiments:
            print("Error: No experiment data found")
            return 1
        
        experiment_collection = ExperimentCollection(experiments)
        print(f"Loaded {len(experiments)} experiments")
        
        # Run comprehensive analysis
        analyzer = ComponentAnalyzer(experiment_collection)
        
        # Answer questions
        question_1_analysis(analyzer)
        question_2_analysis(analyzer)
        question_3_analysis(analyzer)
        
        # Create visualizations and export reports
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(args.experiment_dir, "report_comprehensive")
        
        os.makedirs(output_dir, exist_ok=True)
        
        create_visualizations(experiment_collection, output_dir)
        export_analysis(experiment_collection, output_dir)
        
        print_header("REPORT COMPLETE")
        print(f"Results saved to: {output_dir}/")
        
        return 0
        
    except ImportError:
        # Fallback: use analyze_ablation.py as subprocess
        import subprocess
        
        cmd = [sys.executable, os.path.join(script_path, "analyze_ablation.py"), 
               args.experiment_dir, "--mode", "full"]
        if args.output_dir:
            cmd.extend(["--output", args.output_dir])
        
        return subprocess.call(cmd)


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

  # Create comprehensive report
  python -m mia_experiments.cli report \\
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
    run_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Enable verbose output')
    
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
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Create comprehensive analysis report')
    report_parser.add_argument('--experiment-dir', required=True,
                              help='Experiment directory')
    report_parser.add_argument('--output-dir', 
                              help='Output directory for report (default: report_comprehensive)')
    
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
    elif args.command == 'report':
        return report_command(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())