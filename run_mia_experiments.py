#!/usr/bin/env python3
"""
Simple runner script for MIA Experiments Framework.

This provides an easy-to-use interface for running ablation studies.
"""

import os
import sys

# Add the package to Python path
package_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, package_dir)

def main():
    """Main entry point that provides an interactive interface."""
    print("=" * 50)
    print("MIA EXPERIMENTS FRAMEWORK")
    print("Brain Tissue Segmentation Ablation Studies")
    print("=" * 50)
    
    # Check if data directories exist
    default_data_paths = {
        'atlas': './data/atlas',
        'train': './data/train', 
        'test': './data/test'
    }
    
    print("Checking default data directories...")
    data_available = True
    for name, path in default_data_paths.items():
        if os.path.exists(path):
            print(f"  {name}: {path}")
        else:
            print(f"  {name}: {path} (not found)")
            data_available = False
    
    if not data_available:
        print("\nMissing data directories. Please either:")
        print("1. Place your data in the expected locations above, or")
        print("2. Use the command-line interface with custom paths:")
        print("   python -m mia_experiments.cli run --data-atlas /path/to/atlas --data-train /path/to/train --data-test /path/to/test")
        return
    
    print("\nAvailable commands:")
    print("1. Run complete ablation study")
    print("2. Analyze existing results")
    print("3. Create visualizations")
    print("4. Show experiment plan")
    print("5. Use command-line interface")
    
    while True:
        try:
            choice = input("\nEnter choice (1-5) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Goodbye!")
                break
            elif choice == '1':
                run_ablation_study_interactive()
                break
            elif choice == '2':
                analyze_results_interactive()
                break
            elif choice == '3':
                create_visualizations_interactive()
                break
            elif choice == '4':
                show_experiment_plan()
                break
            elif choice == '5':
                show_cli_help()
                break
            else:
                print("Invalid choice. Please enter 1-5 or 'q'.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def run_ablation_study_interactive():
    """Interactive ablation study runner."""
    from mia_experiments import run_ablation_study, OptimizationLevel
    
    print("\nABLATION STUDY SETUP")
    print("-" * 30)
    
    # Get optimization level
    print("Choose optimization level:")
    print("1. None (use sklearn defaults, fastest)")
    print("2. Quick (small grid search, ~2-3 hours)")
    print("3. Full (large grid search, ~4-6 hours)")
    
    while True:
        opt_choice = input("Enter choice (1-3): ").strip()
        if opt_choice == '1':
            optimization = 'none'
            break
        elif opt_choice == '2':
            optimization = 'quick'
            break
        elif opt_choice == '3':
            optimization = 'full'
            break
        else:
            print("Please enter 1, 2, or 3")
    
    # Confirm
    print(f"\nConfiguration:")
    print(f"  Atlas: ./data/atlas")
    print(f"  Training: ./data/train")
    print(f"  Test: ./data/test")
    print(f"  Optimization: {optimization}")
    print(f"  Output: ./ablation_experiments")
    
    response = input("\nProceed? (y/n): ").lower().strip()
    if response != 'y':
        print("Cancelled.")
        return
    
    try:
        print("\n" + "="*50)
        results = run_ablation_study(
            './data/atlas',
            './data/train', 
            './data/test',
            optimization,
            './ablation_experiments'
        )
        print("Ablation study completed! Check ./ablation_experiments/ for results.")
    except Exception as e:
        print(f"Error: {e}")


def analyze_results_interactive():
    """Interactive results analyzer."""
    from mia_experiments import analyze_results
    
    exp_dir = input("Enter experiment directory (default: ./ablation_experiments): ").strip()
    if not exp_dir:
        exp_dir = "./ablation_experiments"
    
    if not os.path.exists(exp_dir):
        print(f"Directory not found: {exp_dir}")
        return
    
    optimization = input("Enter optimization level used (none/quick/full, default: quick): ").strip()
    if not optimization:
        optimization = "quick"
    
    try:
        analyze_results(exp_dir, optimization)
        print(f"Analysis complete! Check {exp_dir}/analysis_{optimization}/")
    except Exception as e:
        print(f"Error: {e}")


def create_visualizations_interactive():
    """Interactive visualization creator."""
    from mia_experiments import load_and_visualize
    
    exp_dir = input("Enter experiment directory (default: ./ablation_experiments): ").strip()
    if not exp_dir:
        exp_dir = "./ablation_experiments"
    
    if not os.path.exists(exp_dir):
        print(f"Directory not found: {exp_dir}")
        return
    
    try:
        plot_files = load_and_visualize(exp_dir)
        if plot_files:
            print("Visualizations created:")
            for plot_type, path in plot_files.items():
                print(f"  {plot_type}: {path}")
        else:
            print("No experiment data found")
    except Exception as e:
        print(f"Error: {e}")


def show_experiment_plan():
    """Show the experiment plan."""
    from mia_experiments import AblationStudyConfigurator
    
    print("\nABLATION STUDY EXPERIMENT PLAN")
    print("-" * 40)
    
    experiments = AblationStudyConfigurator.get_experiment_summary()
    for exp_id, description in experiments.items():
        print(f"  {exp_id}: {description}")
    
    print("\nThis systematic approach tests the contribution of each")
    print("preprocessing component to brain tissue segmentation performance.")
    

def show_cli_help():
    """Show command-line interface help."""
    print("\n💻 COMMAND-LINE INTERFACE")
    print("-" * 30)
    print("For more advanced usage, use the command-line interface:")
    print()
    print("# Run ablation study")
    print("python -m mia_experiments.cli run \\")
    print("  --data-atlas /path/to/atlas \\")
    print("  --data-train /path/to/train \\")
    print("  --data-test /path/to/test \\")
    print("  --optimization quick")
    print()
    print("# Analyze results")
    print("python -m mia_experiments.cli analyze \\")
    print("  --experiment-dir ./ablation_experiments \\")
    print("  --optimization quick")
    print()
    print("# Create visualizations")
    print("python -m mia_experiments.cli visualize \\")
    print("  --experiment-dir ./ablation_experiments")
    print()
    print("# Show all options")
    print("python -m mia_experiments.cli --help")


if __name__ == '__main__':
    main()