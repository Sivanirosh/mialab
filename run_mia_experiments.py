#!/usr/bin/env python3
"""
Simple launcher script for MIA Experiments Framework.

This script provides an easy entry point to the CLI tools. For interactive use,
use the mia_experiments.cli module directly with command-line arguments.
"""

import os
import sys

# Add the package to Python path
package_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, package_dir)


def main():
    """Main entry point - shows usage examples and launches CLI."""
    print("=" * 70)
    print("MIA EXPERIMENTS FRAMEWORK")
    print("Brain Tissue Segmentation Ablation Studies")
    print("=" * 70)
    
    print("\nUSAGE EXAMPLES:")
    print("-" * 70)
    
    print("\n1. Run ablation study:")
    print("   python -m mia_experiments.cli run \\")
    print("     --data-atlas ./data/atlas \\")
    print("     --data-train ./data/train \\")
    print("     --data-test ./data/test \\")
    print("     --optimization quick")
    
    print("\n2. Analyze results:")
    print("   python analyze_ablation.py ./ablation_experiments")
    
    print("\n3. Create visualizations only:")
    print("   python analyze_ablation.py --mode visualize ./ablation_experiments")
    
    print("\n4. Comprehensive analysis:")
    print("   python -m mia_experiments.cli analyze \\")
    print("     --experiment-dir ./ablation_experiments \\")
    print("     --optimization quick")
    
    print("\n" + "=" * 70)
    print("For more options, run: python -m mia_experiments.cli --help")
    print("=" * 70)
    
    # Check if being called as a script (not imported)
    if sys.argv[0] == __file__ and len(sys.argv) > 1:
        # Forward arguments to CLI
        from mia_experiments import cli
        cli.main()
    else:
        print("\nTip: Add arguments to this script to use the CLI directly.")
        print("Example: python run_mia_experiments.py analyze --experiment-dir ./ablation_experiments")


if __name__ == '__main__':
    main()