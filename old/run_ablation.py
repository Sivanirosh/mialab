#!/usr/bin/env python3
"""
Simple runner script for the 9-experiment ablation study.

This script demonstrates how to run the comprehensive ablation study
to understand the contribution of each preprocessing component.
"""

import os
import sys
import time

def main():
    print("COMPREHENSIVE ABLATION STUDY")
    print("Testing: Baseline → Normalization → Skull Stripping → Registration → Post-processing")
    print("=" * 80)
    
    # Check data availability
    data_paths = {
        'atlas': './data/atlas',
        'train': './data/train',
        'test': './data/test'
    }
    
    print("Checking data directories...")
    missing_data = []
    for name, path in data_paths.items():
        if os.path.exists(path):
            print(f"{name}: {path}")
        else:
            print(f"{name}: {path} (not found)")
            missing_data.append(name)
    
    if missing_data:
        print(f"\nMissing data directories: {missing_data}")
        print("Please run: python prepare_data.py --data_dir /path/to/your/raw/data")
        return
    
    print("\nEXPERIMENT PLAN:")
    print("-" * 40)
    experiments = {
        0: "Baseline (no preprocessing)",
        1: "Normalization only", 
        2: "Skull stripping only",
        3: "Registration only",
        4: "Normalization + Skull stripping",
        5: "Normalization + Registration", 
        6: "Registration + Skull stripping",
        7: "All preprocessing (Norm + Skull + Reg)",
        8: "All preprocessing + Post-processing"
    }
    
    for exp_id, description in experiments.items():
        print(f"  {exp_id}: {description}")
    
    print("\nRANDOM FOREST OPTIMIZATION OPTIONS:")
    print("-" * 40)
    print("  none: Use default sklearn parameters (fastest, baseline)")
    print("  quick: Small hyperparameter grid (~48 combinations, ~30 min)")
    print("  full: Large hyperparameter grid (~864 combinations, ~3 hours)")
    
    # Ask for optimization level
    print("\nWhich optimization level would you like to use?")
    print("1: none (default parameters)")
    print("2: quick (recommended)")
    print("3: full (thorough)")
    
    while True:
        choice = input("Enter choice (1/2/3): ").strip()
        if choice == '1':
            optimization_level = 'none'
            expected_time = "1-2 hours"
            break
        elif choice == '2':
            optimization_level = 'quick'
            expected_time = "2-3 hours"
            break
        elif choice == '3':
            optimization_level = 'full'
            expected_time = "4-6 hours"
            break
        else:
            print("Please enter 1, 2, or 3")
    
    print(f"\nSelected: {optimization_level} optimization")
    print(f"Expected runtime: {expected_time}")
    
    # Ask for confirmation
    response = input(f"\nProceed with ablation study? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Ablation study cancelled.")
        return
    
    # Run the ablation study
    print(f"\nSTARTING ABLATION STUDY")
    print("=" * 40)
    
    try:
        from ablation_study import AblationStudyRunner
        
        # Create runner
        runner = AblationStudyRunner("./ablation_experiments")
        
        # Run complete study
        results = runner.run_ablation_study(
            data_atlas_dir=data_paths['atlas'],
            data_train_dir=data_paths['train'],
            data_test_dir=data_paths['test'],
            optimization_level=optimization_level
        )
        
        print(f"\nABLATION STUDY COMPLETED!")
        print("=" * 40)
        print(f"Results directory: ./ablation_experiments/")
        print(f"Analysis plots: ./ablation_experiments/analysis_{optimization_level}/")
        print(f"Combined results: ./ablation_experiments/combined_results_{optimization_level}.csv")
        
        # Show quick summary
        completed = sum(1 for r in results.values() if r['status'] == 'completed')
        failed = len(results) - completed
        
        print(f"\nSUMMARY:")
        print(f"   Completed experiments: {completed}/9")
        print(f"   Failed experiments: {failed}/9")
        print(f"   Optimization level: {optimization_level}")
        
        if completed > 0:
            print(f"\nNEXT STEPS:")
            print(f"1. Review analysis plots in ./ablation_experiments/analysis_{optimization_level}/")
            print(f"2. Check component_contributions_{optimization_level}.json for quantitative effects")
            print(f"3. Review statistical_tests.csv for significance testing")
            print(f"4. Compare different optimization levels if needed")
            
            # Show what files to look at
            print(f"\nKEY FILES:")
            print(f"   ./ablation_experiments/analysis_{optimization_level}/overall_performance_{optimization_level}.png")
            print(f"   ./ablation_experiments/analysis_{optimization_level}/component_effects_{optimization_level}.png")
            print(f"   ./ablation_experiments/analysis_{optimization_level}/summary_statistics_{optimization_level}.csv")
    
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required modules are available")
    except Exception as e:
        print(f"Error during ablation study: {e}")
        print("Check the logs for detailed error information")


if __name__ == "__main__":
    main()