import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob


def main():
    """Load results and create boxplots of Dice coefficients per tissue type."""

    # Find the most recent results directory
    result_dirs = glob.glob('./mia-result/*/results.csv')
    if not result_dirs:
        print("No results.csv file found in mia-result directory.")
        print("Please run the pipeline first to generate results.")
        return

    # Use the most recent results directory
    latest_results = max(result_dirs, key=os.path.getctime)
    print(f"Loading results from: {latest_results}")

    # Load CSV with proper separator
    try:
        data = pd.read_csv(latest_results, sep=';')
        print(f"Loaded {len(data)} rows of data")
        print(f"Columns: {list(data.columns)}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Check that required columns exist
    required_cols = {'SUBJECT', 'LABEL', 'DICE'}
    if not required_cols.issubset(data.columns):
        print(f"CSV must contain columns: {required_cols}")
        return

    # Convert DICE column to numeric
    data['DICE'] = pd.to_numeric(data['DICE'], errors='coerce')

    # Drop rows with missing values
    data = data.dropna(subset=['DICE', 'LABEL'])

    # Group DICE values by tissue LABEL
    grouped = data.groupby('LABEL')['DICE']

    tissue_labels = list(grouped.groups.keys())
    dice_data = [grouped.get_group(label).values for label in tissue_labels]

    print(f"Found tissue types: {tissue_labels}")

    # Create boxplot
    plt.figure(figsize=(12, 8))
    box_plot = plt.boxplot(dice_data, labels=tissue_labels, patch_artist=True)

    # Customize box colors
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(box_plot['boxes'], colors * (len(dice_data) // len(colors) + 1)):
        patch.set_facecolor(color)

    plt.title('Dice Coefficients per Tissue Type', fontsize=16, fontweight='bold')
    plt.xlabel('Tissue Type', fontsize=12)
    plt.ylabel('Dice Coefficient', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    # Add mean values as text above boxes
    for i, (data_vals, label) in enumerate(zip(dice_data, tissue_labels)):
        mean_val = np.mean(data_vals)
        plt.text(i + 1, mean_val + 0.01, f'μ={mean_val:.3f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()

    # Save the plot
    plot_dir = os.path.dirname(latest_results)
    plot_path = os.path.join(plot_dir, 'dice_coefficients_boxplot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # Print summary statistics
    print("\nDice Coefficient Statistics:")
    print("=" * 50)
    for label, values in zip(tissue_labels, dice_data):
        print(f"{label}:")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std:  {np.std(values):.4f}")
        print(f"  Min:  {np.min(values):.4f}")
        print(f"  Max:  {np.max(values):.4f}")
        print()

    plt.show()


if __name__ == '__main__':
    main()
