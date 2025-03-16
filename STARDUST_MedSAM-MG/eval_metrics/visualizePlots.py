import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Function to generate plots if the required CSV files are found
def generate_plots(folder):
    # File paths
    dsc_summary_path = os.path.join(folder, 'dsc_summary.csv')
    metrics_path = os.path.join(folder, 'metrics.csv')
    nsd_summary_path = os.path.join(folder, 'nsd_summary.csv')

    # Check if all the required files are in the folder
    if os.path.exists(dsc_summary_path) and os.path.exists(metrics_path) and os.path.exists(nsd_summary_path):
        print(f"Found all required files in {folder}, generating plots...")
        # Load the CSVs into dataframes
        dsc_summary_df = pd.read_csv(dsc_summary_path)
        metrics_df = pd.read_csv(metrics_path)
        nsd_summary_df = pd.read_csv(nsd_summary_path)

        # Extract the organ names and values from DSC and NSD summaries
        organs_dsc = dsc_summary_df.iloc[:, 0].str.replace('_DSC', '')
        dsc_values = dsc_summary_df.iloc[:, 1].values

        organs_nsd = nsd_summary_df.iloc[:, 0].str.replace('_NSD', '')
        nsd_values = nsd_summary_df.iloc[:, 1].values

        # Plot DSC summary
        plt.figure(figsize=(10, 6))
        plt.barh(organs_dsc, dsc_values, color='skyblue')
        plt.xlabel('DSC Value')
        plt.title('Average Dice Similarity Coefficient (DSC) for Different Organs')
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, 'dsc_summary_plot.png'))
        print(f"Saved DSC summary plot to {os.path.join(folder, 'dsc_summary_plot.png')}")

        # Plot NSD summary
        plt.figure(figsize=(10, 6))
        plt.barh(organs_nsd, nsd_values, color='lightcoral')
        plt.xlabel('NSD Value')
        plt.title('Average Normalized Surface Distance (NSD) for Different Organs')
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, 'nsd_summary_plot.png'))
        print(f"Saved NSD summary plot to {os.path.join(folder, 'nsd_summary_plot.png')}")

        # Create a combined plot for DSC values across cases, using all organs
        plt.figure(figsize=(14, 8))

        # Plot DSC values for all organs - dynamisch alle DSC-Spalten identifizieren
        dsc_columns = [col for col in metrics_df.columns if '_DSC' in col]
        for organ in dsc_columns:
            plt.plot(range(len(metrics_df['case'])), metrics_df[organ], label=f'{organ}', marker='o')

        # Set plot labels and titles
        plt.xlabel('Case Number')
        plt.ylabel('DSC Value')
        plt.ylim(0, 1)
        plt.title('DSC Values Across Cases for Multiple Organs')
        plt.xticks(range(len(metrics_df['case'])), range(1, len(metrics_df) + 1), rotation=90)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, 'dsc_case_plot.png'))
        print(f"Saved DSC case plot to {os.path.join(folder, 'dsc_case_plot.png')}")

        # Create a combined plot for NSD values across cases, using all organs
        plt.figure(figsize=(14, 8))

        # Plot NSD values for all organs - dynamisch alle NSD-Spalten identifizieren
        nsd_columns = [col for col in metrics_df.columns if '_NSD' in col]
        for organ in nsd_columns:
            plt.plot(range(len(metrics_df['case'])), metrics_df[organ], label=f'{organ}', marker='o')

        # Set plot labels and titles
        plt.xlabel('Case Number')
        plt.ylabel('NSD Value')
        plt.ylim(0, 1)
        plt.title('NSD Values Across Cases for Multiple Organs')
        plt.xticks(range(len(metrics_df['case'])), range(1, len(metrics_df) + 1), rotation=90)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, 'nsd_case_plot.png'))
        print(f"Saved NSD case plot to {os.path.join(folder, 'nsd_case_plot.png')}")
        return True
    else:
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Generate plots from metrics CSV files')
    parser.add_argument(
        '--metrics_dir', 
        type=str, 
        default=None,
        help='Path to the metrics directory. If not specified, the default "metric_results" directory will be used.'
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set the base directory for metrics
    if args.metrics_dir:
        base_dir = args.metrics_dir
        if not os.path.exists(base_dir):
            print(f"Warning: Specified metrics directory {base_dir} does not exist.")
            exit(1)
    else:
        # Default: Use the standard metric_results directory
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metric_results")
    
    print(f"Searching for metrics in: {base_dir}")
    
    # Count how many directories were successfully processed
    success_count = 0
    
    # Process all subdirectories of the metrics directory
    for root, dirs, files in os.walk(base_dir):
        if generate_plots(root):
            success_count += 1
    
    if success_count == 0:
        print(f"No metrics data found in {base_dir}. Make sure to run 'compute_metrics' first.")
    else:
        print(f"Successfully generated plots for {success_count} metric directories.")
