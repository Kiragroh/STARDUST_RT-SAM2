import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import subprocess
from pathlib import Path

def run_metrics_computation(seg_dirs, gt_dir, output_dirs):
    """Run the metrics computation for each segmentation directory"""
    for seg_dir, output_dir in zip(seg_dirs, output_dirs):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the metrics computation script
        cmd = [
            "python", 
            "./eval_metrics/compute_metrics_flare22.py", 
            "-s", seg_dir, 
            "-g", gt_dir, 
            "-csv_dir", output_dir
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

def load_metrics(model_dirs):
    """Load metrics from each model directory"""
    model_metrics = {}
    
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        dsc_summary_path = os.path.join(model_dir, 'dsc_summary.csv')
        nsd_summary_path = os.path.join(model_dir, 'nsd_summary.csv')
        
        if os.path.exists(dsc_summary_path) and os.path.exists(nsd_summary_path):
            dsc_df = pd.read_csv(dsc_summary_path)
            nsd_df = pd.read_csv(nsd_summary_path)
            model_metrics[model_name] = {
                'dsc': dsc_df,
                'nsd': nsd_df
            }
        else:
            print(f"Warning: Could not find metrics files for {model_name}")
    
    return model_metrics

def generate_comparison_plots(model_metrics, comparison_dir):
    """Generate comparison plots for DSC and NSD metrics"""
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Get model names
    model_names = list(model_metrics.keys())
    
    if len(model_names) < 2:
        print("Error: Need at least two models to compare")
        return
    
    # Compare DSC values
    plt.figure(figsize=(12, 8))
    
    # Extract organ names and values for each model
    all_organs = []
    for model_name in model_names:
        organs = model_metrics[model_name]['dsc'].iloc[:, 0].str.replace('_DSC', '')
        all_organs.extend(organs.tolist())
    
    # Get unique organ names
    unique_organs = sorted(set(all_organs))
    
    # Prepare data for bar chart
    x = np.arange(len(unique_organs))
    width = 0.35
    offsets = np.linspace(-width/2, width/2, len(model_names))
    
    # Plot DSC values for each model
    for i, model_name in enumerate(model_names):
        dsc_df = model_metrics[model_name]['dsc']
        organs = dsc_df.iloc[:, 0].str.replace('_DSC', '')
        values = dsc_df.iloc[:, 1].values
        
        # Create a dictionary mapping organ names to values
        organ_values = {organ: value for organ, value in zip(organs, values)}
        
        # Get values for all unique organs (use 0 if not present)
        model_values = [organ_values.get(organ, 0) for organ in unique_organs]
        
        plt.bar(x + offsets[i], model_values, width/len(model_names), label=model_name)
    
    plt.xlabel('Organs')
    plt.ylabel('DSC Value')
    plt.title('Comparison of DSC Values Across Models')
    plt.xticks(x, unique_organs, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'dsc_comparison.png'))
    plt.close()
    
    # Compare NSD values
    plt.figure(figsize=(12, 8))
    
    # Extract organ names and values for each model
    all_organs = []
    for model_name in model_names:
        organs = model_metrics[model_name]['nsd'].iloc[:, 0].str.replace('_NSD', '')
        all_organs.extend(organs.tolist())
    
    # Get unique organ names
    unique_organs = sorted(set(all_organs))
    
    # Prepare data for bar chart
    x = np.arange(len(unique_organs))
    
    # Plot NSD values for each model
    for i, model_name in enumerate(model_names):
        nsd_df = model_metrics[model_name]['nsd']
        organs = nsd_df.iloc[:, 0].str.replace('_NSD', '')
        values = nsd_df.iloc[:, 1].values
        
        # Create a dictionary mapping organ names to values
        organ_values = {organ: value for organ, value in zip(organs, values)}
        
        # Get values for all unique organs (use 0 if not present)
        model_values = [organ_values.get(organ, 0) for organ in unique_organs]
        
        plt.bar(x + offsets[i], model_values, width/len(model_names), label=model_name)
    
    plt.xlabel('Organs')
    plt.ylabel('NSD Value')
    plt.title('Comparison of NSD Values Across Models')
    plt.xticks(x, unique_organs, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'nsd_comparison.png'))
    plt.close()
    
    # Create a summary table with both DSC and NSD values
    summary_data = []
    
    for organ in unique_organs:
        row = {'Organ': organ}
        
        for model_name in model_names:
            dsc_df = model_metrics[model_name]['dsc']
            nsd_df = model_metrics[model_name]['nsd']
            
            # Get DSC value for this organ and model
            dsc_value = dsc_df.loc[dsc_df.iloc[:, 0].str.replace('_DSC', '') == organ, dsc_df.columns[1]].values
            dsc_value = dsc_value[0] if len(dsc_value) > 0 else 0
            
            # Get NSD value for this organ and model
            nsd_value = nsd_df.loc[nsd_df.iloc[:, 0].str.replace('_NSD', '') == organ, nsd_df.columns[1]].values
            nsd_value = nsd_value[0] if len(nsd_value) > 0 else 0
            
            row[f'{model_name}_DSC'] = dsc_value
            row[f'{model_name}_NSD'] = nsd_value
        
        summary_data.append(row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV
    summary_df.to_csv(os.path.join(comparison_dir, 'metrics_comparison.csv'), index=False)
    
    # Calculate average metrics across all organs for each model
    avg_metrics = {}
    for model_name in model_names:
        dsc_cols = [col for col in summary_df.columns if col.endswith('_DSC')]
        nsd_cols = [col for col in summary_df.columns if col.endswith('_NSD')]
        
        avg_dsc = summary_df[dsc_cols].mean().to_dict()
        avg_nsd = summary_df[nsd_cols].mean().to_dict()
        
        avg_metrics.update(avg_dsc)
        avg_metrics.update(avg_nsd)
    
    # Create average metrics DataFrame
    avg_df = pd.DataFrame([avg_metrics])
    
    # Save average metrics to CSV
    avg_df.to_csv(os.path.join(comparison_dir, 'average_metrics.csv'), index=False)
    
    # Create a bar chart for average metrics
    plt.figure(figsize=(10, 6))
    
    # Prepare data for bar chart
    metric_types = ['DSC', 'NSD']
    x = np.arange(len(metric_types))
    width = 0.35
    offsets = np.linspace(-width/2, width/2, len(model_names))
    
    # Plot average metrics for each model
    for i, model_name in enumerate(model_names):
        avg_dsc = avg_df[f'{model_name}_DSC'].values[0]
        avg_nsd = avg_df[f'{model_name}_NSD'].values[0]
        
        # Use the same color for both DSC and NSD for each model
        plt.bar(x[0] + offsets[i], avg_dsc, width/len(model_names), label=model_name)
        plt.bar(x[1] + offsets[i], avg_nsd, width/len(model_names), color=plt.gca().patches[-1].get_facecolor())
    
    plt.xlabel('Metric Type')
    plt.ylabel('Average Value')
    plt.title('Comparison of Average Metrics Across Models')
    plt.xticks(x, metric_types)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'average_metrics.png'))
    plt.close()
    
    print(f"Comparison plots and tables saved to {comparison_dir}")

def main():
    parser = argparse.ArgumentParser(description="Compare metrics between different segmentation models")
    parser.add_argument(
        "-s", "--seg_dirs", 
        nargs='+', 
        required=True,
        help="List of segmentation directories to compare"
    )
    parser.add_argument(
        "-g", "--gt_dir", 
        required=True,
        help="Ground truth directory"
    )
    parser.add_argument(
        "-o", "--output_base_dir", 
        default="./metric_results",
        help="Base directory for output metrics"
    )
    parser.add_argument(
        "-c", "--comparison_dir", 
        default=None,
        help="Directory to save comparison results (default: output_base_dir/comparison)"
    )
    parser.add_argument(
        "--compute_metrics", 
        action="store_true",
        help="Compute metrics before comparison (if metrics already exist, this can be skipped)"
    )
    
    args = parser.parse_args()
    
    # Create output directories for each segmentation directory
    output_dirs = []
    model_names = []
    
    for seg_dir in args.seg_dirs:
        model_name = os.path.basename(seg_dir)
        model_names.append(model_name)
        output_dir = os.path.join(args.output_base_dir, Path(seg_dir).parts[-2], model_name)
        output_dirs.append(output_dir)
    
    # Set comparison directory
    if args.comparison_dir is None:
        comparison_dir = os.path.join(args.output_base_dir, Path(args.seg_dirs[0]).parts[-2], "comparison")
    else:
        comparison_dir = args.comparison_dir
    
    # Compute metrics if requested
    if args.compute_metrics:
        run_metrics_computation(args.seg_dirs, args.gt_dir, output_dirs)
    
    # Load metrics
    model_metrics = load_metrics(output_dirs)
    
    # Generate comparison plots
    generate_comparison_plots(model_metrics, comparison_dir)

if __name__ == "__main__":
    main()
