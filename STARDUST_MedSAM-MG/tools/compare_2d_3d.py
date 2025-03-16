import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(model_dirs, model_names=None):
    """Load metrics from each model directory"""
    model_metrics = {}
    
    for i, model_dir in enumerate(model_dirs):
        # Use provided model name if available, otherwise use directory basename
        if model_names and i < len(model_names):
            model_name = model_names[i]
        else:
            model_name = os.path.basename(os.path.dirname(model_dir)) + "_" + os.path.basename(model_dir)
        
        dsc_summary_path = os.path.join(model_dir, 'dsc_summary.csv')
        nsd_summary_path = os.path.join(model_dir, 'nsd_summary.csv')
        
        if os.path.exists(dsc_summary_path) and os.path.exists(nsd_summary_path):
            dsc_df = pd.read_csv(dsc_summary_path, index_col=0)
            nsd_df = pd.read_csv(nsd_summary_path, index_col=0)
            
            # Extract organ names and values
            dsc_values = {}
            for idx, row in dsc_df.iterrows():
                if idx.endswith('_DSC'):
                    organ = idx.replace('_DSC', '')
                    dsc_values[organ] = row[0]
            
            nsd_values = {}
            for idx, row in nsd_df.iterrows():
                if idx.endswith('_NSD'):
                    organ = idx.replace('_NSD', '')
                    nsd_values[organ] = row[0]
            
            model_metrics[model_name] = {
                'dsc_values': dsc_values,
                'nsd_values': nsd_values
            }
            print(f"Loaded metrics for {model_name}")
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
    
    print(f"Comparing models: {', '.join(model_names)}")
    
    # Get all unique organs across all models
    all_organs = set()
    for model_name in model_names:
        all_organs.update(model_metrics[model_name]['dsc_values'].keys())
    
    all_organs = sorted(list(all_organs))
    print(f"Found {len(all_organs)} organs: {', '.join(all_organs)}")
    
    # Create comparison table
    comparison_data = []
    
    for model_name in model_names:
        dsc_values = model_metrics[model_name]['dsc_values']
        nsd_values = model_metrics[model_name]['nsd_values']
        
        for organ in all_organs:
            dsc_val = dsc_values.get(organ, 0)
            nsd_val = nsd_values.get(organ, 0)
            
            comparison_data.append({
                'Model': model_name,
                'Organ': organ,
                'DSC': dsc_val,
                'NSD': nsd_val
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(comparison_dir, 'metrics_comparison.csv'), index=False)
    
    # Create DSC comparison plot
    plt.figure(figsize=(14, 8))
    x = np.arange(len(all_organs))
    width = 0.35
    
    for i, model_name in enumerate(model_names):
        model_data = comparison_df[comparison_df['Model'] == model_name]
        dsc_values = []
        
        for organ in all_organs:
            organ_data = model_data[model_data['Organ'] == organ]
            dsc_values.append(organ_data['DSC'].values[0] if not organ_data.empty else 0)
        
        plt.bar(x + (i - len(model_names)/2 + 0.5) * width, dsc_values, width, label=model_name)
    
    plt.xlabel('Organ')
    plt.ylabel('DSC')
    plt.title('Dice Similarity Coefficient (DSC) Comparison')
    plt.xticks(x, all_organs, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'dsc_comparison.png'))
    plt.close()
    
    # Create NSD comparison plot
    plt.figure(figsize=(14, 8))
    
    for i, model_name in enumerate(model_names):
        model_data = comparison_df[comparison_df['Model'] == model_name]
        nsd_values = []
        
        for organ in all_organs:
            organ_data = model_data[model_data['Organ'] == organ]
            nsd_values.append(organ_data['NSD'].values[0] if not organ_data.empty else 0)
        
        plt.bar(x + (i - len(model_names)/2 + 0.5) * width, nsd_values, width, label=model_name)
    
    plt.xlabel('Organ')
    plt.ylabel('NSD')
    plt.title('Normalized Surface Distance (NSD) Comparison')
    plt.xticks(x, all_organs, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'nsd_comparison.png'))
    plt.close()
    
    # Create average metrics comparison
    plt.figure(figsize=(10, 6))
    metric_types = ['DSC', 'NSD']
    x = np.arange(len(metric_types))
    width = 0.8 / len(model_names)
    offsets = np.linspace(-width * (len(model_names) - 1) / 2, width * (len(model_names) - 1) / 2, len(model_names))
    
    avg_metrics = []
    for model_name in model_names:
        model_data = comparison_df[comparison_df['Model'] == model_name]
        avg_dsc = model_data['DSC'].mean()
        avg_nsd = model_data['NSD'].mean()
        avg_metrics.append({
            'Model': model_name,
            'DSC': avg_dsc,
            'NSD': avg_nsd
        })
    
    avg_df = pd.DataFrame(avg_metrics)
    avg_df.to_csv(os.path.join(comparison_dir, 'average_metrics.csv'), index=False)
    
    for i, model_name in enumerate(model_names):
        avg_dsc = avg_df[avg_df['Model'] == model_name]['DSC'].values[0]
        avg_nsd = avg_df[avg_df['Model'] == model_name]['NSD'].values[0]
        
        plt.bar(x[0] + offsets[i], avg_dsc, width, label=model_name)
        plt.bar(x[1] + offsets[i], avg_nsd, width)
    
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
    # Paths to the metric results
    medsam2_2d_dir = "./metric_results/2D/medsam2"
    medsam2_3d_dir = "./metric_results/3D/medsam2"
    comparison_dir = "./metric_results/2D_vs_3D_comparison"
    
    # Load metrics with explicit model names
    model_metrics = load_metrics(
        [medsam2_2d_dir, medsam2_3d_dir],
        ["MedSAM2_2D", "MedSAM2_3D"]
    )
    
    # Generate comparison plots
    generate_comparison_plots(model_metrics, comparison_dir)

if __name__ == "__main__":
    main()
