import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
from pathlib import Path
import glob

# Define a consistent color mapping for each organ
def create_organ_colormap():
    """Create a consistent color mapping for each organ"""
    organ_colors = {
        'GTV+': [1, 0, 0, 1.0],                # Red
        'Target+': [0, 1, 0, 1.0],             # Green
        'Spleen': [0, 0.7, 0, 1.0],            # Green
        'Pancreas': [1, 1, 0, 1.0],            # Yellow
        'Aorta': [1, 0, 1, 1.0],               # Magenta
        'Inferior Vena Cava': [0, 1, 1, 1.0],  # Cyan
        'Right Adrenal Gland': [0, 0.8, 0.8, 1.0], # Light Cyan
        'Left Adrenal Gland': [0.8, 0, 0.8, 1.0],  # Light Purple
        'Gallbladder': [0, 0.5, 0, 1.0],       # Dark Green
        'Esophagus': [0.5, 0, 0.5, 1.0],       # Purple
        'Stomach': [0.5, 0.5, 0, 1.0],         # Olive
        'Left Kidney': [1, 0, 0, 1.0],         # Red
        'Other': [0.7, 0.7, 0.7, 1.0]          # Gray
    }
    return organ_colors

# Dictionary mapping label IDs to organ names
label_dict = {
    1: 'GTV+',
    2: 'Target+',
    3: 'Spleen',
    4: 'Pancreas',
    5: 'Aorta',
    6: 'Inferior Vena Cava',
    7: 'Right Adrenal Gland',
    8: 'Left Adrenal Gland',
    9: 'Gallbladder',
    10: 'Esophagus',
    11: 'Stomach',
    13: 'Left Kidney'
}

# Normalize image to 0-1 range for visualization
def normalize_image(image):
    """Normalize image to 0-1 range for visualization"""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return image

# Load data from NPZ file
def load_npz(file_path, key=None):
    """Load data from NPZ file"""
    data = np.load(file_path, allow_pickle=True)
    if key is not None:
        return data[key]
    return data

# Find files matching a pattern in a directory
def find_files(base_dir, pattern):
    """Find files matching a pattern in a directory"""
    return glob.glob(os.path.join(base_dir, pattern))

def compare_label_segmentations(gt_npz_file, result_npz_file, output_dir, label_id=1, slice_indices=None, rotation=0):
    """
    Create a 2-panel visualization comparing:
    1. CT image with ground truth segmentation for a specific label
    2. CT image with result segmentation for the same label
    
    Args:
        gt_npz_file: Path to ground truth NPZ file
        result_npz_file: Path to result segmentation NPZ file
        output_dir: Directory to save visualization images
        label_id: Label ID to visualize (default: 1 for GTV+)
        slice_indices: List of slice indices to visualize (if None, will select automatically)
        rotation: Rotation to apply to images (0=none, 1=90deg, 2=180deg, 3=270deg)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    gt_data = load_npz(gt_npz_file)
    result_data = load_npz(result_npz_file)
    
    # Get image and segmentation data
    ct_image = gt_data['imgs']
    gt_segmentation = gt_data['gts']
    
    # Get result segmentation
    if 'pred' in result_data:
        result_segmentation = result_data['pred']
    elif f'{label_dict[label_id]}' in result_data:
        result_segmentation = result_data[f'{label_dict[label_id]}']
    else:
        # Try to find the key that contains the label name
        label_name = label_dict[label_id]
        matching_keys = [key for key in result_data.keys() if label_name in key]
        if matching_keys:
            result_segmentation = result_data[matching_keys[0]]
        else:
            print(f"Could not find segmentation for label {label_id} ({label_dict[label_id]}) in result file")
            print(f"Available keys: {list(result_data.keys())}")
            return
    
    # Apply rotation if needed
    if rotation > 0:
        ct_image = np.rot90(ct_image, k=rotation, axes=(1, 2))
        gt_segmentation = np.rot90(gt_segmentation, k=rotation, axes=(1, 2))
        result_segmentation = np.rot90(result_segmentation, k=rotation, axes=(1, 2))
    
    # Get image dimensions
    depth, height, width = ct_image.shape
    
    # Create binary masks for the specific label
    gt_mask = (gt_segmentation == label_id)
    result_mask = (result_segmentation > 0)  # Assuming binary mask or threshold > 0
    
    # Determine which slices to visualize
    if slice_indices is None:
        # Find slices where the label appears in either GT or result
        label_slices = []
        for z in range(depth):
            if np.any(gt_mask[z]) or np.any(result_mask[z]):
                label_slices.append(z)
        
        # If no slices found, use middle slice
        if not label_slices:
            slice_indices = [depth // 2]
        else:
            # Use slices where the label appears
            slice_indices = label_slices
    
    # Create colormap for visualization
    organ_colors = create_organ_colormap()
    label_name = label_dict.get(label_id, f'Unknown_{label_id}')
    label_color = organ_colors.get(label_name, organ_colors['Other'])
    
    # Process each slice
    for z_idx in slice_indices:
        if z_idx < 0 or z_idx >= depth:
            continue
            
        # Get slice data
        ct_slice = ct_image[z_idx]
        gt_slice = gt_mask[z_idx]
        result_slice = result_mask[z_idx]
        
        # Normalize CT image for visualization
        ct_slice_norm = normalize_image(ct_slice)
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot ground truth
        axes[0].imshow(ct_slice_norm, cmap='gray')
        gt_overlay = np.zeros((height, width, 4))
        gt_overlay[gt_slice] = label_color
        axes[0].imshow(gt_overlay)
        axes[0].set_title(f'Ground Truth - {label_name}')
        axes[0].axis('off')
        
        # Plot result
        axes[1].imshow(ct_slice_norm, cmap='gray')
        result_overlay = np.zeros((height, width, 4))
        result_overlay[result_slice] = label_color
        axes[1].imshow(result_overlay)
        axes[1].set_title(f'Segmentation Result - {label_name}')
        axes[1].axis('off')
        
        # Add case and slice information
        case_name = os.path.basename(gt_npz_file).split('.')[0]
        plt.suptitle(f'Case: {case_name}, Slice: {z_idx}, Label: {label_name}')
        
        # Save figure
        output_file = os.path.join(output_dir, f'{case_name}_slice{z_idx:03d}_label{label_id}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved visualization to {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='Compare ground truth and result segmentations for a specific label')
    parser.add_argument('--gt_npz', type=str, required=True, help='Path to ground truth NPZ file')
    parser.add_argument('--result_npz', type=str, required=True, help='Path to result segmentation NPZ file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for visualizations')
    parser.add_argument('--label_id', type=int, default=1, help='Label ID to visualize (default: 1 for GTV+)')
    parser.add_argument('--slice_indices', type=str, help='Comma-separated list of slice indices to visualize (e.g., "10,20,30")')
    parser.add_argument('--rotation', type=int, default=0, choices=[0, 1, 2, 3], help='Rotation to apply to images (0=none, 1=90deg, 2=180deg, 3=270deg)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Process slice indices if provided
    slice_indices = None
    if args.slice_indices:
        slice_indices = [int(idx) for idx in args.slice_indices.split(',')]
    
    # Compare segmentations
    compare_label_segmentations(
        args.gt_npz,
        args.result_npz,
        args.output_dir,
        args.label_id,
        slice_indices,
        args.rotation
    )

if __name__ == "__main__":
    main()
