import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
from pathlib import Path
import glob
import datetime
from skimage import measure

# Define a consistent color mapping for each organ
def create_organ_colormap():
    """Create a consistent color mapping for each organ"""
    organ_colors = {
        'GTV+': [1, 0, 0, 1.0],               # Rot - Höchste Priorität
        'Target+': [1, 1, 0, 1.0],            # Gelb
        'GTV': [1, 0, 0, 1.0],                # Rot (gleich wie GTV+)
        'Target': [1, 1, 0, 1.0],             # Gelb (gleich wie Target+)        
        'Other': [0.7, 0.7, 0.7, 1.0],           # Gray
        'Unknown': [0.7, 0.7, 0.7, 1.0]          # Gray
    }
    return organ_colors

# Dictionary mapping label IDs to organ names
label_dict = {
    1: 'GTV+',
    2: 'Target+'
}

# Define a colormap for segmentations with distinct colors
def create_segmentation_colormap():
    """Create a colormap for segmentations using the consistent organ colors"""
    # Get the organ colors
    organ_colors = create_organ_colormap()
    
    # Create a list of colors for the colormap
    # First color (index 0) is transparent for background
    colors = [[0, 0, 0, 0]]  # Background - transparent
    
    # Use consistent colors for our most common labels (1-15)
    # Map label IDs directly to colors from organ_colors
    label_to_color = {
        1: organ_colors['GTV'],      # GTV/GTV+
        2: organ_colors['Target']
    }
    
    # Add 15 colors to the colormap to ensure enough colors available
    for i in range(1, 16):
        if i in label_to_color:
            colors.append(label_to_color[i])
        else:
            # Use a gray color with varying intensity for any undefined labels
            gray_intensity = 0.4 + (i / 30)  # varies from ~0.4 to ~0.9
            colors.append([gray_intensity, gray_intensity, gray_intensity, 0.7])
    
    return ListedColormap(colors)

def normalize_image(image):
    """Normalize image to 0-1 range for visualization"""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return image

def plot_contours(ax, mask, label_dict, organ_colors):
    """
    Zeichnet Konturen für jedes Label in der gegebenen Maske und nutzt die zugehörige Farbe aus `organ_colors`.
    
    Args:
        ax: Matplotlib-Achse
        mask: 2D-Array der Segmentierung
        label_dict: Dictionary {Label-ID: Organ-Name}
        organ_colors: Dictionary {Organ-Name: Farbe}
    """
    unique_labels = np.unique(mask)
    
    for label_id in unique_labels:
        if label_id == 0:  # Hintergrund ignorieren
            continue

        # Organname zum Label finden
        organ_name = label_dict.get(label_id, "Unknown")

        # Farbe aus `organ_colors` abrufen, falls unbekannt, Grau als Standard
        color = organ_colors.get(organ_name, [0.7, 0.7, 0.7, 1.0])  # Grau als Fallback

        # Konturen für dieses Label extrahieren
        contours = measure.find_contours(mask == label_id, 0.5)

        # Zeichne jede Kontur
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color=color[:3])  # RGB-Farbe nutzen
def load_npz(file_path):
    """
    Load NPZ file with allow_pickle=True to handle object arrays
    """
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading NPZ file {file_path}: {e}")
        return None

def find_files(base_dir, pattern):
    """Find files matching a pattern in a directory"""
    return glob.glob(os.path.join(base_dir, pattern))

def create_comparison_visualization(gt_npz_file, sam2_npz_file, medsam2_npz_file, output_dir, slice_indices=None, rotation=2, labels=None, clinical=False, row_size=2):
    """
    Create a 4-panel visualization comparing:
    1. CT image with ground truth segmentation
    2. CT image with ground truth bounding boxes
    3. CT image with SAM2 segmentation
    4. CT image with MedSAM2 segmentation
    
    Args:
        gt_npz_file: Path to ground truth NPZ file
        sam2_npz_file: Path to SAM2 segmentation NPZ file
        medsam2_npz_file: Path to MedSAM2 segmentation NPZ file
        output_dir: Directory to save visualization images
        slice_indices: List of slice indices to visualize (if None, will select automatically)
        rotation: Rotation to apply to images (0=none, 1=90deg, 2=180deg, 3=270deg)
        labels: Specific label IDs to visualize (if None, will visualize all labels found in the data)
        clinical: Clinical mode flag (if True, focus on GTV+/Target+ labels)
        row_size: Number of images per row in visualization grid
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract case name from file path
    case_name = os.path.basename(gt_npz_file).split('.')[0]
    
    # Create a debug log file
    debug_log_path = os.path.join(output_dir, f"{case_name}_debug.log")
    with open(debug_log_path, 'w') as debug_file:
        debug_file.write(f"Debug log for {case_name}\n")
        debug_file.write(f"Ground truth file: {gt_npz_file}\n")
        debug_file.write(f"SAM2 file: {sam2_npz_file}\n")
        debug_file.write(f"MedSAM2 file: {medsam2_npz_file}\n\n")
    
    # Load data
    gt_data = load_npz(gt_npz_file)
    sam2_data = load_npz(sam2_npz_file)
    medsam2_data = load_npz(medsam2_npz_file)
    
    # Log the keys in each file
    with open(debug_log_path, 'a') as debug_file:
        debug_file.write(f"Ground truth file keys: {list(gt_data.keys())}\n")
        debug_file.write(f"SAM2 file keys: {list(sam2_data.keys())}\n")
        debug_file.write(f"MedSAM2 file keys: {list(medsam2_data.keys())}\n")
    
    # Extract available structures from segmentation files (excluding 'imgs')
    available_structures = set()
    for file_data in [sam2_data, medsam2_data]:
        for key in file_data.keys():
            # Nur Schlüssel berücksichtigen, die nicht numerisch sind und nicht 'imgs' sind
            if key != 'imgs' and not key.isdigit():
                available_structures.add(key)
    
    # Map available structures to label IDs
    structure_to_label = {}
    for i, structure in enumerate(sorted(available_structures), 1):
        structure_to_label[structure] = i
    
    # Get CT images and ground truth segmentations
    ct_images = gt_data['imgs']
    gt_segmentations = gt_data['gts']
    
    # Determine which slices to visualize
    num_slices = ct_images.shape[0]
    if slice_indices is None:
        slice_indices = list(np.linspace(0, num_slices - 1, 4, dtype=int))
    
    # Create segmentation colormap
    seg_cmap = create_segmentation_colormap()
    
    # Visualize each slice
    for slice_idx in slice_indices:
        if slice_idx >= num_slices:
            with open(debug_log_path, 'a') as debug_file:
                debug_file.write(f"Skipping slice {slice_idx} as it is out of bounds (max: {num_slices-1})\n")
            continue
        
        with open(debug_log_path, 'a') as debug_file:
            debug_file.write(f"\nProcessing slice {slice_idx}\n")
        
        # Get CT slice and rotate as specified
        ct_slice = ct_images[slice_idx].astype(float)
        ct_slice = np.rot90(ct_slice, rotation)
        
        # Normalize the CT slice for visualization
        ct_slice = normalize_image(ct_slice)
        
        # Create empty label maps for ground truth, SAM2 and MedSAM2
        gt_slice = np.zeros_like(ct_slice, dtype=np.uint8)
        sam2_slice = np.zeros_like(ct_slice, dtype=np.uint8)
        medsam2_slice = np.zeros_like(ct_slice, dtype=np.uint8)
        
        # Copy ground truth labels for this slice and rotate
        gt_labels = gt_segmentations[slice_idx]
        gt_slice = np.rot90(gt_labels, rotation)
        
        # Log the unique labels in the ground truth
        unique_labels = np.unique(gt_slice)
        with open(debug_log_path, 'a') as debug_file:
            debug_file.write(f"  Ground truth unique labels: {unique_labels}\n")
        
        # Count pixels for each label in ground truth
        for label_id in unique_labels:
            if label_id == 0:  # Skip background
                continue
            
            # Count pixels with this label
            gt_pixel_count = np.sum(gt_slice == label_id)
                
            # Add to debug log
            with open(debug_log_path, 'a') as debug_file:
                debug_file.write(f"  Label ID {label_id} in ground truth: {gt_pixel_count} pixels\n")
        
        # Fill in segmentation masks from NPZ files for each structure
        for structure, label_id in structure_to_label.items():
            # Skip if specific labels are requested and this is not one of them
            if labels is not None and label_id not in labels:
                continue
            
            # Process SAM2 segmentation
            if structure in sam2_data.files:
                sam2_struct = sam2_data[structure]
                if slice_idx < sam2_struct.shape[0]:
                    # Count pixels before assignment
                    pixel_count_before = np.sum(sam2_slice == label_id)
                    
                    # Assign the segmentation
                    sam2_slice[sam2_struct[slice_idx] > 0] = label_id
                    
                    # Count pixels after assignment
                    pixel_count_after = np.sum(sam2_slice == label_id)
                    
                    # Log the pixel counts
                    with open(debug_log_path, 'a') as debug_file:
                        debug_file.write(f"  {structure} in SAM2: {pixel_count_after - pixel_count_before} pixels assigned\n")
            
            # Process MedSAM2 segmentation
            if structure in medsam2_data.files:
                medsam2_struct = medsam2_data[structure]
                if slice_idx < medsam2_struct.shape[0]:
                    # Count pixels before assignment
                    pixel_count_before = np.sum(medsam2_slice == label_id)
                    
                    # Assign the segmentation
                    medsam2_slice[medsam2_struct[slice_idx] > 0] = label_id
                    
                    # Count pixels after assignment
                    pixel_count_after = np.sum(medsam2_slice == label_id)
                    
                    # Log the pixel counts
                    with open(debug_log_path, 'a') as debug_file:
                        debug_file.write(f"  {structure} in MedSAM2: {pixel_count_after - pixel_count_before} pixels assigned\n")
        
        # Rotate segmentation masks based on the same rotation parameter
        sam2_slice = np.rot90(sam2_slice, rotation)
        medsam2_slice = np.rot90(medsam2_slice, rotation)
        
        # Create a figure with 4 panels
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Set the title for the figure
        fig.suptitle(f"Case: {case_name}, Slice: {slice_idx}", fontsize=14)
        
        # Panel 1: CT image with ground truth segmentation (top left)
        axes[0, 0].imshow(ct_slice, cmap='gray')
        
        # Create filtered ground truth that only shows the labels that are requested
        filtered_gt_slice = np.zeros_like(gt_slice)
        
        # Create the organ colormap
        organ_colors = create_organ_colormap()
        
        # If labels were specified, only show those labels in the ground truth visualization
        if labels is not None:
            for label_id in labels:
                filtered_gt_slice[gt_slice == label_id] = label_id
            #axes[0, 0].imshow(filtered_gt_slice, alpha=0.3)
            axes[0, 0].imshow(ct_slice, cmap='gray')
            plot_contours(axes[0, 0], filtered_gt_slice, label_dict, organ_colors)  # Zeichne die Konturen in Rot
        # Otherwise, show all labels
        else:
            axes[0, 0].imshow(gt_slice, alpha=0.3)
        
        axes[0, 0].set_title("Ground Truth Segmentation")
        axes[0, 0].axis('off')
        
        # Panel 2: CT image with ground truth bounding boxes (top right)
        axes[0, 1].imshow(ct_slice, cmap='gray')
        
        # Draw bounding boxes for each structure in the ground truth
        for label_id in unique_labels:
            if label_id == 0:  # Skip background
                continue
                
            # Skip if labels were specified and this is not one of them
            if labels is not None and label_id not in labels:
                continue
                
            
                
            # Find pixels with this label
            organ_pixels = np.where(gt_slice == label_id)
            
            # If organ is present in this slice
            if len(organ_pixels[0]) > 0:
                # Get bounding box coordinates
                min_y, max_y = np.min(organ_pixels[0]), np.max(organ_pixels[0])
                min_x, max_x = np.min(organ_pixels[1]), np.max(organ_pixels[1])
                
                # Find structure name for this label if possible
                structure_name = "Unknown"
                for struct, lid in structure_to_label.items():
                    if lid == label_id:
                        structure_name = struct
                        break
                
                # Get color for this structure/organ
                if structure_name in organ_colors:
                    color = organ_colors[structure_name]
                else:
                    # Use a default color if not found
                    color = organ_colors.get('Other', [0.7, 0.7, 0.7, 1.0])
                
                # Draw rectangle
                rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                                    linewidth=1.5, edgecolor=color, facecolor='none')
                axes[0, 1].add_patch(rect)
                
                # Add structure name label
                axes[0, 1].text(min_x, min_y - 5, structure_name, color=color, fontsize=8)
        
        axes[0, 1].set_title("Ground Truth Bounding Boxes")
        axes[0, 1].axis('off')
        
        # Panel 3: SAM2 segmentation (bottom left)
        axes[1, 0].imshow(ct_slice, cmap='gray')
        #axes[1, 0].imshow(sam2_slice, cmap=seg_cmap, alpha=0.7)
        plot_contours(axes[1, 0], sam2_slice, label_dict, organ_colors)

        axes[1, 0].set_title("SAM2 Segmentation")
        axes[1, 0].axis('off')
        
        # Panel 4: MedSAM2 segmentation (bottom right)
        axes[1, 1].imshow(ct_slice, cmap='gray')
        #axes[1, 1].imshow(medsam2_slice, cmap=seg_cmap, alpha=0.7)
        plot_contours(axes[1, 1], medsam2_slice, label_dict, organ_colors)
        axes[1, 1].set_title("MedSAM2 Segmentation")
        axes[1, 1].axis('off')
        
        # Create a custom legend
        # First, create a list of labels and colors for structures present in this slice
        present_labels = []
        present_colors = []
        
        # Add each structure that exists in this slice
        for structure, label_id in structure_to_label.items():
            if np.sum(gt_slice == label_id) > 0 or np.sum(sam2_slice == label_id) > 0 or np.sum(medsam2_slice == label_id) > 0:
                present_labels.append(structure)
                if structure in organ_colors:
                    present_colors.append(organ_colors[structure])
                else:
                    present_colors.append(organ_colors.get('Other', [0.7, 0.7, 0.7, 1.0]))
        
        # Create custom patches for the legend
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=color, label=label) 
                  for color, label in zip(present_colors, present_labels)]
        
        # Add the legend to a better position (inside the figure)
        # Place the legend in the upper right corner of the figure
        fig.legend(handles=patches, loc='upper right', 
                   bbox_to_anchor=(0.99, 0.99), frameon=True, fontsize=10)
        
        # Adjust layout to make room for the legend while keeping it compact
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        
        # Save the figure
        output_file = os.path.join(output_dir, f"{case_name}_slice{slice_idx}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        with open(debug_log_path, 'a') as debug_file:
            debug_file.write(f"Saved visualization to {output_file}\n")
    
    print(f"Saved visualization to {output_dir}\\{case_name}_slice{slice_indices[-1]}.png")

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize segmentations from different models')
    parser.add_argument('-g', '--gt_dir', type=str, required=True, help='Directory containing ground truth NPZ files')
    parser.add_argument('-s2', '--sam2_dir', type=str, required=True, help='Directory containing SAM2 segmentation NPZ files')
    parser.add_argument('-ms2', '--medsam2_dir', type=str, required=True, help='Directory containing MedSAM2 segmentation NPZ files')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save visualization images')
    parser.add_argument('-n', '--num_slices', type=int, default=5, help='Number of slices to visualize per case')
    parser.add_argument('-c', '--case', type=str, default=None, help='Specific case to visualize (if None, will process all cases)')
    parser.add_argument('-s', '--slices', type=int, nargs='+', default=None, help='Specific slice indices to visualize')
    parser.add_argument('-r', '--rotate', type=int, default=0, help='Rotation to apply to images (0=none, 1=90deg, 2=180deg, 3=270deg)')
    parser.add_argument('-l', '--labels', type=int, nargs='+', default=None, help='Specific label IDs to visualize (if None, will visualize all labels found in the data)')
    parser.add_argument('-m', '--metrics_dir', type=str, required=True, help='Directory containing metrics files')
    parser.add_argument('--clinical', action='store_true', help='Clinical mode (focus on GTV+/Target+ labels)')
    parser.add_argument('--row_size', type=int, default=2, help='Number of images per row in visualization grid')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use metrics directory for finding the input files if provided
    metrics_base_dir = args.metrics_dir
    if not os.path.exists(metrics_base_dir):
        print(f"Metrics directory not found: {metrics_base_dir}")
        return
    
    # Set up ground truth, SAM2, and MedSAM2 directories based on metrics_dir if paths not absolute
    if not os.path.isabs(args.gt_dir):
        args.gt_dir = os.path.join(metrics_base_dir, args.gt_dir)
    if not os.path.isabs(args.sam2_dir):
        args.sam2_dir = os.path.join(metrics_base_dir, args.sam2_dir)
    if not os.path.isabs(args.medsam2_dir):
        args.medsam2_dir = os.path.join(metrics_base_dir, args.medsam2_dir)
    
    # Find ground truth NPZ files
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, '*.npz')))
    
    if len(gt_files) == 0:
        print(f"No NPZ files found in {args.gt_dir}")
        return
    
    # Process specific case if provided
    if args.case is not None:
        gt_files = [f for f in gt_files if args.case in f]
        if len(gt_files) == 0:
            print(f"No NPZ files found for case {args.case}")
            return
    
    # Process each ground truth file
    for gt_file in gt_files:
        case_name = os.path.basename(gt_file).replace('.npz', '')
        print(f"Processing {case_name}...")
        
        # Find corresponding SAM2 and MedSAM2 files
        sam2_file = os.path.join(args.sam2_dir, f"{case_name}.npz")
        medsam2_file = os.path.join(args.medsam2_dir, f"{case_name}.npz")
        
        if not os.path.exists(sam2_file):
            print(f"SAM2 file not found: {sam2_file}")
            continue
        
        if not os.path.exists(medsam2_file):
            print(f"MedSAM2 file not found: {medsam2_file}")
            continue
        
        # Determine slices to visualize based on labels if provided
        slice_indices = args.slices
        
        # If no slices provided but labels are specified, find slices with those labels
        if slice_indices is None and args.labels is not None:
            # Load the ground truth data to find slices with the specified labels
            gt_data = load_npz(gt_file)
            if gt_data is not None and 'gts' in gt_data:
                gt_segmentations = gt_data['gts']
                
                # Find slices that contain any of the specified labels
                label_slices = []
                
                with open(os.path.join(args.output_dir, f"{case_name}_debug.log"), 'a') as debug_file:
                    debug_file.write(f"Looking for slices with labels: {args.labels}\n")
                
                for slice_idx in range(gt_segmentations.shape[0]):
                    has_labels = False
                    for label in args.labels:
                        if np.any(gt_segmentations[slice_idx] == label) and np.sum(gt_segmentations[slice_idx] == label) > 50:
                            has_labels = True
                            with open(os.path.join(args.output_dir, f"{case_name}_debug.log"), 'a') as debug_file:
                                debug_file.write(f"Slice {slice_idx} has {np.sum(gt_segmentations[slice_idx] == label)} pixels of label {label}\n")
                            break
                    
                    if has_labels:
                        label_slices.append(slice_idx)
                
                # If we found slices with the specified labels, use those
                if label_slices:
                    # If too many slices, select a subset
                    if len(label_slices) > 4:
                        # Select approximately evenly spaced slices
                        step = len(label_slices) // 4
                        slice_indices = [label_slices[i] for i in range(0, len(label_slices), step)[:4]]
                    else:
                        slice_indices = label_slices
                    
                    with open(os.path.join(args.output_dir, f"{case_name}_debug.log"), 'a') as debug_file:
                        debug_file.write(f"Selected slices with labels {args.labels}: {slice_indices}\n")
                else:
                    # Fall back to default behavior if no slices with the labels found
                    with open(os.path.join(args.output_dir, f"{case_name}_debug.log"), 'a') as debug_file:
                        debug_file.write(f"No slices found with labels {args.labels}\n")
        
        # Create the comparison visualization
        create_comparison_visualization(
            gt_file,
            sam2_file,
            medsam2_file,
            args.output_dir,
            slice_indices=slice_indices,
            rotation=args.rotate,
            labels=args.labels,
            clinical=args.clinical,
            row_size=args.row_size
        )

if __name__ == "__main__":
    main()
