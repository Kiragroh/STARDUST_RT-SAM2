#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DICOM to NPZ Converter with Label Overlap Detection

Converts DICOM data to NPZ format compatible with MedSAM2 training/inference.
- Combines all CT*.dcm files as image
- Uses RS*.dcm as structure set
- Labels gtv+ as primary label (ID 1)
- Labels other structures as pos+ or neg+ based on overlap with gtv+
- Optionally saves NIfTI files
- Debug mode creates visualizations
"""

import os
import sys
import glob
import argparse
import numpy as np
import pydicom
import SimpleITK as sitk
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

DEBUG = True  # Set to True for debugging output and visualizations

def get_structure_name_with_overlap_info(roi_name, gtv_mask, structure_mask):
    """
    Determine if structure overlaps with GTV+ and append suffix.
    Returns:
    - roi_name_pos+ if structure overlaps with GTV+
    - roi_name_neg+ if structure does not overlap with GTV+
    """
    # Convert masks to boolean for simpler operations
    gtv_mask_bool = gtv_mask > 0
    structure_mask_bool = structure_mask > 0
    
    # Check if there's any overlap
    overlap = np.logical_and(gtv_mask_bool, structure_mask_bool)
    
    if not np.any(overlap):
        # No overlap with GTV+
        return f"{roi_name}_neg+"
    else:
        # Structure overlaps with GTV+
        return f"{roi_name}_pos+"

def create_color_map(num_labels):
    """
    Create a colormap for visualizing different labels.
    """
    # Start with some basic colors (red = 1 for GTV+)
    colors = [
        [0, 0, 0],       # 0: Background (black)
        [255, 0, 0],     # 1: GTV+ (red)
    ]
    
    # Generate more colors if needed
    import colorsys
    if num_labels > len(colors):
        for i in range(len(colors), num_labels):
            # Generate colors using HSV space
            h = (i - 1) / (num_labels - 1) * 0.8  # hue (avoid red which is reserved)
            s = 0.8                              # saturation
            v = 0.9                              # value
            # Convert to RGB and scale to 0-255
            rgb = colorsys.hsv_to_rgb(h, s, v)
            colors.append([int(c * 255) for c in rgb])
    
    return np.array(colors, dtype=np.uint8)

def visualize_slices(image_data, label_data, label_dict, output_dir, case_name, num_slices=5):
    """
    Create visualizations of sample slices with labels.
    
    Args:
        image_data: 3D CT image array
        label_data: 3D label array (segmentation)
        label_dict: Dictionary mapping label IDs to names
        output_dir: Directory to save visualizations
        case_name: Case name for filenames
        num_slices: Number of slices to visualize
    """
    # Create output directory for debug visualizations
    debug_dir = os.path.join(output_dir, "debug_viz")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Get unique labels
    unique_labels = np.unique(label_data)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    num_labels = len(label_dict)
    cmap = create_color_map(num_labels)
    
    # Find slices that contain GTV+ (label 1) and Target+ (label 2)
    has_gtv = 1 in unique_labels
    has_target = 2 in unique_labels
    
    # Count voxels for each structure
    label_voxel_counts = {}
    for label_id in unique_labels:
        label_voxel_counts[label_id] = np.sum(label_data == label_id)
        label_name = label_dict.get(label_id, f"Unknown_{label_id}")
        print(f"Label {label_id} ({label_name}): {label_voxel_counts[label_id]} voxels")
    
    slices_with_gtv = []
    slices_with_target = []
    slices_with_labels = {}
    
    # Identify slices with specific structures
    for z in range(label_data.shape[0]):
        slice_labels = np.unique(label_data[z])
        slice_labels = slice_labels[slice_labels > 0]  # Exclude background
        
        if len(slice_labels) > 0:
            slices_with_labels[z] = slice_labels
            
            if 1 in slice_labels:
                slices_with_gtv.append(z)
            elif 2 in slice_labels:
                slices_with_target.append(z)
    
    # Select slices to visualize
    slices_to_viz = []
    
    if has_gtv and len(slices_with_gtv) > 0:
        print(f"Found {len(slices_with_gtv)} slices with GTV+")
        # Show at least 3 slices with GTV+ if possible
        num_gtv_slices = min(3, len(slices_with_gtv))
        
        # Evenly distributed slices with GTV+
        selected_indices = np.linspace(0, len(slices_with_gtv) - 1, num_gtv_slices, dtype=int)
        for idx in selected_indices:
            z = slices_with_gtv[idx]
            slices_to_viz.append(z)
            
            # Also add a slice above and below if possible
            if z > 0 and z-1 not in slices_to_viz:
                slices_to_viz.append(z-1)
            if z < label_data.shape[0] - 1 and z+1 not in slices_to_viz:
                slices_to_viz.append(z+1)
    elif has_target and len(slices_with_target) > 0:
        print(f"No GTV+ slices found. Using {len(slices_with_target)} slices with Target+ instead.")
        # Show a few slices with Target+
        num_target_slices = min(3, len(slices_with_target))
        
        # Evenly distributed slices with Target+
        selected_indices = np.linspace(0, len(slices_with_target) - 1, num_target_slices, dtype=int)
        for idx in selected_indices:
            slices_to_viz.append(slices_with_target[idx])
    else:
        print("No GTV+ or Target+ found. Selecting slices with the most structure diversity.")
        # If neither GTV+ nor Target+ found, select slices with the most different structures
        slices_by_diversity = sorted(slices_with_labels.keys(), 
                                    key=lambda z: len(slices_with_labels[z]), 
                                    reverse=True)
        
        slices_to_viz = slices_by_diversity[:num_slices]
    
    # Ensure we have at most the requested number of slices
    slices_to_viz = sorted(slices_to_viz)
    if len(slices_to_viz) > num_slices:
        # Keep a few evenly spaced slices
        slices_to_viz = [slices_to_viz[i] for i in 
                         np.linspace(0, len(slices_to_viz) - 1, num_slices, dtype=int)]
    
    # Visualize selected slices
    print(f"Visualizing slices: {slices_to_viz}")
    
    # Generate a legend list for consistent colors
    legend_items = []
    for label_id in range(1, num_labels):
        if label_id in label_dict:
            legend_items.append((label_id, label_dict[label_id], cmap[label_id]))
    
    for z in slices_to_viz:
        # Set up the figure
        plt.figure(figsize=(14, 7))
        
        # Get original slice and label
        ct_slice = image_data[z]
        label_slice = label_data[z]
        
        # Check what's in this slice
        has_gtv_in_slice = 1 in np.unique(label_slice)
        has_target_in_slice = 2 in np.unique(label_slice)
        
        # Plot the original image
        plt.subplot(1, 2, 1)
        plt.imshow(ct_slice, cmap='gray')
        
        # Adjust title based on content
        title = f'Slice {z}: CT Image'
        if has_gtv_in_slice:
            title += " (with GTV+)"
        elif has_target_in_slice:
            title += " (with Target+)"
        plt.title(title)
        plt.axis('off')
        
        # Plot filled regions
        plt.subplot(1, 2, 2)
        plt.imshow(ct_slice, cmap='gray')
        
        # Create a colored mask overlay
        mask_rgb = np.zeros((*ct_slice.shape, 4))  # RGBA
        
        # Draw filled regions for each label
        for i in range(1, num_labels):
            if i in np.unique(label_slice):
                # Get the binary mask for this label
                mask = label_slice == i
                if np.any(mask):
                    # Get color and convert to RGBA with alpha
                    color_rgb = np.array(cmap[i]) / 255
                    # Use alpha of 0.5 for most structures, but higher for GTV+ and Target+
                    alpha = 0.7 if i <= 2 else 0.5
                    # Apply color where mask is True
                    mask_color = np.zeros((*mask.shape, 4))
                    mask_color[mask] = [*color_rgb, alpha]
                    
                    # Alpha-blend with existing overlay (using higher alpha for GTVs)
                    existing_alpha = mask_rgb[..., 3:4]
                    new_alpha = mask_color[..., 3:4]
                    blend_alpha = existing_alpha + new_alpha * (1 - existing_alpha)
                    
                    # Avoid division by zero
                    safe_blend_alpha = np.where(blend_alpha > 0, blend_alpha, 1)
                    
                    # Blend colors
                    mask_rgb[..., :3] = (
                        mask_rgb[..., :3] * existing_alpha + 
                        mask_color[..., :3] * new_alpha * (1 - existing_alpha)
                    ) / safe_blend_alpha
                    
                    # Update alpha channel
                    mask_rgb[..., 3:4] = blend_alpha
        
        # Overlay the colored mask on the CT slice
        plt.imshow(mask_rgb)
        
        # Adjust title based on content
        title = f'Slice {z}: Segmentation Overlay'
        if has_gtv_in_slice:
            title += " (with GTV+)"
        elif has_target_in_slice:
            title += " (with Target+)"
        plt.title(title)
        plt.axis('off')
        
        # Add legend
        legend_patches = []
        for label_id, label_name, color in legend_items:
            from matplotlib.patches import Patch
            legend_patches.append(Patch(color=[c/255 for c in color], label=f"{label_id}: {label_name}"))
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save figure
        out_path = os.path.join(output_dir, "debug_viz", f"{case_name}_slice_{z}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    
    print(f"Saved {len(slices_to_viz)} debug visualizations to {os.path.join(output_dir, 'debug_viz')}")

def preprocess_ct_image(image_array, window_level=40, window_width=400):
    """
    Preprocess CT image by applying window/level and normalizing.
    
    Args:
        image_array: The raw CT array
        window_level: Center of window (default 40 HU)
        window_width: Width of window (default 400 HU)
    
    Returns:
        Preprocessed CT array (uint8)
    """
    # Apply window/level
    lower_bound = window_level - window_width / 2
    upper_bound = window_level + window_width / 2
    windowed = np.clip(image_array, lower_bound, upper_bound)
    
    # Normalize to 0-255
    normalized = ((windowed - lower_bound) / window_width) * 255.0
    
    return np.uint8(normalized)

def convert_dicom_to_npz(dicom_dir, output_dir, case_name, save_nii=False):
    """
    Convert DICOM series to NPZ format with custom labeling.
    
    Args:
        dicom_dir: Directory containing DICOM files
        output_dir: Directory to save output files
        case_name: Name for the output files
        save_nii: Whether to save NIfTI files (images and labels)
    
    Returns:
        Path to saved NPZ file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Automatische Erkennung des Dateityps (CT oder MR)
    ct_files = sorted(glob.glob(os.path.join(dicom_dir, "CT*.dcm")))
    mr_files = sorted(glob.glob(os.path.join(dicom_dir, "MR*.dcm")))

    if mr_files:
        modality = "MR"
        image_files = mr_files
    elif ct_files:
        modality = "CT"
        image_files = ct_files
    else:
        print(f"No CT or MR DICOM files found in {dicom_dir}")
        return None

    print(f"Found {len(image_files)} {modality} DICOM files")
    
    # Find structure set file (starting with "RS")
    rs_files = sorted(glob.glob(os.path.join(dicom_dir, "RS*.dcm")))
    if not rs_files:
        print("No structure set (RS) file found")
        structure_file = None
    else:
        structure_file = rs_files[0]
        print(f"Found structure set file: {os.path.basename(structure_file)}")
    
    # Load CT image series
    print("Reading DICOM series...")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    
    # Filter to only include files starting with CT if needed
    if len(image_files) != len(dicom_names):
        print(f"Warning: Found {len(dicom_names)} DICOM files but only {len(image_files)} start with 'CT'")
        # Just use the CT files we found instead of all DICOM files
        reader.SetFileNames(image_files)
    else:
        reader.SetFileNames(dicom_names)
    
    try:
        image = reader.Execute()
        # Get image array and spacing
        image_array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        
        print(f"Image loaded: shape={image_array.shape}, spacing={spacing}, origin={image.GetOrigin()}, direction={image.GetDirection()}")
        
        if modality == "CT":
            print("Applying CT preprocessing...")
            WINDOW_LEVEL = 40
            WINDOW_WIDTH = 400
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_array_preprocessed = np.clip(image_array, lower_bound, upper_bound)
            image_array_preprocessed = (
                (image_array_preprocessed - np.min(image_array_preprocessed))
                / (np.max(image_array_preprocessed) - np.min(image_array_preprocessed))
                * 255.0
            )
        else:  # modality == "MR"
            print("Applying MR preprocessing...")
            lower_bound, upper_bound = np.percentile(image_array[image_array > 0], [0.5, 99.5])
            image_array_preprocessed = np.clip(image_array, lower_bound, upper_bound)
            image_array_preprocessed = (
                (image_array_preprocessed - np.min(image_array_preprocessed))
                / (np.max(image_array_preprocessed) - np.min(image_array_preprocessed))
                * 255.0
            )
            image_array_preprocessed[image_array == 0] = 0

        image_array_preprocessed = np.uint8(image_array_preprocessed)
        
        # Create output dictionary
        output_dict = {
            'imgs': image_array_preprocessed,
            'spacing': np.array(spacing),
            'origin': np.array(image.GetOrigin()),
            'direction': np.array(image.GetDirection())
        }
        
        # Process structure set if available
        if structure_file and os.path.exists(structure_file):
            print(f"Processing structure set: {os.path.basename(structure_file)}")
            try:
                # Read the structure set file
                ds = pydicom.dcmread(structure_file)
                
                # Dictionary to store all structure masks
                structure_masks = {}
                label_dict = {0: "background"}
                
                # Create separate binary masks for each structure
                # We'll combine them later based on priority
                
                # Try to find ROI contour data
                if hasattr(ds, 'ROIContourSequence') and hasattr(ds, 'StructureSetROISequence'):
                    # Get ROI names from StructureSetROISequence
                    roi_names = {}
                    for roi in ds.StructureSetROISequence:
                        roi_names[roi.ROINumber] = roi.ROIName
                    
                    # Print all ROI names for debugging
                    print("Available ROIs in structure set:")
                    for roi_id, roi_name in roi_names.items():
                        print(f"  ROI ID {roi_id}: {roi_name}")
                    
                    # Search for ROI that might be GTV+ with different naming conventions
                    possible_gtv_names = ["gtv+", "gtv +", "GTV+", "GTV +", "GTV", "gtv", "GTV_Plus", "gtv_plus"]
                    possible_target_names = ["target+", "target +", "TARGET+", "TARGET +", "TARGET", "target", "Target+", "Target +"]
                    
                    print("Searching for GTV+ and Target+ using these possible names:")
                    print(f"  GTV+ candidates: {possible_gtv_names}")
                    print(f"  Target+ candidates: {possible_target_names}")
                    
                    # First, create individual binary masks for all structures
                    all_masks = {}  # Dictionary to store binary masks
                    structure_info = {}  # Dictionary to store structure information
                    label_to_roi_name = {}  # Mapping of label IDs to ROI names
                    
                    print("STEP 1: Creating individual binary masks for all structures")
                    # Process each ROI to create a binary mask
                    for roi in ds.ROIContourSequence:
                        roi_number = roi.ReferencedROINumber
                        roi_name = roi_names.get(roi_number, f"ROI_{roi_number}")
                        roi_name_normalized = roi_name.lower().replace(" ", "").replace("_", "").replace("-", "")
                        
                        print(f"Processing ROI: '{roi_name}' (ID: {roi_number})")
                        
                        # Create a binary mask for this ROI
                        roi_mask = np.zeros_like(image_array, dtype=np.uint8)
                        has_valid_contours = False
                        
                        # Process contours
                        if hasattr(roi, 'ContourSequence'):
                            for contour in roi.ContourSequence:
                                if hasattr(contour, 'ContourData'):
                                    contour_data = contour.ContourData
                                    num_points = len(contour_data) // 3
                                    
                                    if num_points == 0:
                                        continue
                                    
                                    has_valid_contours = True
                                    # Reshape to (N, 3)
                                    points = np.array(contour_data).reshape(num_points, 3)
                                    
                                    # Get slice index (z coordinate)
                                    #z_coord = points[0, 2]
                                    
                                    # Find the closest slice in the CT volume
                                    #z_index = int((z_coord - image.GetOrigin()[2]) / image.GetSpacing()[2])
                                    _, _, z_index = image.TransformPhysicalPointToIndex(tuple(points[0]))

                                    
                                    # Ensure z_index is within bounds
                                    if z_index < 0 or z_index >= image_array.shape[0]:
                                        continue
                                    
                                    # Debug-Ausgabe zur Kontrolle der Image-Eigenschaften
                                    if DEBUG and roi_number == 1 and contour == roi.ContourSequence[0]:
                                        print("Image Origin:", image.GetOrigin())
                                        print("Image Direction:", image.GetDirection())
                                        print("Spacing:", image.GetSpacing())
                                        sample_contour_point = contour.ContourData[:3]
                                        sample_idx = image.TransformPhysicalPointToIndex(tuple(sample_contour_point))
                                        print("Beispiel-Konturpunkt:", sample_contour_point, "=> Index:", sample_idx)

                                    # Use SimpleITK for precise transformation from physical to image indices
                                    pixel_points = []
                                    for point in points:
                                        try:
                                            idx = image.TransformPhysicalPointToIndex((point[0], point[1], point[2]))
                                            px, py, _ = idx
                                            pixel_points.append([px, py])
                                        except Exception as e:
                                            print(f"Error transforming point {point}: {e}")
                                    # Convert contour points to pixel coordinates (works only for CT)
                                    #pixel_points = []
                                    #for point in points:
                                        #px = int((point[0] - ct_image.GetOrigin()[0]) / ct_image.GetSpacing()[0])
                                        #py = int((point[1] - ct_image.GetOrigin()[1]) / ct_image.GetSpacing()[1])
                                        #pixel_points.append([px, py])
                                    
                                    # Convert to numpy array
                                    pixel_points = np.array(pixel_points, dtype=np.int32)
                                    
                                    # Draw the contour on the slice
                                    if len(pixel_points) > 2:  # Need at least 3 points for a polygon
                                        # Create a mask for this contour
                                        mask = np.zeros((image_array.shape[1], image_array.shape[2]), dtype=np.uint8)
                                        cv2.fillPoly(mask, [pixel_points], 1)
                                        
                                        # Add to the ROI mask
                                        roi_mask[z_index] = np.logical_or(roi_mask[z_index], mask)
                        
                        # If ROI has valid contours and sufficient volume, add it to our collection
                        if has_valid_contours and np.sum(roi_mask) > 10:  # Minimum size threshold (lowered to ensure small GTVs are included)
                            # Determine if this is GTV+ or Target+
                            is_gtv_plus = any(gtv_name.lower().replace(" ", "").replace("_", "").replace("-", "") in roi_name_normalized 
                                             for gtv_name in possible_gtv_names)
                            
                            is_target_plus = any(target_name.lower().replace(" ", "").replace("_", "").replace("-", "") in roi_name_normalized
                                               for target_name in possible_target_names)
                            
                            # Store the mask and metadata
                            all_masks[roi_number] = roi_mask
                            structure_info[roi_number] = {
                                'name': roi_name,
                                'normalized_name': roi_name_normalized,
                                'is_gtv_plus': is_gtv_plus,
                                'is_target_plus': is_target_plus,
                                'voxel_count': np.sum(roi_mask)
                            }
                            
                            print(f"  -> Created mask for '{roi_name}', voxel count: {np.sum(roi_mask)}")
                            if is_gtv_plus:
                                print(f"  -> Identified as GTV+")
                            if is_target_plus:
                                print(f"  -> Identified as Target+")
                    
                    # STEP 2: Identify GTV+ and Target+ structures
                    gtv_plus_id = None
                    target_plus_id = None
                    
                    print("\nSTEP 2: Identifying GTV+ and Target+ structures")
                    
                    # First, look for explicit GTV+ matches
                    for roi_id, info in structure_info.items():
                        if info['is_gtv_plus']:
                            if gtv_plus_id is None or np.sum(all_masks[roi_id]) > np.sum(all_masks[gtv_plus_id]):
                                gtv_plus_id = roi_id
                                print(f"Found GTV+ structure: '{info['name']}' (ID: {roi_id})")
                        
                        if info['is_target_plus'] and not info['is_gtv_plus']:  # Avoid marking the same structure as both
                            if target_plus_id is None or np.sum(all_masks[roi_id]) > np.sum(all_masks[target_plus_id]):
                                target_plus_id = roi_id
                                print(f"Found Target+ structure: '{info['name']}' (ID: {roi_id})")
                    
                    # If no GTV+ was found, check if any ROI has "gtv" in the name
                    if gtv_plus_id is None:
                        print("No explicit GTV+ found. Looking for ROIs with 'gtv' in the name...")
                        for roi_id, info in structure_info.items():
                            if "gtv" in info['normalized_name'] and not info['is_target_plus']:
                                gtv_plus_id = roi_id
                                print(f"Using '{info['name']}' (ID: {roi_id}) as GTV+")
                                break
                    
                    # STEP 3: Create the final segmentation with priority hierarchy
                    print("\nSTEP 3: Creating final segmentation with priority hierarchy")
                    
                    # Create empty segmentation
                    segmentation = np.zeros_like(image_array, dtype=np.uint8)
                    
                    # Sort other ROIs by volume size (largest first)
                    other_rois = [(roi_id, info) for roi_id, info in structure_info.items() 
                                 if roi_id != gtv_plus_id and roi_id != target_plus_id]
                    
                    # Sort by voxel count in descending order (largest first)
                    other_rois.sort(key=lambda x: x[1]['voxel_count'], reverse=True)
                    
                    print("Adding structures in order of volume (largest first):")
                    
                    # Store original masks to compare with final segmentation
                    original_masks = {}
                    
                    # Process other ROIs first (lowest priority), largest to smallest
                    label_id = 3  # Start from 3 (1 is for GTV+, 2 is for Target+)
                    for roi_id, info in other_rois:
                        roi_mask = all_masks[roi_id]
                        roi_name = info['name']
                        
                        # Store original mask for later comparison
                        original_masks[label_id] = roi_mask.copy()
                        
                        # Add suffix indicating overlap with GTV+ if applicable
                        if gtv_plus_id is not None:
                            roi_name_with_suffix = get_structure_name_with_overlap_info(
                                roi_name, all_masks[gtv_plus_id], roi_mask)
                        else:
                            roi_name_with_suffix = roi_name
                        
                        # Assign label and update segmentation
                        segmentation[roi_mask > 0] = label_id
                        label_dict[label_id] = roi_name_with_suffix
                        label_to_roi_name[label_id] = roi_name
                        
                        print(f"  -> Added structure {label_id}: {roi_name_with_suffix} ({info['voxel_count']} voxels)")
                        label_id += 1
                    
                    # Store Target+ mask for later comparison if it exists
                    if target_plus_id is not None:
                        target_mask = all_masks[target_plus_id]
                        original_masks[2] = target_mask.copy()
                    
                    # Then add Target+ (higher priority than other ROIs)
                    if target_plus_id is not None:
                        target_mask = all_masks[target_plus_id]
                        segmentation[target_mask > 0] = 2  # Target+ is always label 2
                        label_dict[2] = "target+"
                        label_to_roi_name[2] = structure_info[target_plus_id]['name']
                        print(f"  -> Added Target+ as label 2: {structure_info[target_plus_id]['name']}")
                        print(f"  -> Target+ voxel count: {np.sum(target_mask)}")
                    
                    # Store GTV+ mask for later comparison if it exists
                    if gtv_plus_id is not None:
                        gtv_mask = all_masks[gtv_plus_id]
                        original_masks[1] = gtv_mask.copy()
                    
                    # Finally add GTV+ (highest priority - overwrites anything else)
                    if gtv_plus_id is not None:
                        gtv_mask = all_masks[gtv_plus_id]
                        segmentation[gtv_mask > 0] = 1  # GTV+ is always label 1
                        label_dict[1] = "gtv+"
                        label_to_roi_name[1] = structure_info[gtv_plus_id]['name']
                        print(f"  -> Added GTV+ as label 1: {structure_info[gtv_plus_id]['name']}")
                        print(f"  -> GTV+ voxel count: {np.sum(gtv_mask)}")
                    
                    # Check which structures have disappeared in the final segmentation
                    print("\nSTEP 4: Checking for completely overlapped structures")
                    for label_id, original_mask in original_masks.items():
                        # Count how many voxels of this structure remain in the final segmentation
                        remaining_voxels = np.sum(segmentation == label_id)
                        total_voxels = np.sum(original_mask)
                        
                        # Skip GTV+ and tiny structures
                        if label_id == 1 or total_voxels < 10:
                            continue
                        
                        if remaining_voxels == 0:
                            # Structure completely disappeared
                            print(f"  -> Structure {label_id} ({label_dict[label_id]}) is completely overlapped!")
                            # Replace the + with - at the end of the label name
                            if label_dict[label_id].endswith('+'):
                                label_dict[label_id] = label_dict[label_id][:-1] + '-'
                            else:
                                label_dict[label_id] = label_dict[label_id] + '-'
                        elif remaining_voxels < total_voxels * 0.25:  # Less than 25% remaining
                            # Structure significantly disappeared
                            print(f"  -> Structure {label_id} ({label_dict[label_id]}) has only {remaining_voxels}/{total_voxels} voxels remaining")
                    
                    # Save the individual structure masks for debugging
                    structure_masks = {label_to_roi_name.get(label_id, f"label_{label_id}"): 
                                    (segmentation == label_id).astype(np.uint8) 
                                    for label_id in label_dict if label_id > 0}
                    
                    # Check if GTV+ exists in final segmentation
                    if 1 in label_dict:
                        print(f"GTV+ in final segmentation: {np.sum(segmentation == 1)} voxels")
                    else:
                        print("No GTV+ in final segmentation")
                    
                    # Print overlap information
                    if gtv_plus_id is not None and target_plus_id is not None:
                        gtv_mask = all_masks[gtv_plus_id]
                        target_mask = all_masks[target_plus_id]
                        overlap = np.logical_and(gtv_mask, target_mask)
                        overlap_count = np.sum(overlap)
                        print(f"Overlap between GTV+ and Target+: {overlap_count} voxels")
                
                # Save labels to output dictionary
                output_dict['gts'] = segmentation
                output_dict['labels'] = label_dict
                
                print("Label dictionary:")
                for label_id, label_name in label_dict.items():
                    print(f"  {label_id}: {label_name}")
            
            except Exception as e:
                print(f"Error processing structure set: {e}")
                import traceback
                traceback.print_exc()
                # Create empty segmentation
                output_dict['gts'] = np.zeros_like(image_array, dtype=np.uint8)
                output_dict['labels'] = {0: "background"}
        else:
            # Create empty segmentation if no structure set
            output_dict['gts'] = np.zeros_like(image_array, dtype=np.uint8)
            output_dict['labels'] = {0: "background"}
        
        # Save as NPZ
        npz_path = os.path.join(output_dir, f"{case_name}.npz")
        np.savez_compressed(npz_path, **output_dict)
        print(f"Saved NPZ file to {npz_path}")
        # Direkt vor np.savez_compressed
        print("Final keys in NPZ:", output_dict.keys())
        
        # Create debug visualizations if enabled
        if DEBUG and 'gts' in output_dict and np.any(output_dict['gts'] > 0):
            visualize_slices(
                output_dict['imgs'], 
                output_dict['gts'], 
                output_dict['labels'], 
                output_dir, 
                case_name
            )
        
        # Save as NIfTI if requested
        if save_nii:
            # Save preprocessed image
            img_sitk = sitk.GetImageFromArray(output_dict['imgs'])
            img_sitk.SetSpacing(spacing)
            img_nii_path = os.path.join(output_dir, f"{case_name}_img.nii.gz")
            sitk.WriteImage(img_sitk, img_nii_path)
            print(f"Saved NIfTI image to {img_nii_path}")
            
            # Save segmentation if it contains labels
            if 'gts' in output_dict and np.any(output_dict['gts'] > 0):
                gt_sitk = sitk.GetImageFromArray(output_dict['gts'])
                gt_sitk.SetSpacing(spacing)
                gt_nii_path = os.path.join(output_dir, f"{case_name}_gt.nii.gz")
                sitk.WriteImage(gt_sitk, gt_nii_path)
                print(f"Saved NIfTI segmentation to {gt_nii_path}")
        
        return npz_path
        
    except Exception as e:
        print(f"Error converting DICOM to NPZ: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert DICOM data to NPZ format for MedSAM2")
    parser.add_argument("--dicom_dir", type=str, default="./data/dicomtest",
                        help="Directory containing DICOM files")
    parser.add_argument("--output_dir", type=str, default="./data/npz_output",
                        help="Output directory for NPZ files")
    parser.add_argument("--case_name", type=str, default="CT_Patient",
                        help="Case name for output files")
    parser.add_argument("--save-nii", action="store_true",
                        help="Save the image and ground truth as NIfTI files")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert DICOM to NPZ
    print(f"Converting DICOM data from {args.dicom_dir} to NPZ format...")
    npz_path = convert_dicom_to_npz(
        args.dicom_dir, 
        args.output_dir, 
        args.case_name,
        args.save_nii
    )
    
    if npz_path:
        print(f"Conversion complete. NPZ file saved to {npz_path}")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main()
