import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import cv2
from PIL import Image
from os.path import join, basename, exists
import matplotlib.pyplot as plt
import random
import json

# Laden der Label-Namen aus der GUI-Konfiguration
def load_label_names():
    if exists("sam_gui_settings.json"):
        try:
            with open("sam_gui_settings.json", 'r') as f:
                settings = json.load(f)
                label_names = {}
                for k, v in settings.get("label_names", {}).items():
                    label_names[int(k)] = v
                return label_names
        except Exception as e:
            print(f"Fehler beim Laden der Label-Namen aus der GUI: {e}")
    
    # Fallback zu einem generischen Format
    return None

# Label-Namen für die Segmentierungen
label_names = load_label_names()

from sam2.build_sam import build_sam2_video_predictor_npz, build_sam2

def normalize_img(img):
    """Normalize image to [0, 1] range"""
    img = img.astype(np.float32)
    if img.max() - img.min() > 0:
        img = (img - img.min()) / (img.max() - img.min())
    return img

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array

def get_bbox_from_mask(mask):
    """Get bounding box from binary mask"""
    if not np.any(mask):
        return None
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    return [y1, x1, y2, x2]

def scale_box(box, orig_size, target_size):
    """Scale box coordinates from original size to target size"""
    y1, x1, y2, x2 = box
    scale = target_size / orig_size
    return [int(y1 * scale), int(x1 * scale), int(y2 * scale), int(x2 * scale)]

def main():
    parser = argparse.ArgumentParser(
        description='3D-Inferenz für MedSAM2 mit Video-Predictor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Daten- und Modellparameter
    parser.add_argument('--data_root', type=str, required=True,
                      help='Root directory mit NPZ-Dateien')
    parser.add_argument('--gts_path', type=str, required=True,
                      help='Path to the ground truth segmentations')
    parser.add_argument('--ckpt_path', type=str, required=True,
                      help='Pfad zum SAM2-Basis-Checkpoint')
    parser.add_argument('--cfg', type=str, default="sam2_hiera_t.yaml",
                      help='Model config file')
    parser.add_argument('--medsam2_checkpoint', type=str, required=True,
                      help='Pfad zum feinjustierten MedSAM2-Checkpoint')
    parser.add_argument('--pred_save_dir', type=str, default='./segs_3d',
                      help='Ausgabeverzeichnis für 3D-Segs')
    parser.add_argument('--save_nii', action='store_true',
                      help='Save the predictions as NIfTI files')
    parser.add_argument('--include_ct', action='store_true',
                      help='Include CT data in the output file')
    parser.add_argument('--label', type=str, default=None,
                      help='Comma-separated list of labels to process (e.g., "1,3,5")')
    parser.add_argument('--generate_preview', action='store_true',
                      help='Generate a preview image of the segmentation')
    
    args = parser.parse_args()
    
    # Verzeichnisse erstellen
    os.makedirs(args.pred_save_dir, exist_ok=True)
    
    # Load both predictors - one for 2D and one for 3D
    print(f"Lade SAM2 Video-Predictor...")
    predictor_3d = build_sam2_video_predictor_npz(
        config_file=args.cfg,
        ckpt_path=args.ckpt_path
    )
    
    predictor_2d = build_sam2(args.cfg, args.ckpt_path)
    
    # MedSAM2 Checkpoint laden
    print(f"Lade MedSAM2 Checkpoint von: {args.medsam2_checkpoint}")
    checkpoint = torch.load(args.medsam2_checkpoint, map_location="cpu")
    
    # Print checkpoint keys and model info for debugging
    print(f"Checkpoint keys: {checkpoint.keys()}")
    print(f"Model state dict contains {len(checkpoint['model_state_dict'])} parameters")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Checkpoint iteration: {checkpoint.get('iteration', 'N/A')}")
    
    # Load the checkpoint into the models
    predictor_3d.load_state_dict(checkpoint["model_state_dict"], strict=False)
    predictor_2d.load_state_dict(checkpoint["model_state_dict"], strict=False)
    predictor_3d.to("cuda")
    predictor_2d.to("cuda")
    
    # NPZ-Dateien finden
    data_root = Path(args.data_root)
    npz_files = list(data_root.glob("*.npz"))
    
    print(f"Gefunden: {len(npz_files)} NPZ-Dateien")
    
    # Define the label dictionary for FLARE22 dataset
    label_dict = {
        1: 'Liver',
        2: 'Right Kidney',
        3: 'Spleen',
        4: 'Pancreas',
        5: 'Aorta',
        6: 'Inferior Vena Cava',  # IVC
        7: 'Right Adrenal Gland',  # RAG
        8: 'Left Adrenal Gland',  # LAG
        9: 'Gallbladder',
        10: 'Esophagus',
        11: 'Stomach',
        13: 'Left Kidney'
    }
    
    # Verarbeite jede NPZ-Datei
    for npz_file in tqdm(npz_files, desc="Verarbeite Volumen"):
        print(f"\nVerarbeite: {npz_file.name}")
        
        # Lade 3D-Volumen und ground truth
        npz_name = npz_file.name
        img_data = np.load(npz_file)
        img_3D = img_data['imgs']
        
        # Load ground truth
        gt_path = os.path.join(args.gts_path, npz_name)
        gt_data = np.load(gt_path)
        gts = gt_data['gts']
        spacing = gt_data.get('spacing', [1.0, 1.0, 1.0])
        
        # Get unique labels in the ground truth
        unique_labels = np.unique(gts)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background
        
        # Filter labels if specified
        if args.label:
            selected_labels = [int(label) for label in args.label.split(',')]
            unique_labels = np.array([label for label in unique_labels if label in selected_labels])
            print(f"Processing only labels: {unique_labels}")
        
        # Initialize 3D segmentation volume
        segs_3D = np.zeros_like(gts, dtype=np.uint16)
        
        # Prepare video dimensions
        video_height = img_3D.shape[1]
        video_width = img_3D.shape[2]
        
        # Process each organ separately
        for label in unique_labels:
            print(f"Processing label {label} ({label_dict[label]})")
            
            # Create binary mask for this organ
            organ_mask = (gts == label)
            
            # Find slices where this organ is present
            organ_slices = []
            for z in range(organ_mask.shape[0]):
                if np.any(organ_mask[z]):
                    organ_slices.append(z)
            
            if not organ_slices:
                print(f"No slices found for label {label}")
                continue
            
            # Find the middle slice for this organ
            middle_slice_idx = organ_slices[len(organ_slices) // 2]
            print(f"Middle slice for label {label}: {middle_slice_idx}")
            
            # Get the ground truth mask for the middle slice
            middle_slice_mask = organ_mask[middle_slice_idx].astype(np.uint8)
            
            # Get the bounding box for the middle slice mask
            bbox = get_bbox_from_mask(middle_slice_mask)
            if bbox is None:
                print(f"No bounding box found for label {label} in middle slice")
                continue
            
            # Scale the bounding box to 1024x1024
            H, W = middle_slice_mask.shape
            bbox_1024 = scale_box(bbox, H, 1024)
            
            # Create a mask prompt from the middle slice
            mask_prompt = np.zeros((1024, 1024), dtype=np.uint8)
            mask_scaled = cv2.resize(middle_slice_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            mask_prompt[mask_scaled > 0] = 1
            
            # Convert mask_prompt to tensor and move to GPU
            mask_prompt = torch.from_numpy(mask_prompt).bool().cuda()
            
            # Prepare the image sequence for this organ
            z_min = min(organ_slices)
            z_max = max(organ_slices)
            img_sequence = img_3D[z_min:z_max+1]
            
            # Resize and normalize the image sequence
            img_resized = resize_grayscale_to_rgb_and_resize(img_sequence, 1024)  # shape: (D, 3, 1024, 1024)
            img_resized = img_resized / 255.0
            img_resized = torch.from_numpy(img_resized).cuda()
            
            # Normalize with ImageNet stats
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
            img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
            img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
            img_resized -= img_mean
            img_resized /= img_std
            
            # Initialize the predictor with the image sequence
            inference_state = predictor_3d.init_state(img_resized, video_height, video_width)
            
            # Add mask at the middle slice
            middle_frame_idx = middle_slice_idx - z_min
            frame_idx, object_ids, masks = predictor_3d.add_new_mask(
                inference_state, 
                frame_idx=middle_frame_idx, 
                obj_id=1, 
                mask=mask_prompt
            )
            
            # Save the segmentation for the middle slice
            mask = (masks[0] > -0.5).cpu().numpy()[0]
            segs_3D[middle_slice_idx][mask] = label
            
            # Propagate forward
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor_3d.propagate_in_video(inference_state):
                mask = (out_mask_logits[0] > -0.5).cpu().numpy()[0]
                z_idx = z_min + out_frame_idx
                if z_idx < segs_3D.shape[0]:  # Ensure we don't go out of bounds
                    segs_3D[z_idx] = np.where(mask, label, segs_3D[z_idx])
            
            # Reset state and propagate backward
            predictor_3d.reset_state(inference_state)
            frame_idx, object_ids, masks = predictor_3d.add_new_mask(
                inference_state, 
                frame_idx=middle_frame_idx, 
                obj_id=1, 
                mask=mask_prompt
            )
            
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor_3d.propagate_in_video(inference_state, reverse=True):
                mask = (out_mask_logits[0] > -0.5).cpu().numpy()[0]
                z_idx = z_min + out_frame_idx
                if z_idx < segs_3D.shape[0]:  # Ensure we don't go out of bounds
                    segs_3D[z_idx] = np.where(mask, label, segs_3D[z_idx])
        
        print(np.unique(segs_3D))
        
        # Save segmentation results
        save_dict = {}
        for i in np.unique(segs_3D):
            if i > 0:  # Ignore background
                mask = (segs_3D == i).astype(np.uint8)
                
                # Verwende den Organ-Namen aus den GUI-Einstellungen, wenn verfügbar
                if label_names and i in label_names:
                    organ_name = label_names[i]
                    save_dict[organ_name] = mask
                else:
                    # Fallback zu generischem Namen
                    save_dict[f'Organ_{i}'] = mask
        
        if args.include_ct:
            # Include the CT data in the output file
            save_dict['imgs'] = img_3D
        
        np.savez_compressed(join(args.pred_save_dir, npz_name), **save_dict)
        
        # Save visualization if requested
        if args.generate_preview:
            # Wähle einen Slice, der tatsächlich Segmentierungen enthält
            valid_slices = []
            for z in range(segs_3D.shape[0]):
                if np.any(segs_3D[z] > 0):
                    valid_slices.append(z)
            
            if valid_slices:
                idx = random.choice(valid_slices)
            else:
                idx = random.randint(0, segs_3D.shape[0]-1)
                
            print('plot for idx ', idx)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img_3D[idx], cmap='gray')
            ax[0].set_title('Original Image')
            ax[1].imshow(img_3D[idx], cmap='gray')
            ax[1].imshow(segs_3D[idx] > 0, alpha=0.5)
            ax[1].set_title('Segmentation')
            ax[0].axis('off')
            ax[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(join(args.pred_save_dir, f"{npz_file.stem}_preview.png"))
            plt.close()
        
        print(f"Segmentation saved: {join(args.pred_save_dir, npz_name)}")

if __name__ == "__main__":
    main()
