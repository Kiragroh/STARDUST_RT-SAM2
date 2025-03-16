import torch
from sam2.build_sam import build_sam2
import glob
from tqdm import tqdm
from time import time
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import matplotlib.pyplot as plt
from os.path import basename, join, exists
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
import pandas as pd
from datetime import datetime
from sam2.build_sam import build_sam2_video_predictor_npz
import cv2
import argparse
import matplotlib.patches as patches
import json

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

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

parser = argparse.ArgumentParser()

parser.add_argument(
    '--checkpoint',
    type=str,
    default="./checkpoints/sam2_hiera_base_plus.pt",
    help='checkpoint path',
)
parser.add_argument(
    '--cfg',
    type=str,
    default="sam2_hiera_b+.yaml",
    help='model config',
)
parser.add_argument(
    '--png_save_dir',
    type=str,
    default="./results/overlay_base",
    help='GT and predicted masks will be saved here',
)
parser.add_argument(
    '--save_overlay',
    default=False,
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '--imgs_path',
    type=str,
    default="./data/imgs",
    help='imgs path',
)
parser.add_argument(
    '--gts_path',
    type=str,
    default="./data/gts",
    help='gts path',
)
parser.add_argument(
    '--pred_save_dir',
    type=str,
    default="./results/segs_base",
    help='segs path',
)
parser.add_argument(
    '--save_nifti',
    default=False,
    action='store_true',
    help='whether to save nifti'
)
parser.add_argument(
    '--nifti_path',
    type=str,
    default="./results/segs_nifti",
    help='segs nifti path',
)
parser.add_argument(
    '--label',
    type=str,
    default=None,
    help='Specify which label to segment, e.g. "1,2,3" for multiple labels or a single label'
)

args = parser.parse_args()
checkpoint = args.checkpoint
model_cfg = args.cfg
save_overlay = args.save_overlay
png_save_dir = args.png_save_dir
imgs_path = args.imgs_path
gts_path = args.gts_path
pred_save_dir = args.pred_save_dir
save_nifti = args.save_nifti
nifti_path = args.nifti_path
user_labels_str = args.label  # Verwende einen anderen Namen für die Variable

print(f"Kommandozeilen-Label: {user_labels_str}")
predictor_perslice = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

os.makedirs(pred_save_dir, exist_ok=True)
if save_overlay:
    os.makedirs(png_save_dir, exist_ok=True)
if save_nifti:
    os.makedirs(nifti_path, exist_ok=True)


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
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
    """Berechnet Bounding Box aus einer binären Maske"""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # Add padding
    pad = 5  # Reduziere Padding
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(mask.shape[1]-1, x_max + pad)
    y_max = min(mask.shape[0]-1, y_max + pad)
    return np.array([x_min, y_min, x_max, y_max])


def scale_box(box, original_size, target_size):
    """Skaliert eine Bounding Box von original_size auf target_size"""
    h_scale = target_size / original_size
    w_scale = target_size / original_size
    return np.array([
        int(box[0] * w_scale),  # x_min
        int(box[1] * h_scale),  # y_min
        int(box[2] * w_scale),  # x_max
        int(box[3] * h_scale)   # y_max
    ])

def resize_box_to_original(box, target_size, original_size):
    """Skaliert eine Bounding Box zurück zur Originalgröße"""
    h_scale = original_size / target_size
    w_scale = original_size / target_size
    return np.array([
        int(box[0] * w_scale),  # x_min
        int(box[1] * h_scale),  # y_min
        int(box[2] * w_scale),  # x_max
        int(box[3] * h_scale)   # y_max
    ])

def medsam_inference_3d(
    medsam_model,
    image_3d: np.ndarray,
    bbox_3d: np.ndarray,
    device,
    patch_size: tuple = (128, 256, 256),
    overlap: tuple = (32, 64, 64)
):
    """
    True 3D-Inferenz mit Patch-basierter Verarbeitung
    Args:
        bbox_3d: [x1, y1, z1, x2, y2, z2] in 3D-Koordinaten
        patch_size: (Tiefe, Höhe, Breite) der 3D-Patches
    """
    
    # 1. 3D-Bild vorverarbeiten
    image_3d = (image_3d - image_3d.min()) / (image_3d.max() - image_3d.min())
    
    # 2. Extrahiere ROI basierend auf 3D-BBox
    x1, y1, z1, x2, y2, z2 = bbox_3d.astype(int)
    roi = image_3d[z1:z2, y1:y2, x1:x2]
    
    # 3. Patch-basierte Verarbeitung
    output = np.zeros_like(roi)
    
    # Berechne Patch-Positionen mit Überlappung
    dz, dy, dx = patch_size
    oz, oy, ox = overlap
    
    for z in range(0, roi.shape[0], dz - oz):
        for y in range(0, roi.shape[1], dy - oy):
            for x in range(0, roi.shape[2], dx - ox):
                patch = roi[
                    z:z+dz,
                    y:y+dy,
                    x:x+dx
                ]
                
                # 3D->2D Reduktion für SAM2
                patch_2d = patch.max(axis=0)  # MIP-Projektion
                
                # Inferenz auf 2D-Patch
                with torch.no_grad():
                    patch_tensor = torch.as_tensor(patch_2d).float().to(device)
                    mask = medsam_model(patch_tensor.unsqueeze(0))
                    
                # Rückprojektion in 3D
                output[
                    z:z+dz,
                    y:y+dy,
                    x:x+dx
                ] = mask.squeeze().cpu().numpy()
    
    return output


@torch.inference_mode()
def infer_3d(img_npz_file):
    # Sichere die globale Label-Variable in einer lokalen Kopie
    global label
    command_line_label = user_labels_str  # Lokale Kopie der ursprünglichen Kommandozeilen-Parameter
    
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    gts = np.load(os.path.join(gts_path, npz_name), 'r', allow_pickle=True)['gts']
    img_3D = npz_data['imgs']  # (D, H, W)
    assert np.max(img_3D) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D)}'
    D, H, W = img_3D.shape
    segs_3D = np.zeros(img_3D.shape, dtype=np.uint8)
    
    # Debug Plots für Bounding Boxes
    def plot_debug(img, gt, boxes, labels, slice_idx, save_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(img, cmap='gray')
        ax2.imshow(gt, cmap='gray')
        ax1.set_title(f'Image Slice {slice_idx}')
        ax2.set_title(f'GT Slice {slice_idx}')
        
        # Original Größe der Boxen
        boxes_orig = [resize_box_to_original(box, 1024, img.shape[0]) for box in boxes]
        
        for box, box_orig, label in zip(boxes, boxes_orig, labels):
            color = np.random.rand(3)
            # Zeige beide Boxen - original und skaliert
            rect1 = patches.Rectangle((box_orig[0], box_orig[1]), 
                                   box_orig[2]-box_orig[0], 
                                   box_orig[3]-box_orig[1],
                                   linewidth=1, edgecolor=color, facecolor='none')
            rect2 = patches.Rectangle((box_orig[0], box_orig[1]), 
                                   box_orig[2]-box_orig[0], 
                                   box_orig[3]-box_orig[1],
                                   linewidth=1, edgecolor=color, facecolor='none')
            ax1.add_patch(rect1)
            ax2.add_patch(rect2)
            ax1.text(box_orig[0], box_orig[1], f'L{label} ({box[0]},{box[1]})-({box[2]},{box[3]})', 
                    color=color, fontsize=8)
            
        plt.savefig(os.path.join(save_dir, f'debug_slice_{slice_idx}.png'))
        plt.close()

    # Berechne Bounding Boxes für jedes Label
    unique_labels = np.unique(gts)[1:]  # Überspringe 0 (Hintergrund)
    
    # Ursprüngliche Labels speichern
    original_labels = list(unique_labels)
    print(f"Gefundene Labels im Datensatz: {original_labels}")
    
    # Filterung der Labels basierend auf dem Kommandozeilenparameter
    labels_to_process = []
    
    # Wenn user_labels_str gesetzt ist, filtern wir die Labels
    if user_labels_str:
        try:
            requested_labels = [int(l) for l in user_labels_str.split(',')]
            labels_to_process = [l for l in unique_labels if l in requested_labels]
            print(f"Filtere auf angeforderte Labels: {labels_to_process}")
            
            # Überprüfe, ob angeforderte Labels nicht im Datensatz vorhanden sind
            missing_labels = [l for l in requested_labels if l not in unique_labels]
            if missing_labels:
                print(f"WARNUNG: Folgende angeforderte Labels wurden im Datensatz nicht gefunden: {missing_labels}")
                
            # Wenn keine Labels übrig bleiben, erstelle leere Segmentierung
            if len(labels_to_process) == 0:
                print(f"Keine angeforderten Labels in diesem Datensatz gefunden. Erstelle leere Segmentierung.")
                
                # Speichere leere Maske für jedes angeforderte Label
                save_dict = {}
                for label_id in requested_labels:
                    empty_mask = np.zeros(img_3D.shape, dtype=np.uint8)
                    
                    # Für Label 1 immer GTV verwenden
                    if label_id == 1:
                        organ_name = 'GTV'
                    elif label_names and label_id in label_names and label_names[label_id]:
                        organ_name = label_names[label_id]
                    else:
                        organ_name = f'Organ_{label_id}'
                    
                    print(f"Label {label_id} wird als {organ_name} (leer) gespeichert")
                    save_dict[str(label_id)] = empty_mask
                    save_dict[organ_name] = empty_mask
                
                save_pred(save_dict, img_3D, npz_name)
                return  # Beende die Funktion frühzeitig
        except Exception as e:
            print(f"Fehler beim Parsen der Label-Liste '{user_labels_str}': {e}")
            labels_to_process = original_labels
    else:
        print(f"Kein Label-Filter angegeben, verwende alle gefundenen Labels: {original_labels}")
        labels_to_process = original_labels
    
    print(f"Verarbeite folgende Labels: {labels_to_process}")
    
    boxes_3D = []
    
    debug_dir = os.path.join(pred_save_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"Found labels: {unique_labels}")
    
    for d in range(D):
        boxes_slice = []
        slice_labels = []
        gt_slice = gts[d]
        for label_id in labels_to_process:
            if np.any(gt_slice == label_id):  # Prüfe ob Label in diesem Slice existiert
                bbox = get_bbox_from_mask(gt_slice == label_id)
                if bbox is not None:
                    # Skaliere Box auf 1024x1024
                    bbox_1024 = scale_box(bbox, H, 1024)
                    # Prüfe ob Box gültig ist
                    if (bbox_1024[2] - bbox_1024[0] > 0 and 
                        bbox_1024[3] - bbox_1024[1] > 0 and
                        bbox_1024[2] <= 1024 and bbox_1024[3] <= 1024):
                        boxes_slice.append(bbox_1024)
                        slice_labels.append(label_id)
                        print(f"Slice {d}: Found box {bbox} -> {bbox_1024} for label {label_id}")
                    else:
                        print(f"Slice {d}: Invalid box {bbox_1024} for label {label_id}")
        
        boxes_3D.append({"boxes": boxes_slice, "labels": slice_labels})
        
        # Debug Plot für jeden 10. Slice
        if d % 10 == 0:
            plot_debug(img_3D[d], gt_slice, boxes_slice, slice_labels, d, debug_dir)
            
    video_height = img_3D.shape[1]
    video_width = img_3D.shape[2]
    img_resized = resize_grayscale_to_rgb_and_resize(img_3D, 1024) #d, 3, 1024, 1024
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std
    z_mids = []
    
    # Finde zusammenhängende Regionen
    for label_id in labels_to_process:
        label_mask = gts == label_id
        # Finde zusammenhängende Regionen für dieses Label
        regions = []
        current_region = []
        for z in range(D):
            if np.any(label_mask[z]):
                current_region.append(z)
            elif current_region:
                regions.append(current_region)
                current_region = []
        if current_region:
            regions.append(current_region)
            
        # Verarbeite jede Region
        for region in regions:
            z_min, z_max = min(region), max(region)
            z_mid = (z_min + z_max) // 2
            z_mids.append((z_mid, z_min, z_max, label_id))
            
    for z_mid, z_min, z_max, label_id in z_mids:
        z_mid_orig = z_mid
        img = img_resized[z_min:z_max+1]  # Verwende nur den relevanten Bereich
        
        # Finde die Box für dieses Label in diesem Slice
        boxes, labels = boxes_3D[z_mid]["boxes"], boxes_3D[z_mid]["labels"]
        if label_id in labels:
            box_idx = labels.index(label_id)
            box = boxes[box_idx]
            
            # Erstelle Prompt-Maske
            mask_prompt = torch.zeros((1024, 1024), dtype=torch.bool)
            y1, x1, y2, x2 = box
            mask_prompt[y1:y2, x1:x2] = True
            mask_prompt = mask_prompt.cuda()
            
            inference_state = predictor.init_state(img, video_height, video_width)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=z_mid-z_min, obj_id=1, mask=mask_prompt)
            # Original Schwellenwert wiederherstellen
            segs_3D[z_mid_orig, ((masks[0] > -0.5).cpu().numpy())[0]] = label_id
            
            # Forward propagation
            forward_masks = []
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                mask = (out_mask_logits[0] > -0.5).cpu().numpy()[0]
                forward_masks.append(mask)
                segs_3D[(z_min + out_frame_idx)] = np.where(mask, label_id, segs_3D[(z_min + out_frame_idx)])
            
            # Backward propagation
            predictor.reset_state(inference_state)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=z_mid-z_min, obj_id=1, mask=mask_prompt)
            backward_masks = []
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                mask = (out_mask_logits[0] > -0.5).cpu().numpy()[0]
                backward_masks.append(mask)
                segs_3D[(z_min + out_frame_idx)] = np.where(mask, label_id, segs_3D[(z_min + out_frame_idx)])
            
            predictor.reset_state(inference_state)

    print(np.unique(segs_3D))
    # Erstelle einzelne Masken für jedes Organ gemäß dem erwarteten Format
    save_dict = {}
    for i in np.unique(segs_3D):
        if i > 0 and i in labels_to_process:  # Ignoriere Hintergrund (0) und nicht ausgewählte Labels
            mask = (segs_3D == i).astype(np.uint8)
            
            # Für Label 1 immer GTV verwenden, unabhängig von label_names
            if i == 1:
                organ_name = 'GTV'  # Hartkodiert für Label 1
                print(f"Label {i} wird als {organ_name} gespeichert")
            elif label_names and i in label_names and label_names[i]:
                organ_name = label_names[i]
                print(f"Label {i} wird als {organ_name} gespeichert (aus label_names)")
            else:
                # Fallback zu generischem Namen
                organ_name = f'Organ_{i}'
                print(f"Label {i} wird als {organ_name} gespeichert (Fallback)")
            
            # Wichtig: Sowohl den numerischen Schlüssel (i) als auch den beschreibenden Schlüssel (organ_name) speichern
            # damit visualize_segmentation.py beide Formate verarbeiten kann
            save_dict[str(i)] = mask  # Numerischer Schlüssel als String
            save_dict[organ_name] = mask  # Beschreibender Schlüssel
    
    print(f"Speichere folgende Schlüssel in NPZ: {list(save_dict.keys())}")
    
    # Optional: Bilder auch speichern
    include_images = False
    if include_images:
        save_dict['imgs'] = img_3D
        print("CT-Bilder wurden zur NPZ-Datei hinzugefügt")
    
    save_pred(save_dict, img_3D, npz_name)

    if save_nifti:
        sitk_image = sitk.GetImageFromArray(segs_3D)
        sitk.WriteImage(sitk_image, os.path.join(nifti_path, npz_name.replace('.npz', '.nii.gz')))

    if save_overlay:
        npz_gts = np.load(join(gts_path, npz_name), 'r', allow_pickle=True)
        gts = npz_gts['gts']
        
        # Wähle einen Slice, der tatsächlich Segmentierungen enthält
        valid_slices = []
        for z in range(D):
            if np.any(segs_3D[z] > 0):
                valid_slices.append(z)
        
        if valid_slices:
            idx = random.choice(valid_slices)
        else:
            idx = random.randint(0, D-1)
            
        print('plot for idx ', idx)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[1].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Ground Truth")
        ax[1].set_title("SAM2 Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        slice_data = boxes_3D[idx]
        boxes = slice_data["boxes"]
        labels = slice_data["labels"]
        
        for i, label in enumerate(labels_to_process, start=1):
            if np.sum(segs_3D[idx]==i) > 0 or np.sum(gts[idx]==i) > 0:
                color = np.random.rand(3)
                # Finde Box für dieses Label in diesem Slice
                box = None
                if label in labels:
                    box_idx = labels.index(label)
                    box = boxes[box_idx]
                
                if box is not None:
                    show_box(box, ax[1], edgecolor=color)
                    show_box(box, ax[0], edgecolor=color)
                show_mask(segs_3D[idx]==i, ax[1], mask_color=color)
                show_mask(gts[idx]==i, ax[0], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()


def save_pred(save_dict, img_3D, npz_name):
    """
    Speichert die Segmentierungen im NPZ-Format.
    
    Wichtig: Speichere die Daten in einem Format, das von compute_metrics.py erkannt wird:
    1. Die Labels müssen sowohl als numerischer Schlüssel (z.B. '1') als auch als Organname (z.B. 'GTV') abgelegt werden
    2. Die Masken müssen exakt im gleichen Format gespeichert werden wie in den 2D-Skripten
    """
    # Füge zuerst Debug-Informationen hinzu
    print(f"Speichere Segmentierungen für {npz_name} mit folgenden Schlüsseln: {list(save_dict.keys())}")

    # Für compute_metrics.py benötigen wir einen speziellen Schlüssel "pred", der die gesamte Segmentierung enthält
    # Erstelle die vollständige Segmentierungsmaske für alle Labels
    full_seg = np.zeros(img_3D.shape, dtype=np.uint8)
    
    # Gehe durch alle Organe und füge sie zur Gesamtmaske hinzu
    for key in save_dict:
        if key.isdigit():  # Numerische Schlüssel (z.B. '1', '2', etc.)
            label_id = int(key)
            mask = save_dict[key]
            # Füge diese Maske zur Gesamtmaske hinzu
            full_seg[mask > 0] = label_id
    
    # Füge "pred" zur save_dict hinzu, damit compute_metrics.py die Segmentierung finden kann
    save_dict['pred'] = full_seg
    
    # Debug: Überprüfe, welche Labels in der finalen pred-Maske vorhanden sind
    if 'pred' in save_dict:
        unique_labels = np.unique(save_dict['pred'])
        print(f"Labels in 'pred': {unique_labels}")
    
    # Speichere die Segmentierungen
    np.savez_compressed(join(pred_save_dir, npz_name), **save_dict)

    if save_nifti:
        sitk_image = sitk.GetImageFromArray(img_3D)
        sitk.WriteImage(sitk_image, os.path.join(nifti_path, npz_name.replace('.npz', '.nii.gz')))


def main():
    # Globale Variablen für Organ-Namen und Farben
    os.makedirs(pred_save_dir, exist_ok=True)
    if save_overlay:
        os.makedirs(png_save_dir, exist_ok=True)

    # Get all npz files
    img_list = []
    for img_path in sorted(glob.glob(os.path.join(imgs_path, '*.npz'))):
        img_list.append(img_path)
    print(len(img_list))

    # Iteriere über alle Bilddateien mit tqdm
    for i, img_npz_file in enumerate(tqdm(img_list)):
        try:
            infer_3d(img_npz_file)
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {img_npz_file}: {e}")
            continue


if __name__ == '__main__':
    main()
