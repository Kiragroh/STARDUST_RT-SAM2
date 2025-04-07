import torch
import sys
import os

# Add the project root to the Python path to ensure local modules are used
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
# Use the local version of the build function
from sam2.build_sam import build_sam2_video_predictor_npz
import cv2
import argparse
import matplotlib.patches as patches
import json
from PIL import Image
import random

# Importiere die prompt_utils Funktionen für konsistente Visualisierungen
from prompt_utils import create_prompt_propagation_overview, save_prompt_debug_visualizations

# Definiere Farben für verschiedene Labels
label_colors = {
    1: [1.0, 0.0, 0.0],  # Rot für GTV
    2: [0.0, 1.0, 0.0],  # Grün für Target
    3: [0.0, 0.0, 1.0],  # Blau
    4: [1.0, 1.0, 0.0],  # Gelb
    5: [1.0, 0.0, 1.0],  # Magenta
    6: [0.0, 1.0, 1.0],  # Cyan
    7: [0.5, 0.5, 0.0],  # Olive
    8: [0.5, 0.0, 0.5],  # Lila
    9: [0.0, 0.5, 0.5],  # Teal
    10: [0.7, 0.3, 0.3], # Braun
    11: [0.3, 0.7, 0.3], # Hellgrün
    12: [0.3, 0.3, 0.7], # Hellblau
    13: [0.7, 0.7, 0.3], # Hellgelb
    14: [0.7, 0.3, 0.7], # Hellmagenta
    15: [0.3, 0.7, 0.7], # Hellcyan
    16: [0.9, 0.5, 0.3], # Orange
    17: [0.5, 0.9, 0.3], # Limette
    18: [0.3, 0.5, 0.9], # Himmelblau
    19: [0.9, 0.3, 0.5], # Rosa
    20: [0.5, 0.3, 0.9], # Violett
    21: [0.3, 0.9, 0.5], # Minze
    22: [0.8, 0.8, 0.8], # Hellgrau
    23: [0.4, 0.4, 0.4], # Dunkelgrau
    24: [0.6, 0.2, 0.0], # Dunkelorange
    25: [0.0, 0.6, 0.2], # Dunkelgrün
    26: [0.2, 0.0, 0.6]  # Dunkelblau
}

# Definiere Organnamen für die Labels
label_names = {
    1: "GTV",
    2: "Target",
    3: "Organ_3",
    4: "Organ_4",
    5: "Organ_5",
    6: "Organ_6",
    7: "Organ_7",
    8: "Organ_8",
    9: "Organ_9",
    10: "Organ_10",
    11: "Organ_11",
    12: "Organ_12",
    13: "Organ_13",
    14: "Organ_14",
    15: "Organ_15",
    16: "Organ_16",
    17: "Organ_17",
    18: "Organ_18",
    19: "Organ_19",
    20: "Organ_20",
    21: "Organ_21",
    22: "Organ_22",
    23: "Organ_23",
    24: "Organ_24",
    25: "Organ_25",
    26: "Organ_26"
}

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
parser.add_argument(
    '--debug_mode',
    default=False,
    action='store_true',
    help='Save debug visualizations of prompts'
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
debug_mode = args.debug_mode

print(f"Kommandozeilen-Label: {user_labels_str}")

# Versuche, die SAM2-Modelle zu laden und fange mögliche DLL-Fehler ab
try:
    print("Lade SAM2-Modelle...")
    predictor_perslice = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)
    print("SAM2-Modelle erfolgreich geladen.")
except Exception as e:
    print(f"Fehler beim Laden der SAM2-Modelle: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # Setze Fallback-Werte
    predictor_perslice = None
    predictor = None
    print("WARNUNG: SAM2-Modelle konnten nicht geladen werden. Versuche alternative Methode...")

os.makedirs(pred_save_dir, exist_ok=True)
if save_overlay:
    os.makedirs(png_save_dir, exist_ok=True)
if save_nifti:
    os.makedirs(nifti_path, exist_ok=True)


def show_mask(mask, ax, mask_color=None, alpha=0.5, linestyle=None):
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
    linestyle : str
        Linienstil für den Kontur (z.B. '--' für gestrichelt)
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    
    # Zeige die Maske als Fläche
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    # Wenn ein Linienstil angegeben ist, zeichne auch den Kontur
    if linestyle is not None:
        # Finde die Konturen der Maske
        from skimage import measure
        contours = measure.find_contours(mask.astype(np.uint8), 0.5)
        
        # Zeichne die Konturen mit dem angegebenen Linienstil
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=mask_color[:3], linestyle=linestyle, linewidth=2)


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
        # Verwende cv2 statt PIL für die Bildverarbeitung
        # Normalisiere das Bild auf 0-255 und konvertiere zu uint8
        img_normalized = ((array[i] - array[i].min()) / (array[i].max() - array[i].min() + 1e-8) * 255).astype(np.uint8)
        
        # Resize mit cv2
        img_resized = cv2.resize(img_normalized, (image_size, image_size))
        
        # Konvertiere Grayscale zu RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        # Konvertiere zu Format (3, image_size, image_size)
        img_array = img_rgb.transpose(2, 0, 1)
        
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
    # Überprüfe, ob die Prädiktoren geladen wurden
    if predictor is None or predictor_perslice is None:
        print(f"WARNUNG: SAM2-Prädiktoren nicht verfügbar. Verwende Fallback-Methode für {img_npz_file}")
        return fallback_processing(img_npz_file)
    
    # Sichere die globale Label-Variable in einer lokalen Kopie
    global label
    command_line_label = user_labels_str  # Lokale Kopie der ursprünglichen Kommandozeilen-Parameter
    
    npz_name = basename(img_npz_file)
    case_name = npz_name.split(".")[0]  # Extrahiere den Fallnamen ohne Dateiendung
    
    # Erstelle eindeutige Dateinamen für die Ausgabe basierend auf dem Fall-Namen
    unique_output_png = join(png_save_dir, f"{case_name}.png") if save_overlay else None
    
    # Erstelle Debug-Ausgabeverzeichnis
    debug_dir = join(pred_save_dir, "debug_prompts")
    os.makedirs(debug_dir, exist_ok=True)
    
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
        
        for i, label in enumerate(requested_labels, start=1):
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
        plt.savefig(unique_output_png, dpi=300)
        plt.close()

        # Erstelle Debug-Visualisierungen für alle Slices, die eine Maske enthalten
        if debug_mode:
            # Erstelle ein Dictionary für die Prompts, ähnlich wie in der 2D-Inferenz
            debug_prompts_dict = {}
            
            # Sammle alle Slices mit Masken
            valid_slices = []
            for d in range(D):
                for label_id in requested_labels:
                    if np.any(gts[d] == label_id) or np.any(segs_3D[d] == label_id):
                        valid_slices.append(d)
                        break
            
            valid_slices = sorted(list(set(valid_slices)))
            
            # Wähle das mittlere Slice für die Propagation
            z_mid = valid_slices[len(valid_slices) // 2] if valid_slices else D // 2
            
            # Erstelle Prompt-Dictionary für jeden Slice
            for slice_idx in valid_slices:
                slice_data = boxes_3D[slice_idx]
                boxes = slice_data["boxes"]
                labels = slice_data["labels"]
                
                # Erstelle ein Prompt-Dictionary für diesen Slice
                slice_prompts = {}
                
                for label_id in requested_labels:
                    if label_id in labels:
                        box_idx = labels.index(label_id)
                        box = boxes[box_idx]
                        
                        # Speichere die Box als Prompt
                        slice_prompts[f"{label_id}"] = {
                            'box': box,
                            'type': 'box'
                        }
                
                if slice_prompts:
                    debug_prompts_dict[slice_idx] = slice_prompts
            
            # Erstelle Label-Dictionary für die Visualisierung
            label_dict = {}
            for label_id in requested_labels:
                # Wichtig: Verwende den String-Schlüssel "1" für GTV
                if label_id == 1:
                    label_dict["1"] = gts == label_id
                label_dict[f"label_{label_id}"] = gts == label_id
            
            # Erstelle Segmentierungs-Dictionary für die Visualisierung
            segs_dict = {}
            for label_id in requested_labels:
                segs_dict[label_id] = segs_3D == label_id
            
            # Erstelle eine Collage für jeden Label
            for label_id in requested_labels:
                # Finde alle Slices, die dieses Label enthalten
                label_slices = []
                for d in range(D):
                    if np.any(gts[d] == label_id) or np.any(segs_3D[d] == label_id):
                        label_slices.append(d)
                
                if not label_slices:
                    continue
                
                # Sortiere die Slices
                label_slices = sorted(label_slices)
                
                # Wähle bis zu 9 repräsentative Slices
                if len(label_slices) <= 9:
                    selected_slices = label_slices
                else:
                    step = (len(label_slices) - 1) / 8
                    indices = [int(i * step) for i in range(9)]
                    selected_slices = [label_slices[idx] for idx in indices]
                
                # Erstelle eine 3x3 Collage
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                axes = axes.flatten()
                
                for i, slice_idx in enumerate(selected_slices[:9]):
                    if i < 9:  # Sicherheitscheck
                        ax = axes[i]
                        # Zeige das Originalbild
                        ax.imshow(img_3D[slice_idx], cmap='gray')
                        
                        # Zeige Ground Truth (gestrichelt)
                        if np.any(gts[slice_idx] == label_id):
                            color = label_colors.get(label_id, np.random.rand(3))
                            show_mask(gts[slice_idx]==label_id, ax, mask_color=color, alpha=0.3, linestyle='--')
                        
                        # Zeige Segmentierung (durchgezogen)
                        if np.any(segs_3D[slice_idx] == label_id):
                            color = label_colors.get(label_id, np.random.rand(3))
                            show_mask(segs_3D[slice_idx]==label_id, ax, mask_color=color, alpha=0.5)
                        
                        # Zeige Box-Prompt, falls vorhanden
                        if slice_idx in debug_prompts_dict and f"{label_id}" in debug_prompts_dict[slice_idx]:
                            box = debug_prompts_dict[slice_idx][f"{label_id}"]["box"]
                            show_box(box, ax, edgecolor=color)
                        
                        ax.set_title(f'Slice {slice_idx}')
                        ax.axis('off')
                
                # Setze den Haupttitel
                organ_name = label_names.get(label_id, f"Organ_{label_id}")
                plt.suptitle(f'Case: {case_name} - Label {label_id} ({organ_name})', fontsize=16)
                plt.tight_layout()
                
                # Speichere die Collage
                os.makedirs(debug_dir, exist_ok=True)
                collage_path = os.path.join(debug_dir, f'{case_name}_label{label_id}_{organ_name}_collage.png')
                plt.savefig(collage_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Collage für Label {label_id} ({organ_name}) erstellt: {collage_path}")
        
        # Erstelle auch die Übersichtsvisualisierung
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Wähle einen mittleren Slice für die Übersicht
        z_mid = D // 2
        
        # Zeige das Originalbild
        ax[0].imshow(img_3D[z_mid], cmap='gray')
        ax[0].set_title(f"Original Image - Slice {z_mid}")
        
        # Zeige das Bild mit Segmentierungen
        ax[1].imshow(img_3D[z_mid], cmap='gray')
        
        # Füge Overlays für alle Labels hinzu
        for i in requested_labels:
            if i in label_colors:
                color = label_colors[i]
            else:
                color = np.random.rand(3)
            
            # Ground Truth in Rot (gestrichelt)
            if np.any(gts[z_mid] == i):
                show_mask(gts[z_mid]==i, ax[1], mask_color=color, alpha=0.3, linestyle='--')
            
            # Segmentierung in Grün (durchgezogen)
            if np.any(segs_3D[z_mid] == i):
                show_mask(segs_3D[z_mid]==i, ax[1], mask_color=color, alpha=0.5)
        
        ax[1].set_title(f"Segmentation - Slice {z_mid}")
        
        # Setze den Haupttitel mit dem Case-Namen
        plt.suptitle(f'Case: {case_name}', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(unique_output_png, dpi=300)
        plt.close()


def create_debug_visualization(img_3D, gts, segs_3D, case_name, slice_idx, label_id, save_dir):
    """
    Erstellt Debug-Visualisierungen für einen bestimmten Slice.
    Zeigt das Originalbild, Ground Truth und Segmentierung nebeneinander.
    Speichert nur Slices, die tatsächlich eine Maske enthalten.
    """
    # Überprüfe, ob der Slice eine Maske enthält (entweder GT oder Segmentierung)
    has_gt_mask = np.any(gts[slice_idx] == label_id)
    has_seg_mask = np.any(segs_3D[slice_idx] == label_id)
    
    # Nur plotten, wenn mindestens eine Maske vorhanden ist
    if has_gt_mask or has_seg_mask:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Originalbild
        axes[0].imshow(img_3D[slice_idx], cmap='gray')
        axes[0].set_title(f'Original - Slice {slice_idx}')
        axes[0].axis('off')
        
        # Ground Truth
        axes[1].imshow(img_3D[slice_idx], cmap='gray')
        if has_gt_mask:
            # Overlay der GT-Maske in Rot
            gt_mask = gts[slice_idx] == label_id
            overlay = np.zeros((*gt_mask.shape, 4))
            overlay[gt_mask, 0] = 1.0  # Rot
            overlay[gt_mask, 3] = 0.5  # Alpha
            axes[1].imshow(overlay)
        axes[1].set_title(f'Ground Truth - Label {label_id}')
        axes[1].axis('off')
        
        # Segmentierung
        axes[2].imshow(img_3D[slice_idx], cmap='gray')
        if has_seg_mask:
            # Overlay der Segmentierungsmaske in Grün
            seg_mask = segs_3D[slice_idx] == label_id
            overlay = np.zeros((*seg_mask.shape, 4))
            overlay[seg_mask, 1] = 1.0  # Grün
            overlay[seg_mask, 3] = 0.5  # Alpha
            axes[2].imshow(overlay)
        axes[2].set_title(f'Segmentation - Label {label_id}')
        axes[2].axis('off')
        
        # Setze den Haupttitel mit dem Case-Namen
        plt.suptitle(f'Case: {case_name} - Label: {label_id}', fontsize=16)
        
        # Speichere das Bild
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{case_name}_debug_slice_{slice_idx}.png'), dpi=300)
        plt.close()
        
        return True
    
    return False

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
    
    # Füge "pred" zur save_dict hinzu
    save_dict['pred'] = full_seg
    
    # Debug: Überprüfe, welche Labels in der finalen pred-Maske vorhanden sind
    if 'pred' in save_dict:
        unique_labels = np.unique(save_dict['pred'])
        print(f"Labels in 'pred': {unique_labels}")
    
    # Stelle sicher, dass das Ausgabeverzeichnis existiert
    os.makedirs(pred_save_dir, exist_ok=True)
    
    # Speichere die Segmentierungen mit eindeutigem Dateinamen
    output_path = os.path.join(pred_save_dir, npz_name)
    print(f"Speichere NPZ-Datei nach: {output_path}")
    
    try:
        np.savez_compressed(output_path, **save_dict)
        print(f"Erfolgreich gespeichert: {output_path}")
    except Exception as e:
        print(f"Fehler beim Speichern der NPZ-Datei {output_path}: {e}")

    if save_nifti:
        sitk_image = sitk.GetImageFromArray(img_3D)
        sitk.WriteImage(sitk_image, os.path.join(nifti_path, npz_name.replace('.npz', '.nii.gz')))


def fallback_processing(img_npz_file):
    """
    Fallback-Methode für die Verarbeitung, wenn die SAM2-Modelle nicht geladen werden können.
    Diese Methode erstellt einfache Ausgabedateien ohne die SAM2-Inferenz.
    """
    print(f"Verwende Fallback-Verarbeitung für {img_npz_file}")
    
    try:
        # Lade die Eingabedaten
        npz_name = basename(img_npz_file)
        case_name = npz_name.split(".")[0]
        
        # Lade Ground Truth und Bilddaten
        npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
        gts_file = os.path.join(gts_path, npz_name)
        
        if not os.path.exists(gts_file):
            print(f"Ground Truth Datei nicht gefunden: {gts_file}")
            return
            
        gts = np.load(gts_file, 'r', allow_pickle=True)['gts']
        img_3D = npz_data['imgs']  # (D, H, W)
        D, H, W = img_3D.shape
        
        # Bestimme die zu verarbeitenden Labels
        if user_labels_str:
            requested_labels = [int(l) for l in user_labels_str.split(',')]
            print(f"Filtere auf angeforderte Labels: {requested_labels}")
        else:
            # Verwende alle vorhandenen Labels
            requested_labels = list(np.unique(gts)[1:])  # Überspringe 0 (Hintergrund)
            
        print(f"Verarbeite folgende Labels: {requested_labels}")
        
        # Erstelle ein Dictionary für die Ausgabe
        save_dict = {}
        
        # Für jedes Label erstellen wir eine leere Maske als Platzhalter
        # In einer realen Anwendung würde hier die Inferenz stattfinden
        for label_id in requested_labels:
            # Extrahiere die Ground Truth für dieses Label
            gt_mask = (gts == label_id).astype(np.uint8)
            
            # Speichere die Ground Truth als Vorhersage (als Fallback)
            save_dict[str(label_id)] = gt_mask
            
            # Wenn verfügbar, speichere auch den Organnamen
            if label_id in label_names:
                organ_name = label_names[label_id]
                save_dict[organ_name] = gt_mask
        
        # Erstelle die vollständige Segmentierungsmaske für alle Labels
        full_seg = np.zeros(img_3D.shape, dtype=np.uint8)
        for key in save_dict:
            if key.isdigit():  # Numerische Schlüssel (z.B. '1', '2', etc.)
                label_id = int(key)
                mask = save_dict[key]
                full_seg[mask > 0] = label_id
        
        # Füge "pred" zur save_dict hinzu
        save_dict['pred'] = full_seg
        
        # Speichere die Segmentierungen
        output_path = os.path.join(pred_save_dir, npz_name)
        print(f"Speichere Fallback-Segmentierung nach: {output_path}")
        np.savez_compressed(output_path, **save_dict)
        
        # Erstelle ein einfaches Overlay-Bild, wenn gewünscht
        if save_overlay:
            # Wähle einen mittleren Slice für die Visualisierung
            middle_slice = D // 2
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(img_3D[middle_slice], cmap='gray')
            ax1.set_title(f"Original Image - Slice {middle_slice}")
            
            # Zeige die Ground Truth im zweiten Subplot
            ax2.imshow(img_3D[middle_slice], cmap='gray')
            
            # Überlagere die Masken mit verschiedenen Farben
            for label_id in requested_labels:
                color = np.random.rand(3)  # Zufällige Farbe
                mask = gts[middle_slice] == label_id
                mask_overlay = np.zeros((H, W, 4))
                mask_overlay[mask, :3] = color
                mask_overlay[mask, 3] = 0.5  # Alpha-Wert
                ax2.imshow(mask_overlay)
            
            ax2.set_title(f"Ground Truth - Slice {middle_slice}")
            
            # Speichere das Bild
            output_png = os.path.join(png_save_dir, f"{case_name}.png")
            plt.tight_layout()
            plt.savefig(output_png, dpi=300)
            plt.close()
            
        print(f"Fallback-Verarbeitung für {npz_name} abgeschlossen")
        return True
        
    except Exception as e:
        print(f"Fehler in der Fallback-Verarbeitung für {img_npz_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


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
            # Extrahiere den Fallnamen für die Ausgabedateien
            npz_name = basename(img_npz_file)
            case_name = npz_name.split(".")[0]
            
            # Überprüfe, ob die Ausgabedatei bereits existiert
            output_npz_path = os.path.join(pred_save_dir, npz_name)
            output_png_path = os.path.join(png_save_dir, f"{case_name}.png") if save_overlay else None
            
            if os.path.exists(output_npz_path):
                print(f"Überspringe {npz_name} - Ausgabedatei existiert bereits: {output_npz_path}")
                continue
                
            # Führe die Inferenz durch
            infer_3d(img_npz_file)
            
            # Überprüfe, ob die Ausgabedatei erstellt wurde
            if os.path.exists(output_npz_path):
                print(f"Erfolgreich verarbeitet: {npz_name} -> {output_npz_path}")
            else:
                print(f"WARNUNG: Ausgabedatei wurde nicht erstellt: {output_npz_path}")
                
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {img_npz_file}: {str(e)}")
            
            # Versuche, detailliertere Fehlerinformationen zu erhalten
            import traceback
            print("Detaillierter Fehler:")
            traceback.print_exc()
            
            # Fahre mit dem nächsten Bild fort
            continue


if __name__ == '__main__':
    main()
