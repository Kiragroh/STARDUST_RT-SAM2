#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt Utilities for Segment Anything Model 2 (SAM2)

Funktionen zum Generieren von automatischen Prompts (Punkten und Bounding Boxes)
basierend auf Ground-Truth-Masken zur Verwendung mit SAM2 und MedSAM2.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage import measure, morphology
from pathlib import Path


def get_bbox(mask, bbox_shift=5):
    """
    Erzeugt eine Bounding Box aus einer Maske mit optionalem Versatz.
    
    Args:
        mask: Binärmaske
        bbox_shift: Versatz der Bounding Box (größer macht Box größer)
        
    Returns:
        box_coords: [x_min, y_min, x_max, y_max] Koordinaten
    """
    if np.sum(mask) == 0:
        return None
    
    # Koordinaten der Maske finden
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Versatz anwenden
    h, w = mask.shape[:2]
    x_min = max(0, x_min - bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    x_max = min(w - 1, x_max + bbox_shift)
    y_max = min(h - 1, y_max + bbox_shift)
    
    box_coords = np.array([x_min, y_min, x_max, y_max])
    return box_coords


def generate_negative_masks(gt_mask, label_data, suffix="_neg+"):
    """
    Identifiziert alle Labels mit dem Suffix '_neg+' und erstellt eine Liste von Masken,
    die als negative Prompts verwendet werden können.
    
    Args:
        gt_mask: Ground-Truth-Maske für das aktuelle Label
        label_data: Dictionary mit allen Labeldaten
        suffix: Suffix für negative Labels
        
    Returns:
        Liste von Masken, die für negative Prompts verwendet werden können
    """
    negative_masks = []
    
    for name, mask in label_data.items():
        if name.endswith(suffix):
            # Nur Bereiche verwenden, die nicht mit der GT-Maske überlappen
            neg_mask = mask & ~gt_mask
            if np.sum(neg_mask) > 0:
                negative_masks.append(neg_mask)
    
    return negative_masks


def generate_random_points(mask, num_points=3, min_distance_from_edge=3):
    """
    Generiert zufällige Punkte innerhalb einer Maske mit Mindestabstand vom Rand.
    
    Args:
        mask: Binärmaske
        num_points: Anzahl zu generierender Punkte
        min_distance_from_edge: Minimaler Abstand vom Rand der Maske
        
    Returns:
        points: Array von [x, y] Koordinaten
    """
    if np.sum(mask) == 0:
        return np.empty((0, 2), dtype=np.int32)
    
    # Erode Maske, um Punkte mit Mindestabstand vom Rand zu erzeugen
    if min_distance_from_edge > 0:
        eroded_mask = morphology.erosion(mask, morphology.disk(min_distance_from_edge))
        if np.sum(eroded_mask) == 0:  # Falls die erodierte Maske leer ist
            eroded_mask = mask  # Zurück zur ursprünglichen Maske
    else:
        eroded_mask = mask
    
    # Indizes aller Punkte innerhalb der erodierten Maske
    y_indices, x_indices = np.where(eroded_mask > 0)
    if len(y_indices) == 0:
        return np.empty((0, 2), dtype=np.int32)
    
    # Zufällige Auswahl von Indizes
    num_valid_points = min(num_points, len(y_indices))
    if num_valid_points == 0:
        return np.empty((0, 2), dtype=np.int32)
    
    idx = np.random.choice(len(y_indices), size=num_valid_points, replace=False)
    
    # Punkte im Format [x, y] zurückgeben
    points = np.column_stack((x_indices[idx], y_indices[idx]))
    return points


def generate_prompts(gt_mask, label_data, prompt_type='box', num_pos_points=3, num_neg_points=1, 
                    min_dist_from_edge=3, bbox_shift=5):
    """
    Generiert Prompts basierend auf der Ground-Truth-Maske und Labels mit Suffix.
    
    Args:
        gt_mask: Ground-Truth-Maske
        label_data: Dictionary mit allen Labeldaten
        prompt_type: 'box' oder 'point'
        num_pos_points: Anzahl positiver Punkte in der Maske
        num_neg_points: Anzahl negativer Punkte aus negativen Masken
        min_dist_from_edge: Mindestabstand vom Rand der Maske
        bbox_shift: Versatz für Bounding Boxes
        
    Returns:
        prompts: Dictionary mit 'box' oder 'points' und 'labels'
    """
    prompts = {}
    
    if prompt_type == 'box':
        # Bounding Box generieren
        box = get_bbox(gt_mask, bbox_shift)
        if box is not None:
            prompts['box'] = box
        
    elif prompt_type == 'point':
        # Positive Punkte innerhalb der GT-Maske
        pos_points = generate_random_points(gt_mask, num_pos_points, min_dist_from_edge)
        
        # Negative Punkte aus negativen Masken
        negative_masks = generate_negative_masks(gt_mask, label_data)
        neg_points = []
        
        points_per_mask = max(1, num_neg_points // len(negative_masks)) if negative_masks else 0
        
        for neg_mask in negative_masks:
            mask_points = generate_random_points(neg_mask, points_per_mask, min_dist_from_edge)
            if len(mask_points) > 0:
                neg_points.extend(mask_points)
        
        neg_points = np.array(neg_points) if neg_points else np.empty((0, 2), dtype=np.int32)
        
        # Kombiniere positive und negative Punkte
        points = np.vstack([pos_points, neg_points]) if len(neg_points) > 0 else pos_points
        labels = np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)
        
        if len(points) > 0:
            prompts['points'] = points
            prompts['labels'] = labels
    
    return prompts


def visualize_prompts(image, gt_mask, prompts, output_path, slice_idx=None):
    """
    Visualisiert die generierten Prompts auf dem Bild und der GT-Maske.
    
    Args:
        image: CT/MRT-Bild
        gt_mask: Ground-Truth-Maske
        prompts: Dictionary mit 'box' oder 'points' und 'labels'
        output_path: Pfad zum Speichern der Visualisierung
        slice_idx: Slice-Index für Dateibenennung
    """
    # Maske auf RGB konvertieren für Overlay
    mask_rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    mask_rgb[gt_mask > 0] = [255, 0, 0]  # Rot für GT-Maske
    
    # Bild auf 3-Kanal konvertieren und normalisieren falls notwendig
    if len(image.shape) == 2:
        img_rgb = np.stack([image] * 3, axis=-1)
    else:
        img_rgb = image
        
    if img_rgb.dtype != np.uint8:
        img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8) * 255
        img_rgb = img_rgb.astype(np.uint8)
    
    # Maske transparent auf Bild überlagern
    alpha = 0.3
    overlay = cv2.addWeighted(img_rgb, 1, mask_rgb, alpha, 0)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(overlay)
    
    if 'box' in prompts:
        box = prompts['box']
        x_min, y_min, x_max, y_max = box
        plt.plot([x_min, x_max, x_max, x_min, x_min], 
                [y_min, y_min, y_max, y_max, y_min], 'g-', linewidth=2)
        plt.title(f'GTV mit Bounding Box Prompt')
        
    elif 'points' in prompts and 'labels' in prompts:
        points = prompts['points']
        labels = prompts['labels']
        
        # Farben: rot für positive (1), blau für negative (0)
        colors = ['blue' if label == 0 else 'green' for label in labels]
        markers = ['x' if label == 0 else '+' for label in labels]
        
        for i, (point, color, marker) in enumerate(zip(points, colors, markers)):
            plt.scatter(point[0], point[1], c=color, marker=marker, s=100)
        
        plt.title(f'GTV mit {len(points)} Punkt-Prompts')
    
    # Speichern der Visualisierung
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_prompt_debug_visualizations(image_data, label_dict, prompts_dict, output_dir, case_name):
    """
    Erzeugt Debug-Visualisierungen für GTV-Slices mit Prompts und speichert sie im Ausgabeverzeichnis.
    
    Args:
        image_data: 3D-Bilddaten
        label_dict: Dictionary mit Label-Masken
        prompts_dict: Dictionary mit generierten Prompts pro Slice
        output_dir: Ausgabeverzeichnis
        case_name: Name des Falls
    """
    # GTV-Maske extrahieren
    gtv_label = None
    for label_name, label_mask in label_dict.items():
        if "gtv" in label_name.lower() or label_name.startswith("1_") or label_name == "1":
            gtv_label = label_name
            break
    
    if gtv_label is None:
        print("Keine GTV-Maske gefunden für Debug-Visualisierungen")
        return
    
    gtv_mask = label_dict[gtv_label]
    
    for z in range(image_data.shape[0]):
        if z in prompts_dict and np.any(gtv_mask[z]):
            image_slice = image_data[z]
            gt_slice = gtv_mask[z]
            prompts = prompts_dict[z]
            
            output_path = os.path.join(output_dir, f"{case_name}_slice{z:03d}_prompts.png")
            visualize_prompts(image_slice, gt_slice, prompts, output_path, z)
