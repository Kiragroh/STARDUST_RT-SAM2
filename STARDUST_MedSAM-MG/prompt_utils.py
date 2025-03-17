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
import logging


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
    Identifiziert Labels mit '_neg+' im Namen und erstellt Masken für negative Prompts.
    
    Args:
        gt_mask: Ground-Truth-Maske des aktuellen Labels
        label_data: Dictionary mit Labeldaten (Schlüssel: 'label_id_name')
        suffix: Suffix für negative Labels (default: '_neg+')
    
    Returns:
        Liste von Masken für negative Prompts
    """
    negative_masks = []
    for name, mask in label_data.items():
        if name.endswith(suffix):  # Prüfe den Suffix im Namen
            neg_mask = mask & ~gt_mask  # Nur Bereiche außerhalb der GT-Maske
            if np.sum(neg_mask) > 0:
                negative_masks.append(neg_mask)
                logging.debug(f"Added negative mask for {name}")
    return negative_masks


def generate_random_points(mask, num_points=3, min_distance_from_edge=3):
    """
    Generiert zufällige Punkte innerhalb einer Maske mit Mindestabstand vom Rand.
    Falls nicht möglich, wird ein Punkt in die Mitte der Maske gesetzt.
    
    Args:
        mask: Binärmaske (numpy array)
        num_points: Anzahl zu generierender Punkte
        min_distance_from_edge: Minimaler Abstand vom Rand der Maske
        
    Returns:
        points: Array von [x, y] Koordinaten
    """
    # Falls die Maske leer ist, geben wir ein leeres Array zurück
    if np.sum(mask) == 0:
        return np.empty((0, 2), dtype=np.int32)
    
    # Erodierte Maske erstellen, um Mindestabstand zum Rand zu gewährleisten
    if min_distance_from_edge > 0:
        eroded_mask = morphology.erosion(mask, morphology.disk(min_distance_from_edge))
    else:
        eroded_mask = mask
    
    # Koordinaten der erodierten Maske abrufen
    y_indices, x_indices = np.where(eroded_mask > 0)
    
    # Falls genügend Punkte verfügbar sind, wähle zufällig aus
    if len(y_indices) >= num_points:
        idx = np.random.choice(len(y_indices), size=num_points, replace=False)
        points = np.column_stack((x_indices[idx], y_indices[idx]))
    else:
        # Fallback: Punkt in die Mitte der Maske setzen
        center_y, center_x = ndimage.center_of_mass(mask)
        points = np.array([[int(center_x), int(center_y)]], dtype=np.int32)
    
    return points


def generate_prompts(gt_mask, label_data, image_slice, prompt_type='point', num_pos_points=3, num_neg_points=1, 
                     min_dist_from_edge=3, threshold=50):
    """
    Generiert Prompts basierend auf der Ground-Truth-Maske und Labels mit Suffix.
    
    Args:
        gt_mask: Ground-Truth-Maske
        label_data: Dictionary mit allen Labeldaten
        image_slice: 2D-Bild-Slice, um Bildwerte zu überprüfen
        prompt_type: 'box' oder 'point'
        num_pos_points: Anzahl positiver Punkte
        num_neg_points: Anzahl negativer Punkte
        min_dist_from_edge: Mindestabstand vom Rand
        threshold: Schwellenwert für Bildwerte, um "Luft" auszuschließen
        
    Returns:
        prompts: Dictionary mit 'points' und 'labels'
    """
    prompts = {}
    if prompt_type == 'point':
        # Positive Punkte
        pos_points = generate_random_points(gt_mask, num_pos_points, min_dist_from_edge)
        
        # Negative Punkte
        negative_masks = generate_negative_masks(gt_mask, label_data)
        if negative_masks:
            neg_points = []
            points_per_mask = max(1, num_neg_points // len(negative_masks))
            for neg_mask in negative_masks:
                mask_points = generate_random_points(neg_mask, points_per_mask, min_dist_from_edge)
                if len(mask_points) > 0:
                    neg_points.extend(mask_points)
        else:
            # Fallback: Punkte außerhalb der GT-Maske, aber nur in Bereichen mit Bildwert > threshold
            foreground_mask = image_slice > threshold  # Schwellenwert für relevante Bereiche
            fallback_neg_mask = foreground_mask & ~gt_mask
            neg_points = generate_random_points(fallback_neg_mask, num_neg_points, min_dist_from_edge)
        
        points = np.vstack([pos_points, neg_points]) if len(neg_points) > 0 else pos_points
        labels = np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)
        
        # Ausgabe der Bildwerte für alle Punkte
        for point in points:
            x, y = point
            image_value = image_slice[y, x]  # Annahme: image_slice ist 2D
            #print(f"Punkt {point} hat Bildwert: {image_value}")
        
        if len(points) > 0:
            prompts['points'] = points
            prompts['labels'] = labels
    elif prompt_type == 'box':
        box = get_bbox(gt_mask, bbox_shift=5)  # bbox_shift standardmäßig auf 5
        if box is not None:
            prompts['box'] = box
    return prompts


def save_prompt_debug_visualizations(image_data, label_dict, prompts_dict, output_dir, case_name, segs_dict=None, middle_slice=None):
    """
    Erzeugt Debug-Visualisierungen für GTV-Slices mit Prompts und speichert sie im Ausgabeverzeichnis.
    Erstellt eine Collage zur Übersicht der Prompt-Propagation.
    
    Args:
        image_data: 3D-Bilddaten
        label_dict: Dictionary mit Label-Masken (inkl. GT-Maske)
        prompts_dict: Dictionary mit generierten Prompts pro Slice
        output_dir: Ausgabeverzeichnis
        case_name: Name des Falls
        segs_dict: Dictionary mit neuen Segmentierungsmasken pro Label (optional)
        middle_slice: Index des mittleren Slices (für Propagation)
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
    gtv_slices = [z for z in range(image_data.shape[0]) if np.any(gtv_mask[z])]
    
    if not gtv_slices:
        print("Keine GTV-Slices für Debug-Visualisierungen gefunden")
        return
    
    # Falls middle_slice nicht angegeben, nehmen wir die Mitte der GTV-Slices
    if middle_slice is None and gtv_slices:
        middle_slice = gtv_slices[len(gtv_slices) // 2]
    
    # Erstelle Collage-Visualisierung
    if gtv_slices:
        create_prompt_propagation_overview(
            image_data, gtv_mask, prompts_dict, segs_dict, 
            output_dir, case_name, 
            middle_slice=middle_slice,
            gtv_slices=gtv_slices
        )


def visualize_prompts(image, gt_mask, prompts, output_path, slice_idx=None, slice_type=None):
    """
    Visualisiert einen einzelnen Slice mit GT-Maske und angewendeten Prompts.
    
    Args:
        image: 2D-Bild
        gt_mask: 2D-Ground-Truth-Maske
        prompts: Dictionary mit Prompt-Informationen
        output_path: Pfad zum Speichern der Visualisierung
        slice_idx: Slice-Index für die Beschriftung (optional)
        slice_type: Typ des Slices (Middle oder Propagiert)
    """
    # Normalisiere das Bild für eine bessere Darstellung
    if len(image.shape) == 3 and image.shape[2] == 3:  # Falls bereits 3 Kanäle
        img_display = image.copy()
    else:  # Einkanal-Bild zu RGB konvertieren
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        img_display = np.repeat(img_norm[:, :, np.newaxis], 3, axis=2)
    
    # Erstelle Subplot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Zeige das Bild
    ax.imshow(img_display, cmap='gray')
    
    # Overlay der GT-Maske (transparent gelb)
    gt_overlay = np.zeros_like(img_display)
    if len(gt_overlay.shape) == 3:
        gt_overlay[gt_mask > 0, 0] = 1.0  # Rot-Kanal
        gt_overlay[gt_mask > 0, 1] = 1.0  # Grün-Kanal
        gt_overlay[gt_mask > 0, 2] = 0.0  # Blau-Kanal
        ax.imshow(gt_overlay, alpha=0.3)
    
    # Visualisiere Box-Prompts
    if 'box' in prompts:
        box = prompts['box']
        if box is not None:
            rect = plt.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
    
    # Visualisiere Point-Prompts
    if 'points' in prompts and 'labels' in prompts:
        points = prompts['points']
        labels = prompts['labels']
        
        if len(points) > 0:
            for i, (point, label) in enumerate(zip(points, labels)):
                # Positive Punkte in Blau, negative in Rot
                color = 'blue' if label == 1 else 'red'
                marker = 'x' if label == 1 else 'x'
                size = 100 if label == 1 else 80
                ax.scatter(point[0], point[1], c=color, marker=marker, s=size)
    
    # Füge Informationen zum Slice hinzu
    title = f"Slice {slice_idx}" if slice_idx is not None else "Slice"
    if slice_type:
        title += f" - {slice_type}"
    ax.set_title(title, fontsize=14)
    
    # Füge Legende hinzu
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color='blue', markersize=10, label='Positiver Punkt'),
        Line2D([0], [0], marker='x', color='red', markersize=10, label='Negativer Punkt'),
        plt.Rectangle((0, 0), 1, 1, fc='yellow', alpha=0.3, label='GT-Maske'),
        plt.Rectangle((0, 0), 1, 1, fc='none', ec='red', label='Bounding Box')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Entferne Achsen
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Speichere die Figur
    #plt.tight_layout()
    #plt.savefig(output_path, dpi=150, bbox_inches='tight')
    #plt.close()


def create_prompt_propagation_overview(image_data, gtv_mask, prompts_dict, segs_dict, output_dir, case_name, middle_slice=None, gtv_slices=None, prompt_type='box', zoom_to_mask=True, padding=40):
    if gtv_slices is None or len(gtv_slices) == 0:
        return
    
    # Sortiere die GTV-Slices
    gtv_slices = sorted(gtv_slices)
    
    # Bestimme das mittlere Slice
    if middle_slice is None or middle_slice not in gtv_slices:
        middle_slice = gtv_slices[len(gtv_slices) // 2]
    
    # Wähle 9 eindeutige Slices
    if len(gtv_slices) <= 9:
        selected_slices = gtv_slices.copy()
    else:
        step = (len(gtv_slices) - 1) / 8
        indices = [int(i * step) for i in range(9)]
        selected_slices = [gtv_slices[idx] for idx in indices]
        if middle_slice not in selected_slices:
            closest_idx = min(range(9), key=lambda i: abs(selected_slices[i] - middle_slice))
            selected_slices[closest_idx] = middle_slice
    
    selected_slices = list(dict.fromkeys(selected_slices))[:9]
    selected_slices.sort()
    
    # Erstelle die Collage
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()
    
    for i, z in enumerate(selected_slices):
        ax = axes[i]
        img_slice = image_data[z]
        
        # Bild vorbereiten
        if len(img_slice.shape) == 3 and img_slice.shape[2] == 3:
            img_display = img_slice.copy()
        else:
            img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8) * 1.5
            img_norm = np.clip(img_norm, 0, 1)
            img_display = np.repeat(img_norm[:, :, np.newaxis], 3, axis=2)
        
        # Zoom auf Segmentierungsmaske, falls aktiviert
        if zoom_to_mask:
            # Verwende Segmentierungsmaske, falls vorhanden, sonst GT-Maske
            mask_to_use = segs_dict[1][z] if segs_dict and 1 in segs_dict and np.any(segs_dict[1][z]) else gtv_mask[z]
            if np.any(mask_to_use):
                # Bildgröße abrufen
                img_h, img_w = img_display.shape[:2]

                if z == middle_slice:
                    # Verwende Ground-Truth-Maske als Zentrum für das Zoom
                    y_indices, x_indices = np.where(mask_to_use > 0)
                    x_center = np.mean(x_indices).astype(int)
                    y_center = np.mean(y_indices).astype(int)

                    # Zoom-Faktor (einstellbar)
                    zoom_factor = 2  

                    # Berechne das gewünschte quadratische Sichtfeld
                    square_size = min(img_h, img_w) // zoom_factor

                else:
                    # Bounding Box der Maske berechnen
                    y_indices, x_indices = np.where(mask_to_use > 0)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)

                    # Padding hinzufügen
                    x_min = max(0, x_min - padding)
                    x_max = min(img_w - 1, x_max + padding)
                    y_min = max(0, y_min - padding)
                    y_max = min(img_h - 1, y_max + padding)

                    # Berechne das gewünschte quadratische Sichtfeld
                    square_size = max(x_max - x_min, y_max - y_min)  # Nimmt größeres Maß für Quadrat

                    # Zentrum setzen
                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2

                # Quadratischen Bildausschnitt berechnen
                x_min = max(0, x_center - square_size // 2)
                x_max = min(img_w, x_center + square_size // 2)
                y_min = max(0, y_center - square_size // 2)
                y_max = min(img_h, y_center + square_size // 2)

                # Setze Achsenlimits für konsistente quadratische Bilder
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_max, y_min)  # y-Achse ist invertiert in matplotlib
        # Bild anzeigen
        ax.imshow(img_display, cmap='gray')
        
        # Segmentierungsmaske (hellblau)
        if segs_dict and 1 in segs_dict:
            seg_slice = segs_dict[1][z]
            if np.any(seg_slice):
                # Erstelle ein RGBA-Bild für die Überlagerung
                seg_overlay = np.zeros((seg_slice.shape[0], seg_slice.shape[1], 4))  # 4 Kanäle: R, G, B, A
                seg_overlay[seg_slice > 0, 0] = 0.0  # Rot
                seg_overlay[seg_slice > 0, 1] = 0.5  # Grün
                seg_overlay[seg_slice > 0, 2] = 1.0  # Blau
                seg_overlay[seg_slice > 0, 3] = 0.8  # Alpha (Deckkraft)
                ax.imshow(seg_overlay)
        
        # GT-Maske (grün)
        if gtv_mask is not None:
            gtv_slice = gtv_mask[z]
            gtv_overlay = np.zeros_like(img_display)
            gtv_overlay[gtv_slice > 0, 1] = 1.0
            ax.imshow(gtv_overlay, alpha=0.3)
        
        # Prompts anzeigen
        if z in prompts_dict:
            prompts = prompts_dict[z]
            if z == middle_slice:
                if 'points' in prompts:
                    points = prompts['points']
                    labels = prompts['labels']
                    for point, label in zip(points, labels):
                        color = 'purple' if label == 1 else 'orange'
                        marker = 'x'
                        ax.scatter(point[0], point[1], c=color, marker=marker, s=100, linewidths=0.75)
                elif prompt_type == 'box' and 'box' in prompts:
                    box = prompts['box']
                    rect = plt.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        linewidth=2, edgecolor='yellow', facecolor='none'
                    )
                    ax.add_patch(rect)
            elif 'box' in prompts and prompt_type == 'box':
                box = prompts['box']
                rect = plt.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
        
        # Titel
        title = f"Slice {z}"
        if z == middle_slice:
            title += " (MIDDLE)"
            ax.set_title(title, fontsize=12, color='red', fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
        else:
            ax.set_title(title, fontsize=12)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Legende
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='GT-Maske'),
        Patch(facecolor='lightblue', alpha=0.8, label='Segmentierungsmaske'),
        Patch(facecolor='none', edgecolor='yellow', label='Bounding Box (GT)'),
        Patch(facecolor='none', edgecolor='red', label='Bounding Box (propagiert)'),
        Line2D([0], [0], marker='x', color='purple', markersize=10, label='Positiver Punkt'),
        Line2D([0], [0], marker='x', color='orange', markersize=10, label='Negativer Punkt')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3)
    
    # Speichern
    output_path = os.path.join(output_dir, f"{case_name}_prompt_propagation_overview.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()