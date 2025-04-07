#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DICOM Workflow Prototyp

Überwacht einen Ordner auf neue DICOM-Dateien (CT und RS), konvertiert sie in NPZ,
führt Inferenz mit MedSAM2 durch und konvertiert die Ergebnisse zurück in DICOM.

Workflow:
1. Überwachung des Eingangsordners
2. Konvertierung von CT und RS in NPZ
3. Extraktion der mittleren Slice des GTV (Label 1) für den Prompt
4. Inferenz mit MedSAM2 mix2
5. Konvertierung der Ergebnisse zurück in DICOM
6. Speicherung im Ausgabeordner
"""

import os
import sys
import time
import glob
import shutil
import argparse
import numpy as np
import pydicom
from pydicom.uid import generate_uid
import SimpleITK as sitk
from tqdm import tqdm
import torch
import logging
import json
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

# Eigene Module importieren
from dicom_converter import convert_dicom_to_npz
from prompt_utils import create_prompt_propagation_overview

# Logger einrichten
logger = logging.getLogger("DicomWorkflow")
logger.setLevel(logging.INFO)

# Konsolenausgabe
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def setup_file_logger(output_dir):
    """
    Richtet einen Logger ein, der in eine Datei im Output-Ordner schreibt.
    
    Args:
        output_dir: Ausgabeverzeichnis
    """
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"workflow_{time.strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Log-Datei erstellt: {log_file}")
    return file_handler

# Globale Variablen
TEMP_DIR = "temp_processing"
DEBUG = True  # Debug-Modus für Visualisierungen

def find_middle_slice_with_gtv(label_data):
    """
    Findet die mittlere Slice mit GTV (Label 1).
    
    Args:
        label_data: 3D-Array mit Labeldaten
        
    Returns:
        Index der mittleren Slice mit GTV
    """
    # GTV-Maske extrahieren (Label 1)
    gtv_mask = (label_data == 1)
    
    # Slices mit GTV finden
    slices_with_gtv = []
    for z in range(gtv_mask.shape[0]):
        if np.any(gtv_mask[z]):
            slices_with_gtv.append(z)
    
    if not slices_with_gtv:
        logger.warning("Kein GTV (Label 1) in den Daten gefunden!")
        return None
    
    # Mittlere Slice mit GTV zurückgeben
    middle_idx = len(slices_with_gtv) // 2
    middle_slice = slices_with_gtv[middle_idx]
    
    logger.info(f"Mittlere GTV-Slice: {middle_slice} (von {len(slices_with_gtv)} Slices mit GTV)")
    return middle_slice

def create_debug_visualizations(npz_path, result_path, output_dir):
    """
    Erstellt Debug-Visualisierungen für den Workflow.
    
    Args:
        npz_path: Pfad zur Original-NPZ-Datei
        result_path: Pfad zur Ergebnis-NPZ-Datei
        output_dir: Ausgabeverzeichnis
    """
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # NPZ-Dateien laden
    npz_data = np.load(npz_path, allow_pickle=True)
    result_data = np.load(result_path, allow_pickle=True)
    
    # Daten extrahieren
    image_data = npz_data['image']
    label_data = npz_data['gts']
    
    # Segmentierungsmaske extrahieren
    seg_mask = None
    for key in ['pred', 'GTV', '1', 'Organ_1']:
        if key in result_data:
            seg_mask = result_data[key]
            break
    
    if seg_mask is None:
        logger.error("Keine Segmentierungsmaske für Debug-Visualisierung gefunden!")
        return
    
    # GTV-Maske extrahieren (Label 1)
    gtv_mask = (label_data == 1)
    
    # Mittlere Slice finden
    middle_slice = find_middle_slice_with_gtv(label_data)
    if middle_slice is None:
        logger.error("Keine GTV-Slice für Debug-Visualisierung gefunden!")
        return
    
    # Slices mit GTV finden
    gtv_slices = []
    for z in range(gtv_mask.shape[0]):
        if np.any(gtv_mask[z]):
            gtv_slices.append(z)
    
    # Label-Dictionary für die Visualisierung erstellen
    label_dict = {
        'gtv': gtv_mask,
        'pred': seg_mask
    }
    
    # Prompts-Dictionary für die Visualisierung erstellen
    # (Wir haben keine echten Prompts, aber wir können ein leeres Dictionary verwenden)
    prompts_dict = {}
    
    # Fallname aus dem Dateipfad extrahieren
    case_name = os.path.basename(npz_path).replace('.npz', '')
    
    # Collage-Ansicht erstellen
    logger.info("Erstelle Debug-Visualisierung (Collage-Ansicht)...")
    create_prompt_propagation_overview(
        image_data,
        gtv_mask,
        prompts_dict,
        {'pred': seg_mask},
        debug_dir,
        case_name,
        middle_slice=middle_slice,
        gtv_slices=gtv_slices,
        prompt_type='box',
        zoom_to_mask=True,
        padding=40
    )
    
    logger.info(f"Debug-Visualisierung gespeichert in: {debug_dir}")

def run_inference(npz_path, output_dir, sam2_checkpoint, medsam2_checkpoint, model_cfg, 
               bbox_shift=5, debug=True, prompt_type="box", num_pos_points=3, num_neg_points=1, min_dist_from_edge=3):
    """
    Führt die Inferenz mit MedSAM2 durch.
    
    Args:
        npz_path: Pfad zur NPZ-Datei
        output_dir: Ausgabeverzeichnis
        sam2_checkpoint: Pfad zum SAM2-Checkpoint
        medsam2_checkpoint: Pfad zum MedSAM2-Checkpoint
        model_cfg: Pfad zur Modellkonfiguration
        bbox_shift: Versatz für die Bounding Box (nur für Box-Prompts)
        debug: Debug-Modus aktivieren
        prompt_type: Typ des Prompts ("box" oder "point")
        num_pos_points: Anzahl positiver Punkte (nur für Point-Prompts)
        num_neg_points: Anzahl negativer Punkte (nur für Point-Prompts)
        min_dist_from_edge: Mindestabstand vom Maskenrand (nur für Point-Prompts)
        
    Returns:
        Pfad zur Ergebnisdatei
    """
    logger.info(f"Starte Inferenz für: {npz_path}")
    
    # Temporäres Verzeichnis für die Inferenz
    npz_dir = os.path.join(output_dir, "npz_temp")
    os.makedirs(npz_dir, exist_ok=True)
    
    # NPZ-Datei in das temporäre Verzeichnis kopieren
    npz_name = os.path.basename(npz_path)
    npz_temp_path = os.path.join(npz_dir, npz_name)
    shutil.copy(npz_path, npz_temp_path)
    
    try:
        # Inferenz-Befehl zusammenstellen
        cmd = [
            "python", 
            "eval_inference/infer_medsam2_2d.py", 
            "-data_root", npz_dir, 
            "-pred_save_dir", os.path.join(output_dir, "predictions"), 
            "-sam2_checkpoint", sam2_checkpoint, 
            "-medsam2_checkpoint", medsam2_checkpoint, 
            "-model_cfg", model_cfg, 
            "-num_workers", "1", 
            "-prompt_type", prompt_type, 
            "--label", "1"  # GTV ist immer Label 1
        ]
        
        # Parameter je nach Prompt-Typ hinzufügen
        if prompt_type == "box":
            cmd.extend(["-bbox_shift", str(bbox_shift)])
        elif prompt_type == "point":
            cmd.extend([
                "-num_pos_points", str(num_pos_points),
                "-num_neg_points", str(num_neg_points),
                "-min_dist_from_edge", str(min_dist_from_edge)
            ])
        
        # Debug-Flag hinzufügen, wenn aktiviert
        if debug:
            cmd.append("-debug_mode")
        
        logger.info(f"Starte Inferenz mit Befehl: {' '.join(cmd)}")
        
        # Inferenz ausführen
        subprocess.run(cmd, check=True)
        
        # Ergebnisdatei finden - wir müssen nach allen NPZ-Dateien im Ergebnisverzeichnis suchen
        pred_dir = os.path.join(output_dir, "predictions")
        result_files = glob.glob(os.path.join(pred_dir, "*.npz"))
        
        # Wir suchen nach der neuesten Datei, die den Basisnamen enthält
        base_name = os.path.splitext(npz_name)[0]
        matching_files = [f for f in result_files if base_name in f]
        
        if matching_files:
            # Sortiere nach Erstellungszeit (neueste zuerst)
            matching_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            result_path = matching_files[0]
            logger.info(f"Inferenz erfolgreich: {result_path}")
            # HIER NEU: Übertrage spacing und origin
            original_data = np.load(npz_path, allow_pickle=True)
            result_data = np.load(result_path, allow_pickle=True)

            # Neues Dictionary mit allen benötigten Daten
            updated_result_dict = dict(result_data)
            updated_result_dict['spacing'] = original_data['spacing']
            updated_result_dict['origin'] = original_data['origin']
            updated_result_dict['direction'] = original_data['direction']

            # Ergebnisdatei überschreiben
            np.savez_compressed(result_path, **updated_result_dict)
            logger.info(f"Metadaten (spacing, origin) erfolgreich hinzugefügt zu: {result_path}")
            return result_path
        else:
            logger.error(f"Keine Ergebnisdatei gefunden in {pred_dir}!")
            # Liste alle Dateien im Verzeichnis auf
            all_files = os.listdir(pred_dir)
            logger.info(f"Dateien im Verzeichnis: {all_files}")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Fehler bei der Inferenz: {e}")
        return None
    finally:
        # Temporäres Verzeichnis aufräumen
        shutil.rmtree(npz_dir)

def convert_npz_to_dicom(npz_path, original_rs_path, output_dir):
    """
    Konvertiert die NPZ-Ergebnisse zurück in DICOM.
    
    Args:
        npz_path: Pfad zur NPZ-Ergebnisdatei
        original_rs_path: Pfad zur ursprünglichen RS-Datei
        output_dir: Ausgabeverzeichnis
        
    Returns:
        Pfad zur neuen RS-Datei
    """
    logger.info(f"Konvertiere NPZ zurück zu DICOM: {npz_path}")
    
    # NPZ-Datei laden
    npz_data = np.load(npz_path, allow_pickle=True)
    # Übersicht ausgeben
    print("Keys und Daten aus der NPZ-Datei:")
    for key in npz_data:
        data = npz_data[key]
        if isinstance(data, np.ndarray):
            print(f"- {key}: type={data.dtype}, shape={data.shape}")
        else:
            print(f"- {key}: {data}")
    
    # Originale RS-Datei laden
    rs_dcm = pydicom.dcmread(original_rs_path)
    
    # Neue UID für die Struktur generieren
    new_sop_instance_uid = generate_uid()
    rs_dcm.SOPInstanceUID = new_sop_instance_uid
    
    # Segmentierungsmaske extrahieren
    # Suche nach der Segmentierung in verschiedenen möglichen Schlüsseln
    seg_mask = None
    for key in ['pred', 'GTV', '1', 'Organ_1']:
        if key in npz_data:
            seg_mask = npz_data[key]
            logger.info(f"Segmentierung unter Schlüssel '{key}' gefunden")
            break
    
    if seg_mask is None:
        logger.error("Keine Segmentierungsmaske in der NPZ-Datei gefunden!")
        return None
    
    # Bildgeometrie und Spacing aus der NPZ-Datei extrahieren
    spacing = None
    origin = None
    direction = None
    if 'spacing' in npz_data:
        spacing = npz_data['spacing']
        print(f"Spacing: {spacing}")
        logger.info(f"Spacing aus NPZ-Datei: {spacing}")
    else:
        spacing=(0.9765625, 0.9765625, 3.0)
        print(f"Spacing Fallback: {spacing}")
    if 'origin' in npz_data:
        origin = npz_data['origin']
        print(f"Origin: {origin}")
        logger.info(f"Origin aus NPZ-Datei: {origin}")
    else:
        origin = (-249.51171875, -500.51171875, -621.0)
        print(f"Origin Fallback: {origin}")
    if 'direction' in npz_data:
        direction = npz_data['direction']
        print(f"Direction: {direction}")
        logger.info(f"Direction aus NPZ-Datei: {direction}")
    else:
        direction = rs_dcm.Direction
        print(f"Direction Fallback: {direction}")
    
    # Neue Kontur für die Segmentierung erstellen
    # Hier müssen wir die Konturen für jede Slice extrahieren
    
    # ROI-Nummer und ROI-Name für die neue Struktur festlegen
    roi_number = len(rs_dcm.StructureSetROISequence) + 1
    roi_name = "GTV_STARDUST_MIX2"
    
    # Neue ROI zum StructureSetROISequence hinzufügen
    new_roi = pydicom.dataset.Dataset()
    new_roi.ROINumber = roi_number
    new_roi.ROIName = roi_name
    new_roi.ROIGenerationAlgorithm = "AUTOMATIC"
    rs_dcm.StructureSetROISequence.append(new_roi)
    
    # Neue ROI-Kontur zum ROIContourSequence hinzufügen
    new_roi_contour = pydicom.dataset.Dataset()
    new_roi_contour.ROIDisplayColor = [255, 0, 0]  # Rot
    new_roi_contour.ReferencedROINumber = roi_number
    new_roi_contour.ContourSequence = []
    
    # Extrahiere Referenz-Kontur aus der Original-RS-Datei
    ref_contour_image_sequence = None
    try:
        if hasattr(rs_dcm, 'ROIContourSequence') and len(rs_dcm.ROIContourSequence) > 0:
            first_roi = rs_dcm.ROIContourSequence[0]
            if hasattr(first_roi, 'ContourSequence') and len(first_roi.ContourSequence) > 0:
                first_contour = first_roi.ContourSequence[0]
                if hasattr(first_contour, 'ContourImageSequence'):
                    ref_contour_image_sequence = first_contour.ContourImageSequence
    except Exception as e:
        logger.warning(f"Fehler beim Extrahieren der Referenz-Kontur: {e}")
    
    # Extrahiere Z-Koordinaten aus der Original-RS-Datei
    z_coords = []
    try:
        for roi in rs_dcm.ROIContourSequence:
            if hasattr(roi, 'ContourSequence'):
                for contour in roi.ContourSequence:
                    if hasattr(contour, 'ContourData') and len(contour.ContourData) >= 3:
                        # Z-Koordinate ist jeder dritte Wert, beginnend mit Index 2
                        z = contour.ContourData[2]
                        if z not in z_coords:
                            z_coords.append(z)
        
        # Sortiere Z-Koordinaten
        z_coords.sort()
        logger.info(f"Extrahierte Z-Koordinaten aus RS-Datei: {len(z_coords)} Werte")
    except Exception as e:
        logger.warning(f"Fehler beim Extrahieren der Z-Koordinaten: {e}")
        z_coords = []
    
    # Wenn keine Z-Koordinaten gefunden wurden, generiere sie basierend auf Spacing
    if not z_coords:
        z_coords = [origin[2] + z * spacing[2] for z in range(seg_mask.shape[0])]
        logger.warning(f"Keine Z-Koordinaten gefunden, generiere {len(z_coords)} basierend auf Spacing")
    
    # Stelle sicher, dass wir genug Z-Koordinaten haben
    if len(z_coords) < seg_mask.shape[0]:
        # Erweitere die Z-Koordinaten, falls nötig
        z_step = spacing[2] if len(z_coords) > 1 else 3.0
        while len(z_coords) < seg_mask.shape[0]:
            next_z = z_coords[-1] + z_step
            z_coords.append(next_z)
        logger.warning(f"Z-Koordinaten erweitert auf {len(z_coords)} Werte")
    
    # Konturen für jede Slice extrahieren und hinzufügen
    for z_idx in range(seg_mask.shape[0]):
        slice_mask = seg_mask[z_idx]
        if not np.any(slice_mask):
            continue  # Überspringe leere Slices
        
        # Z-Koordinate für diese Slice
        z_coord = z_coords[z_idx] if z_idx < len(z_coords) else z_coords[-1]
        
        # Konturen aus der Maske extrahieren
        contours, _ = cv2.findContours(
            slice_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Jede Kontur als eigenes Kontur-Dataset hinzufügen
        for contour in contours:
            if len(contour) < 3:
                continue  # Ignoriere zu kleine Konturen
            
            contour_dataset = pydicom.dataset.Dataset()
            contour_dataset.ContourGeometricType = "CLOSED_PLANAR"
            
            # Konturpunkte in DICOM-Format konvertieren
            contour_data = []
            for point in contour:
                x_pixel, y_pixel = point[0]
                
                # Direkte Übernahme der Koordinaten aus der Original-Kontur
                # Wir verwenden die gleiche Skalierung wie in der Original-Kontur,
                # anstatt eine Umrechnung mit Spacing vorzunehmen
                
                # Berechne die physikalischen Koordinaten basierend auf dem Origin
                # und der Position im Bild
                # Korrekte Umrechnung der Konturpunkte:
                x_phys = origin[0] + x_pixel * spacing[0]
                y_phys = origin[1] + y_pixel * spacing[1]
                
                # Verwende die exakte Z-Koordinate aus der Original-Kontur
                contour_data.extend([float(x_phys), float(y_phys), float(z_coord)])
            
            # Versuche, die Kontur zu schließen (erster und letzter Punkt identisch)
            if len(contour) >= 3:
                first_point = contour[0][0]
                last_point = contour[-1][0]
                if first_point[0] != last_point[0] or first_point[1] != last_point[1]:
                    # Füge den ersten Punkt am Ende hinzu, um die Kontur zu schließen
                    x_pixel, y_pixel = first_point
                    x_phys = origin[0] + x_pixel * spacing[0]
                    y_phys = origin[1] + y_pixel * spacing[1]
                    contour_data.extend([float(x_phys), float(y_phys), float(z_coord)])
                    # Erhöhe die Anzahl der Konturpunkte
                    contour_dataset.NumberOfContourPoints = len(contour) + 1
                else:
                    contour_dataset.NumberOfContourPoints = len(contour)
            else:
                contour_dataset.NumberOfContourPoints = len(contour)
            
            contour_dataset.ContourData = contour_data
            
            # Referenz zur CT-Serie hinzufügen
            if ref_contour_image_sequence:
                contour_dataset.ContourImageSequence = ref_contour_image_sequence
            
            new_roi_contour.ContourSequence.append(contour_dataset)
    
    # Neue ROI-Kontur zum ROIContourSequence hinzufügen
    rs_dcm.ROIContourSequence.append(new_roi_contour)
    
    # Neue RT ROI Observations Sequence hinzufügen
    new_rt_roi_obs = pydicom.dataset.Dataset()
    new_rt_roi_obs.ObservationNumber = roi_number
    new_rt_roi_obs.ReferencedROINumber = roi_number
    new_rt_roi_obs.ROIObservationLabel = roi_name
    new_rt_roi_obs.RTROIInterpretedType = "ORGAN"
    
    # Wenn die Sequenz noch nicht existiert, erstellen
    if not hasattr(rs_dcm, 'RTROIObservationsSequence'):
        rs_dcm.RTROIObservationsSequence = pydicom.sequence.Sequence()
    
    rs_dcm.RTROIObservationsSequence.append(new_rt_roi_obs)
    
    # Ausgabeverzeichnis erstellen
    os.makedirs(output_dir, exist_ok=True)
    
    # Neue RS-Datei speichern
    output_path = os.path.join(output_dir, f"AI_RS.{new_sop_instance_uid}.dcm")
    rs_dcm.save_as(output_path)
    
    logger.info(f"Neue RS-Datei gespeichert: {output_path}")
    
    return output_path

class DicomHandler(FileSystemEventHandler):
    """
    Handler für die Überwachung des Eingangsordners.
    """
    def __init__(self, input_dir, output_dir, sam2_checkpoint, medsam2_checkpoint, model_cfg, 
                 bbox_shift=5, keep_originals=False, prompt_type="box", 
                 num_pos_points=3, num_neg_points=1, min_dist_from_edge=3, debug=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.temp_dir = os.path.join(output_dir, TEMP_DIR)
        self.sam2_checkpoint = sam2_checkpoint
        self.medsam2_checkpoint = medsam2_checkpoint
        self.model_cfg = model_cfg
        self.bbox_shift = bbox_shift
        self.keep_originals = keep_originals
        self.prompt_type = prompt_type
        self.num_pos_points = num_pos_points
        self.num_neg_points = num_neg_points
        self.min_dist_from_edge = min_dist_from_edge
        self.debug = debug
        self.processed_files = set()  # Speichert bereits verarbeitete Dateien
        self.pending_cases = {}  # Speichert unvollständige Fälle: {case_id: {'ct': [ct_files], 'rs': rs_file}}
        self.last_file_event_time = 0  # Zeitpunkt des letzten Dateiereignisses
        self.process_delay = 2  # Sekunden Wartezeit nach dem letzten Dateiereignis
        
        # Temporäres Verzeichnis erstellen
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Ausgabeverzeichnisse erstellen
        os.makedirs(os.path.join(output_dir, "npz"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "dicom"), exist_ok=True)
        
        # Wenn Original-DICOM behalten werden sollen, Verzeichnis erstellen
        if self.keep_originals:
            os.makedirs(os.path.join(output_dir, "original_dicom"), exist_ok=True)
        
        # Datei-Logger einrichten
        self.file_handler = setup_file_logger(output_dir)
    
    def on_created(self, event):
        """
        Wird aufgerufen, wenn eine neue Datei im überwachten Ordner erstellt wird.
        
        Args:
            event: Dateiereignis
        """
        try:
            # Ignoriere Verzeichnisse und temporäre Dateien
            if event.is_directory or event.src_path.endswith(".tmp"):
                return
            
            # Ignoriere bereits verarbeitete Dateien
            if event.src_path in self.processed_files:
                return
            
            # Warte kurz, um sicherzustellen, dass die Datei vollständig geschrieben wurde
            time.sleep(0.1)
            
            # Prüfe, ob es sich um eine DICOM-Datei handelt
            if not os.path.exists(event.src_path):
                return
            
            try:
                dcm = pydicom.dcmread(event.src_path, force=True)
                # Aktualisiere den Zeitpunkt des letzten Dateiereignisses
                self.last_file_event_time = time.time()
                logger.info(f"Neue DICOM-Datei erkannt: {event.src_path}")
                
                # Füge die Datei zur Liste der ausstehenden Dateien hinzu
                self.add_to_pending_cases(event.src_path, dcm)
            except:
                # Keine DICOM-Datei oder beschädigt
                logger.debug(f"Ignoriere Nicht-DICOM-Datei: {event.src_path}")
                return
            
            # Wir verarbeiten nicht sofort, sondern aktualisieren nur den Zeitstempel
            # Die Verarbeitung erfolgt in der Hauptschleife, wenn genug Zeit vergangen ist
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung der neuen Datei {event.src_path}: {e}")
    
    def check_and_process_pending_cases(self):
        """
        Überprüft, ob ausstehende Fälle vollständig sind und verarbeitet sie.
        """
        # Prüfe, ob genug Zeit seit dem letzten Dateiereignis vergangen ist
        current_time = time.time()
        if current_time - self.last_file_event_time < self.process_delay:
            # Noch nicht genug Zeit vergangen, warte weiter
            return
        
        # Finde vollständige Fälle (mit CT und RS)
        complete_cases = []
        for case_id, case_data in list(self.pending_cases.items()):
            if 'ct' in case_data and 'rs' in case_data and len(case_data['ct']) > 0:
                complete_cases.append(case_id)
        
        # Verarbeite vollständige Fälle
        for case_id in complete_cases:
            logger.info(f"Fall {case_id} ist vollständig. Starte Verarbeitung...")
            self.process_case(case_id)
    
    def add_to_pending_cases(self, file_path, dcm):
        """
        Fügt eine Datei zu den ausstehenden Fällen hinzu.
        
        Args:
            file_path: Pfad zur Datei
            dcm: Geladene DICOM-Datei
        """
        # Versuche, eine eindeutige Patienten-ID zu bekommen
        if hasattr(dcm, 'PatientID') and dcm.PatientID:
            case_id = dcm.PatientID
        elif hasattr(dcm, 'StudyInstanceUID') and dcm.StudyInstanceUID:
            case_id = dcm.StudyInstanceUID
        else:
            # Fallback: Verwende den Dateinamen ohne Modalitätsprefix und Erweiterung
            filename = os.path.basename(file_path)
            case_id = filename.replace('CT', '').replace('RS', '').replace('.dcm', '')
        
        # Initialisiere den Fall, wenn er noch nicht existiert
        if case_id not in self.pending_cases:
            self.pending_cases[case_id] = {'ct': [], 'rs': None}
        
        # Füge die Datei hinzu
        if dcm.Modality == "CT":
            self.pending_cases[case_id]['ct'].append(file_path)
            logger.info(f"CT-Datei zu Fall {case_id} hinzugefügt. Aktuell {len(self.pending_cases[case_id]['ct'])} CT-Dateien.")
        elif dcm.Modality == "RTSTRUCT":
            self.pending_cases[case_id]['rs'] = file_path
            logger.info(f"RS-Datei zu Fall {case_id} hinzugefügt.")
    
    def process_case(self, case_id):
        """
        Verarbeitet einen vollständigen Fall (CT + RS).
        
        Args:
            case_id: ID des Falls
        """
        case_data = self.pending_cases[case_id]
        ct_files = case_data['ct']
        rs_file = case_data['rs']
        
        # Temporäres Verzeichnis für diesen Fall erstellen
        case_temp_dir = os.path.join(self.temp_dir, case_id)
        
        # Falls das Verzeichnis bereits existiert, löschen und neu erstellen
        if os.path.exists(case_temp_dir):
            shutil.rmtree(case_temp_dir)
        os.makedirs(case_temp_dir)
        
        logger.info(f"Verarbeite Fall: {case_id}")
        
        try:
            # CT-Dateien in das temporäre Verzeichnis kopieren
            for ct_file in ct_files:
                shutil.copy(ct_file, case_temp_dir)
                self.processed_files.add(ct_file)
            
            # RS-Datei in das temporäre Verzeichnis kopieren
            rs_temp_path = os.path.join(case_temp_dir, os.path.basename(rs_file))
            shutil.copy(rs_file, rs_temp_path)
            self.processed_files.add(rs_file)
            
            # Original-DICOM-Dateien in den Output-Ordner kopieren, wenn gewünscht
            if self.keep_originals:
                original_dicom_dir = os.path.join(self.output_dir, "original_dicom")
                
                # CT-Dateien kopieren
                for ct_file in ct_files:
                    ct_dest = os.path.join(original_dicom_dir, os.path.basename(ct_file))
                    shutil.copy(ct_file, ct_dest)
                
                # RS-Datei kopieren
                rs_dest = os.path.join(original_dicom_dir, os.path.basename(rs_file))
                shutil.copy(rs_file, rs_dest)
                
                logger.info(f"Original-DICOM-Dateien gespeichert in: {original_dicom_dir}")
            
            # NPZ konvertieren
            npz_dir = os.path.join(self.output_dir, "npz")
            npz_path = convert_dicom_to_npz(case_temp_dir, npz_dir, case_id, save_nii=False)
            
            if npz_path and os.path.exists(npz_path):
                # Inferenz durchführen
                result_path = run_inference(
                    npz_path, 
                    self.output_dir, 
                    self.sam2_checkpoint, 
                    self.medsam2_checkpoint, 
                    self.model_cfg,
                    self.bbox_shift,
                    debug=self.debug,
                    prompt_type=self.prompt_type,
                    num_pos_points=self.num_pos_points,
                    num_neg_points=self.num_neg_points,
                    min_dist_from_edge=self.min_dist_from_edge
                )
                
                if result_path and os.path.exists(result_path):
                    # Zurück zu DICOM konvertieren
                    dicom_dir = os.path.join(self.output_dir, "dicom")
                    dicom_path = convert_npz_to_dicom(result_path, rs_temp_path, dicom_dir)
                    
                    if dicom_path and os.path.exists(dicom_path):
                        logger.info(f"Workflow abgeschlossen für {case_id}")
                        
                        # Dateien aus dem Eingangsordner entfernen - IMMER, unabhängig von keep_originals
                        self.cleanup_input_files(ct_files, rs_file)
                        
                        # Fall aus der Liste der ausstehenden Fälle entfernen
                        del self.pending_cases[case_id]
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung von Fall {case_id}: {e}")
            # Auch bei Fehlern aufräumen
            self.cleanup_input_files(ct_files, rs_file)
        finally:
            # Temporäres Verzeichnis aufräumen
            if os.path.exists(case_temp_dir):
                shutil.rmtree(case_temp_dir)
    
    def cleanup_input_files(self, ct_files, rs_file):
        """
        Entfernt alle Dateien aus dem Eingangsordner.
        
        Args:
            ct_files: Liste der CT-Dateien
            rs_file: Pfad zur RS-Datei
        """
        logger.info("Räume Eingangsordner auf...")
        
        # CT-Dateien entfernen
        for ct_file in ct_files:
            try:
                if os.path.exists(ct_file):
                    os.remove(ct_file)
                    logger.debug(f"Datei entfernt: {ct_file}")
            except Exception as e:
                logger.warning(f"Fehler beim Entfernen von {ct_file}: {e}")
        
        # RS-Datei entfernen
        try:
            if os.path.exists(rs_file):
                os.remove(rs_file)
                logger.debug(f"Datei entfernt: {rs_file}")
        except Exception as e:
            logger.warning(f"Fehler beim Entfernen von {rs_file}: {e}")
        
        # Prüfen, ob leere Verzeichnisse zurückbleiben und diese entfernen
        self.cleanup_empty_directories(self.input_dir)
        
        logger.info("Eingangsordner aufgeräumt")
    
    def cleanup_empty_directories(self, directory):
        """
        Entfernt rekursiv alle leeren Verzeichnisse.
        
        Args:
            directory: Zu prüfendes Verzeichnis
        """
        for root, dirs, files in os.walk(directory, topdown=False):
            # Überspringe das Hauptverzeichnis
            if root == directory:
                continue
                
            # Wenn das Verzeichnis leer ist (keine Dateien und keine Unterverzeichnisse mehr), entfernen
            if not os.listdir(root):
                try:
                    os.rmdir(root)
                    logger.debug(f"Leeres Verzeichnis entfernt: {root}")
                except Exception as e:
                    logger.warning(f"Fehler beim Entfernen des leeren Verzeichnisses {root}: {e}")
    
    def process_dicom_files(self):
        """
        Verarbeitet alle DICOM-Dateien im Eingangsordner.
        Sucht rekursiv nach CT- und RS-Dateien und gruppiert sie nach Fällen.
        """
        # Rekursive Suche nach allen DICOM-Dateien im Eingangsordner
        ct_pattern = os.path.join(self.input_dir, "**", "{CT,MR}*.dcm")
        rs_pattern = os.path.join(self.input_dir, "**", "RS*.dcm")
        
        ct_files = glob.glob(ct_pattern, recursive=True)
        rs_files = glob.glob(rs_pattern, recursive=True)
        
        if not ct_files:
            logger.info("Keine CT-Dateien gefunden.")
        if not rs_files:
            logger.info("Keine RS-Dateien gefunden.")
        
        if not ct_files or not rs_files:
            logger.info("Warte auf vollständige CT und RS Dateien...")
            return
        
        # Versuche, die Dateien nach Fällen zu gruppieren
        for rs_file in rs_files:
            # Überspringe bereits verarbeitete Dateien
            if rs_file in self.processed_files:
                continue
            
            try:
                # Lese die RS-Datei, um die Patienten-ID zu bekommen
                rs_dcm = pydicom.dcmread(rs_file)
                
                # Versuche, eine eindeutige Patienten-ID zu bekommen
                if hasattr(rs_dcm, 'PatientID') and rs_dcm.PatientID:
                    case_id = rs_dcm.PatientID
                elif hasattr(rs_dcm, 'StudyInstanceUID') and rs_dcm.StudyInstanceUID:
                    case_id = rs_dcm.StudyInstanceUID
                else:
                    # Fallback: Verwende den Dateinamen ohne Modalitätsprefix und Erweiterung
                    filename = os.path.basename(rs_file)
                    case_id = filename.replace('RS', '').replace('.dcm', '')
                
                # Initialisiere den Fall
                if case_id not in self.pending_cases:
                    self.pending_cases[case_id] = {'ct': [], 'rs': None}
                
                # Füge die RS-Datei hinzu
                self.pending_cases[case_id]['rs'] = rs_file
                
                # Suche nach zugehörigen CT-Dateien
                for ct_file in ct_files:
                    if ct_file in self.processed_files:
                        continue
                    
                    try:
                        ct_dcm = pydicom.dcmread(ct_file)
                        
                        # Prüfe, ob die CT-Datei zum selben Patienten gehört
                        if (hasattr(ct_dcm, 'PatientID') and ct_dcm.PatientID == case_id) or \
                           (hasattr(ct_dcm, 'StudyInstanceUID') and ct_dcm.StudyInstanceUID == case_id):
                            self.pending_cases[case_id]['ct'].append(ct_file)
                    except:
                        # Überspringe Dateien, die nicht gelesen werden können
                        continue
                
                # Wenn der Fall vollständig ist, verarbeite ihn
                if self.pending_cases[case_id]['ct'] and self.pending_cases[case_id]['rs']:
                    logger.info(f"Fall {case_id} ist vollständig. Starte Verarbeitung...")
                    self.process_case(case_id)
            
            except Exception as e:
                logger.warning(f"Fehler beim Lesen der DICOM-Datei {rs_file}: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description="DICOM Workflow Prototyp")
    parser.add_argument("--input_dir", type=str, default="Automate/Input", help="Eingangsordner für DICOM-Dateien")
    parser.add_argument("--output_dir", type=str, default="Automate/Output", help="Ausgabeordner für Ergebnisse")
    #parser.add_argument("--sam2_checkpoint", type=str, default="C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/checkpoints/sam2_hiera_tiny.pt", help="Pfad zum SAM2-Checkpoint")
    #parser.add_argument("--medsam2_checkpoint", type=str, default="work_dir/MedSAM2-Tiny-STARDUST_Mix2/best_checkpoint.pth", help="Pfad zum MedSAM2-Checkpoint")
    #parser.add_argument("--model_cfg", type=str, default="C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/sam2_configs/sam2_hiera_t.yaml", help="Pfad zur Modellkonfiguration")
    parser.add_argument("--sam2_checkpoint", type=str, default="C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/checkpoints/sam2.1_hiera_tiny.pt", help="Pfad zum SAM2-Checkpoint")
    parser.add_argument("--medsam2_checkpoint", type=str, default="work_dir/MedSAM2-Tiny-STARDUST_MixMix.1/best_checkpoint.pth", help="Pfad zum MedSAM2-Checkpoint")
    parser.add_argument("--model_cfg", type=str, default="C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/sam2_configs/sam2.1_hiera_t.yaml", help="Pfad zur Modellkonfiguration")
    parser.add_argument("--bbox_shift", type=int, default=5, help="Versatz für die Bounding Box")
    parser.add_argument("--keep_originals", action="store_true", help="Original-DICOM-Dateien im Output-Ordner behalten")
    parser.add_argument("--watch", action="store_true", help="Ordner kontinuierlich überwachen")
    parser.add_argument("--process_once", action="store_true", help="Einmalige Verarbeitung und dann beenden")
    parser.add_argument("--recursive", action="store_true", help="Rekursiv in Unterordnern suchen", default=True)
    parser.add_argument("--force_process", action="store_true", help="Sofortige Verarbeitung aller gefundenen Dateien erzwingen", default=True)
    parser.add_argument("--prompt_type", type=str, choices=["box", "point"], default="box", help="Prompt-Typ für die Inferenz (box oder point)")
    parser.add_argument("--num_pos_points", type=int, default=3, help="Anzahl positiver Punkte für Point-Prompts")
    parser.add_argument("--num_neg_points", type=int, default=1, help="Anzahl negativer Punkte für Point-Prompts")
    parser.add_argument("--min_dist_from_edge", type=int, default=3, help="Mindestabstand vom Maskenrand für Point-Prompts")
    parser.add_argument("--debug", action="store_true", default=True, help="Debug-Modus aktivieren für Prompt-Visualisierungen")
    args = parser.parse_args()
    
    # Absolute Pfade für Input und Output
    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # Verzeichnisse erstellen
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Datei-Logger einrichten
    file_handler = setup_file_logger(args.output_dir)
    
    # Globale Debug-Einstellung setzen
    global DEBUG
    DEBUG = args.debug
    
    logger.info(f"Starte DICOM Workflow mit folgenden Parametern:")
    logger.info(f"  Input-Ordner: {args.input_dir}")
    logger.info(f"  Output-Ordner: {args.output_dir}")
    logger.info(f"  SAM2-Checkpoint: {args.sam2_checkpoint}")
    logger.info(f"  MedSAM2-Checkpoint: {args.medsam2_checkpoint}")
    logger.info(f"  Modellkonfiguration: {args.model_cfg}")
    logger.info(f"  Bounding Box Shift: {args.bbox_shift}")
    logger.info(f"  Original-DICOM behalten: {args.keep_originals}")
    logger.info(f"  Prompt-Typ: {args.prompt_type}")
    if args.prompt_type == "point":
        logger.info(f"  Positive Punkte: {args.num_pos_points}")
        logger.info(f"  Negative Punkte: {args.num_neg_points}")
        logger.info(f"  Mindestabstand vom Rand: {args.min_dist_from_edge}")
    logger.info(f"  Debug-Modus: {args.debug}")
    
    if args.watch:
        # Ordner kontinuierlich überwachen
        logger.info(f"Überwache Ordner: {args.input_dir}")
        event_handler = DicomHandler(
            args.input_dir, 
            args.output_dir, 
            args.sam2_checkpoint, 
            args.medsam2_checkpoint, 
            args.model_cfg,
            args.bbox_shift,
            args.keep_originals,
            args.prompt_type,
            args.num_pos_points,
            args.num_neg_points,
            args.min_dist_from_edge,
            args.debug
        )
        observer = Observer()
        observer.schedule(event_handler, args.input_dir, recursive=args.recursive)
        observer.start()
        
        try:
            # Initiale Verarbeitung vorhandener Dateien
            logger.info("Starte initiale Verarbeitung vorhandener Dateien...")
            event_handler.process_dicom_files()
            
            # Wenn force_process gesetzt ist, verarbeiten wir alle ausstehenden Fälle sofort
            if args.force_process:
                logger.info("Erzwinge Verarbeitung aller ausstehenden Fälle...")
                event_handler.check_and_process_pending_cases()
            
            # Wenn process_once gesetzt ist, beenden wir nach der ersten Verarbeitung
            if args.process_once:
                logger.info("Einmalige Verarbeitung abgeschlossen, beende Programm.")
                observer.stop()
                observer.join()
                return
            
            # Warten auf neue Dateien
            logger.info("Warte auf neue Dateien... (Drücke Strg+C zum Beenden)")
            while True:
                time.sleep(0.5)  # Kürzere Wartezeit für schnellere Reaktion
                # Regelmäßig prüfen, ob es ausstehende Fälle gibt
                event_handler.check_and_process_pending_cases()
        except KeyboardInterrupt:
            logger.info("Benutzerabbruch erkannt, beende Programm...")
            observer.stop()
        finally:
            observer.join()
            # Logger aufräumen
            logger.removeHandler(file_handler)
    else:
        # Einmalige Verarbeitung
        logger.info("Einmalige Verarbeitung der Dateien im Eingangsordner")
        handler = DicomHandler(
            args.input_dir, 
            args.output_dir, 
            args.sam2_checkpoint, 
            args.medsam2_checkpoint, 
            args.model_cfg,
            args.bbox_shift,
            args.keep_originals,
            args.prompt_type,
            args.num_pos_points,
            args.num_neg_points,
            args.min_dist_from_edge,
            args.debug
        )
        handler.process_dicom_files()
        # Erzwinge Verarbeitung aller ausstehenden Fälle
        handler.check_and_process_pending_cases()
        # Logger aufräumen
        logger.removeHandler(file_handler)

if __name__ == "__main__":
    main()
