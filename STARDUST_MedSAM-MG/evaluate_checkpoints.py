#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Checkpoint Evaluation Script für SAM2 und MedSAM2

Dieses Skript evaluiert mehrere fine-getunte Checkpoints auf verschiedenen Testdatensätzen
und berechnet Metriken für alle Kombinationen. Die Ergebnisse werden in Excel-Dateien gespeichert.
"""

import os
import sys
import pandas as pd
import numpy as np
import subprocess
import time
import shutil
import glob
from pathlib import Path
from datetime import datetime
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Nicht-interaktiver Backend für Matplotlib
import seaborn as sns
import io
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from skimage import measure
import re

# Importiere prompt_utils für Debug-Visualisierungen
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from prompt_utils import save_prompt_debug_visualizations

# Definiere die Checkpoints und ihre zugehörigen Testdaten
CHECKPOINTS = {
    "Kopf": {
        "name": "MedSAM2-Tiny-STARDUST_Kopf",
        "path": "work_dir/MedSAM2-Tiny-STARDUST_Kopf/best_checkpoint.pth",
        "test_data": "data/npz_custom_Kopf/test"
    },
    "Lunge": {
        "name": "MedSAM2-Tiny-STARDUST_Lunge",
        "path": "work_dir/MedSAM2-Tiny-STARDUST_Lunge/best_checkpoint.pth",
        "test_data": "data/npz_custom_Lunge/test"
    },
    "Mix": {
        "name": "MedSAM2-Tiny-STARDUST_Mix",
        "path": "work_dir/MedSAM2-Tiny-STARDUST_Mix/best_checkpoint.pth",
        "test_data": "data/npz_custom_Mix/test"
    },
    "Mix2": {
        "name": "MedSAM2-Tiny-STARDUST_Mix2",
        "path": "work_dir/MedSAM2-Tiny-STARDUST_Mix2/best_checkpoint.pth",
        "test_data": "data/npz_custom_Mix2/test"
    },
    "Rest": {
        "name": "MedSAM2-Tiny-STARDUST_Rest",
        "path": "work_dir/MedSAM2-Tiny-STARDUST_Rest/best_checkpoint.pth",
        "test_data": "data/npz_custom_Rest/test"
    },    
    "MR-Mix": {
        "name": "MedSAM2-Tiny-STARDUST_MR",
        "path": "work_dir/MedSAM2-Tiny-STARDUST_MR/best_checkpoint.pth",
        "test_data": "data/npz_files_MR/test"
    },
    "MR-Glio": {
        "name": "MedSAM2-Tiny-STARDUST_MRglio",
        "path": "work_dir/MedSAM2-Tiny-STARDUST_MRglio/best_checkpoint.pth",
        "test_data": "data/npz_files_MRglio/test"
    },
    "MR+CT": {
        "name": "MedSAM2-Tiny-STARDUST_MixMix",
        "path": "work_dir/MedSAM2-Tiny-STARDUST_MixMix/best_checkpoint.pth",
        "test_data": "data/npz_files_MixMix/test"
    }
}
# Definiere die Metriknamen
METRICS = ["dice"]

def load_gui_settings():
    """Lade die GUI-Settings für den Vanilla-Checkpoint"""
    if os.path.exists("stardust_gui_settings.json"):
        with open("stardust_gui_settings.json", "r") as f:
            settings = json.load(f)
            # Konvertiere die spezifischen Box/Point-Parameter basierend auf der GUI-Struktur
            if "prompt_type" in settings:
                prompt_type = settings["prompt_type"]
                if prompt_type == "box" and "box_shift" in settings:
                    box_shift = settings["box_shift"]
                elif prompt_type == "point":
                    # Point-spezifische Einstellungen
                    if "num_pos_points" in settings:
                        num_pos_points = settings["num_pos_points"]
                    if "num_neg_points" in settings:
                        num_neg_points = settings["num_neg_points"]
                    if "min_dist_from_mask_edge" in settings:
                        min_dist_from_mask_edge = settings["min_dist_from_mask_edge"]
            return settings
    return {}

def get_test_cases_count(data_dir):
    """Zähle die Anzahl der Testfälle im Verzeichnis"""
    if not os.path.exists(data_dir):
        return 0
    return len(glob.glob(os.path.join(data_dir, "*.npz")))

def run_inference(model_type, checkpoint_path, data_dir, output_dir, config_path=None, prompt_type="box", sam2_base_checkpoint=None):
    """
    Führe Inferenz mit dem angegebenen Modell und Checkpoint durch
    
    Args:
        model_type: "sam2" oder "medsam2"
        checkpoint_path: Pfad zum Checkpoint
        data_dir: Pfad zu den Testdaten
        output_dir: Ausgabeverzeichnis für Segmentierungen
        config_path: Pfad zur Konfigurationsdatei
        prompt_type: "box" oder "point"
    
    Returns:
        success: True wenn die Inferenz erfolgreich war
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Lade GUI-Settings für Prompt-Parameter
    settings = load_gui_settings()
    box_shift = settings.get("box_shift", 5)
    num_pos_points = settings.get("num_pos_points", 3)
    num_neg_points = settings.get("num_neg_points", 4)
    min_dist_from_mask_edge = settings.get("min_dist_from_mask_edge", 3)
    #box_shift = 20
    #num_pos_points = 12
    #num_neg_points = 12
    #min_dist_from_mask_edge = settings.get("min_dist_from_mask_edge", 3)
    
    # Standardwerte für Config und Modelldateien
    if not config_path:
        config_path = "C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/sam2_configs/sam2_hiera_t.yaml"
    if not sam2_base_checkpoint:
        sam2_base_checkpoint = "C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/checkpoints/sam2_hiera_tiny.pt"
    
    # Erstelle Befehl basierend auf dem Modelltyp
    if model_type == "sam2":
        script_path = os.path.join("eval_inference", "infer_sam2_2d.py")
        command = [
            f"python {script_path}",
            f"-data_root {data_dir}",
            f"-pred_save_dir {output_dir}",
            f"-sam2_checkpoint {checkpoint_path}",
            f"-model_cfg {config_path}",
            f"-bbox_shift {box_shift}",
            f"-num_workers 10",
            f"-prompt_type {prompt_type}",
            f"-debug_mode",
            f"--label 1"
        ]
    else:
        script_path = os.path.join("eval_inference", "infer_medsam2_2d.py")
        command = [
            f"python {script_path}",
            f"-data_root {data_dir}",
            f"-pred_save_dir {output_dir}",
            f"-sam2_checkpoint {sam2_base_checkpoint}",
            f"-medsam2_checkpoint {checkpoint_path}",
            f"-model_cfg {config_path}",
            f"-bbox_shift {box_shift}",
            f"-num_workers 10",
            f"-prompt_type {prompt_type}",
            f"-debug_mode",
            f"--label 1"
        ]
            
    # Füge spezifische Parameter für Prompt-Typ hinzu
    if prompt_type == "point":
        # Ersetze bbox_shift durch Point-Parameter
        command = [cmd for cmd in command if "-bbox_shift" not in cmd]
        command.append(f"-num_pos_points {num_pos_points}")
        command.append(f"-num_neg_points {num_neg_points}")
        command.append(f"-min_dist_from_edge {min_dist_from_mask_edge}")
    
    # Führe Befehl aus
    cmd_str = " ".join(command)
    print(f"Ausführung von: {cmd_str}")
    try:
        result = subprocess.run(cmd_str, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Inferenz abgeschlossen: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei der Inferenz: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def compute_metrics(checkpoint_name, dataset_name, prompt_type, metrics_dir, pred_dir):
    """
    Berechne Metriken für die Vorhersagen
    
    Args:
        checkpoint_name: Name des Checkpoints
        dataset_name: Name des Datensatzes
        prompt_type: "box" oder "point"
        metrics_dir: Verzeichnis für Metrik-Ausgabe
        pred_dir: Verzeichnis mit Vorhersagen
    
    Returns:
        metrics_csv: Pfad zur CSV-Datei mit Metriken
    """
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_csv = os.path.join(metrics_dir, "metrics.csv")
    
    # Die tatsächliche Datenverzeichnisstruktur basierend auf dem eval_metrics-Skript anpassen
    # Ground Truth ist das ursprüngliche Testdatenverzeichnis
    dataset_path = ""
    for ckpt_key, info in CHECKPOINTS.items():
        if dataset_name.startswith(ckpt_key):
            dataset_path = info["test_data"]
            break
    
    # Wenn kein Datensatz gefunden wurde, versuche das Standard-Schema
    if not dataset_path:
        dataset_path = f"data/npz_custom_{dataset_name}/test"
    
    # Prüfe ob der Pfad existiert
    if not os.path.exists(dataset_path):
        print(f"Warnung: Ground-Truth-Daten nicht gefunden in {dataset_path}")
        return None
    
    # Benutze die richtigen Parameter für compute_metrics.py
    command = [
        "python", os.path.join("eval_metrics", "compute_metrics.py"),
        "-s", pred_dir,         # Segmentierungen
        "-g", dataset_path,     # Original Ground Truth aus dem Datensatz
        "-csv_dir", metrics_dir  # Ausgabeverzeichnis für Metriken
    ]
    
    cmd_str = " ".join(command)
    print(f"Berechne Metriken: {cmd_str}")
    try:
        result = subprocess.run(cmd_str, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Metrikberechnung abgeschlossen: {result.stdout}")
        return metrics_csv
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei der Metrikberechnung: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None

def read_metrics(csv_path):
    """
    Lese Metriken aus einer CSV-Datei und berechne Durchschnittswerte
    
    Args:
        csv_path: Pfad zur CSV-Datei mit Metriken
    
    Returns:
        avg_metrics: Dictionary mit durchschnittlichen Metrikwerten
    """
    if not os.path.exists(csv_path):
        print(f"Warnung: Metrikdatei nicht gefunden: {csv_path}")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        # Berechne Durchschnitt für jede Metrik
        avg_metrics = {}
        for metric in METRICS:
            if metric in df.columns:
                avg_metrics[metric] = df[metric].mean()
        return avg_metrics
    except Exception as e:
        print(f"Fehler beim Lesen der Metriken aus {csv_path}: {e}")
        return {}

def create_histogram_plots(df_all, df_cases, output_dir):
    """
    Creates boxplots and heatmaps for DICE scores and saves them as PNG files

    Args:
        df_all: DataFrame with all combinations and average values
        df_cases: DataFrame with all individual cases
        output_dir: Directory where plots should be saved
    """
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Liste deiner Checkpoints in gewünschter Reihenfolge
    checkpoints = [
        'SAM2.0-T', 'SAM2.1-T', 'SAM2.1-S', 'SAM2.1-B',
        'Head', 'Lung', 'Others', 'Mix', 'Mix (noHead)',
        'MR-Glio', 'MR-Head', 'MR+CT', 'MR+CT-t2.1','MR+CT-s2.1'
    ]

    # Hole 12 Farben aus der 'Paired' Palette
    paired_colors = sns.color_palette("Paired", n_colors=len(checkpoints))

    # Feste Farbzuweisung als Dictionary
    fixed_palette = dict(zip(checkpoints, paired_colors))
    fixed_palette = sns.color_palette("tab20", n_colors=15)
    
    # Create directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Mapping for dataset and checkpoint names
    name_mapping = {
        'sam2': 'SAM2.0-T',
        'SAM2.1-Tiny': 'SAM2.1-T',
        'SAM2.1-Small': 'SAM2.1-S',
        'SAM2.1-BasePlus': 'SAM2.1-B',
        'Kopf': 'Head',
        'Lunge': 'Lung',
        'Rest': 'Others',
        'Mix': 'Mix',
        'Mix2': 'Mix (noHead)',
        'Mr-glio': 'MR-Glio',
        'Mr-mix': 'MR-Head',
        'Mr+ct': 'MR+CT'
    }
    # Anzeige-Mapping für Checkpoints
    checkpoint_display_mapping = {
        "SAM2.1-FT-tiny": "MR+CT-t2.1", 
        "SAM2.1-FT-small": "MR+CT-s2.1",
        "SAM2-FT-small": "MR+CT-s2"
    }
    dataset_order = ['Head', 'Lung', 'Others', 'Mix', 'Mix (noHead)', 'MR-Glio', 'MR-Head', 'MR+CT']
    #checkpoint_order = ['SAM2.0-T', 'SAM2.1-T', 'SAM2.1-S', 'SAM2.1-B', 'SAM2.1-FT']
    checkpoint_order = [
    'SAM2.0-T', 'SAM2.1-T', 'SAM2.1-S', 'SAM2.1-B',
    'Head', 'Lung', 'Others', 'Mix', 'Mix (noHead)',
    'MR-Glio', 'MR-Head', 'MR+CT', 'MR+CT-t2.1', 'MR+CT-s2.1', 'MR+CT-s2'  # statt SAM2.1-FT
]

    def apply_mapping(df):
        df = df.copy()

        # Erst alle Namen mappen
        df['Dataset'] = df['Dataset'].map(lambda x: name_mapping.get(x, x))
        df['Checkpoint'] = df['Checkpoint'].map(lambda x: name_mapping.get(x, x))

        # Dann den speziellen Display-Mapping anwenden (z. B. SAM2.1-FT → MR+CT-2.1)
        df['Checkpoint'] = df['Checkpoint'].replace(checkpoint_display_mapping)

        # Jetzt sind alle Namen korrekt → jetzt Categorical setzen
        df['Dataset'] = pd.Categorical(df['Dataset'], categories=dataset_order, ordered=True)
        combined_checkpoints = list(dict.fromkeys(checkpoint_order))  # sicherstellen: keine Duplikate
        df['Checkpoint'] = pd.Categorical(df['Checkpoint'], categories=combined_checkpoints, ordered=True)

        return df

    #df_all = apply_mapping(df_all)
    df_cases = apply_mapping(df_cases)
    # Debug output
    #print("[DEBUG] Unique Datasets (df_cases):", df_cases['Dataset'].dropna().unique())
    print("[DEBUG] Unique Checkpoints (df_cases):", df_cases['Checkpoint'].dropna().unique())

    for prompt_type in ["box", "point"]:
        #df_all_filtered = df_all[df_all["Prompt Type"] == prompt_type] if not df_all.empty else pd.DataFrame()
        df_cases_filtered = df_cases[df_cases["Prompt Type"] == prompt_type] if not df_cases.empty else pd.DataFrame()
        
        if not df_cases_filtered.empty and 'DICE' in df_cases_filtered.columns:
            plt.figure(figsize=(14, 8))
            ax = sns.boxplot(x="Dataset", y="DICE", hue="Checkpoint", data=df_cases_filtered,
                             palette=fixed_palette, showfliers=True, width=0.7)
            plt.title(f'Distribution of DICE Scores by Dataset and Checkpoint ({prompt_type.capitalize()}-Prompts)', fontsize=16)
            #plt.xlabel('Dataset', fontsize=14)
            plt.ylabel('DICE Score', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            #plt.legend(title="Checkpoint", fontsize=12, title_fontsize=14)
            plt.ylim(0, 1)
             # Legend unten platzieren
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles, labels,
                title="Checkpoint",
                title_fontsize=12,
                fontsize=12,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.1),  # verschiebt Legende unter den Plot
                ncol=5,  # Anzahl der Spalten in der Legende
                frameon=False
            )
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, f"boxplot_cases_{prompt_type}.png")
            plt.savefig(plot_path, format='png', dpi=600)
            plt.close()
            print(f"Plot saved: {plot_path}")

    for prompt_type in ["box", "point"]:
        df_filtered_cases = df_cases[df_cases["Prompt Type"] == prompt_type]
        if not df_filtered_cases.empty:
            pivot_median = df_filtered_cases.pivot_table(
                values='DICE',
                index='Dataset',
                columns='Checkpoint',
                aggfunc='median'
            ).reindex(index=dataset_order, columns=checkpoint_order)

            plt.figure(figsize=(12, 10))
            ax = sns.heatmap(pivot_median, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5,
                             cbar_kws={'label': 'DICE Score (Median)'}, vmin=0, vmax=1)
            plt.xlabel('Checkpoint', fontsize=14)
            #plt.ylabel('Dataset', fontsize=14)
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, f"heatmap_{prompt_type}.png")
            plt.savefig(plot_path, format='png', dpi=600)
            plt.close()
            print(f"Plot saved: {plot_path}")

    for prompt_type in ["box", "point"]:
        df_sam2_prompt = df_cases[
            (df_cases['Model Type'] == 'SAM2') & 
            (df_cases['Prompt Type'] == prompt_type)
        ]
        if not df_sam2_prompt.empty:
            plt.figure(figsize=(14, 8))
            sns.boxplot(
                x="Dataset", 
                y="DICE", 
                hue="Checkpoint", 
                data=df_sam2_prompt,
                palette=fixed_palette, 
                showfliers=True,
                hue_order=["SAM2.0-T", "SAM2.1-T", "SAM2.1-S", "SAM2.1-B"]
            )
            plt.title(f"DICE-Verteilung nur SAM2-Varianten ({prompt_type.capitalize()}-Prompts)")
            plt.ylim(0, 1)
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, f"boxplot_sam2_variants_{prompt_type}.png")
            plt.savefig(plot_path, dpi=600)
            plt.close()
            print(f"Plot saved: {plot_path}")


def create_combined_excel(sam2_results, medsam2_results, output_path):
    """
    Erstelle einen kombinierten Excel-Bericht mit SAM2 und MedSAM2 Ergebnissen
    
    Args:
        sam2_results: Dictionary mit SAM2 Evaluationsergebnissen (wird nicht verwendet)
        medsam2_results: Dictionary mit MedSAM2 Evaluationsergebnissen (wird nicht verwendet)
        output_path: Pfad zur Ausgabedatei
    """
    base_output_dir = os.path.join("results", "checkpoint_evaluation")
    
    print("Erstelle kombinierten Excel-Bericht direkt aus CSV-Dateien...")
    
    # Finde alle metrics.csv und dsc_summary.csv Dateien im Ausgabeverzeichnis
    all_metric_files = []
    all_summary_files = []
    
    # Durchsuche das Basisverzeichnis nach metrics.csv und dsc_summary.csv Dateien
    for root, dirs, files in os.walk(base_output_dir):
        for file in files:
            if file == "metrics.csv":
                metrics_csv = os.path.join(root, file)
                rel_path = os.path.relpath(root, base_output_dir)
                all_metric_files.append((rel_path, metrics_csv))
            elif file == "dsc_summary.csv":
                summary_csv = os.path.join(root, file)
                rel_path = os.path.relpath(root, base_output_dir)
                all_summary_files.append((rel_path, summary_csv))
    
    if not all_metric_files:
        print("Keine Metrik-CSV-Dateien gefunden.")
        return
    
    print(f"Gefunden: {len(all_metric_files)} metrics.csv Dateien und {len(all_summary_files)} dsc_summary.csv Dateien")
    
    # Gesamtdaten für alle Kombinationen
    all_combinations_data = []
    
    # Liste für alle Einzelfalldaten
    all_cases_data = []
    
    # Verarbeite zuerst die summary-Dateien für Durchschnittswerte
    for model_id, summary_csv in all_summary_files:
        # Parsen der Modell-ID (Format wie vorher)
        parts = model_id.split(os.sep)
        if len(parts) != 2:
            print(f"Unerwartetes Verzeichnisformat: {model_id}, überspringe...")
            continue
        
        model_info = parts[0]
        model_parts = model_info.split('_')
        
        if model_parts[0].lower().startswith("sam2"):
            model_type = "SAM2"
            if model_parts[0].lower() == "sam2":
                checkpoint = "sam2"  # → klein geschrieben!
            else:
                checkpoint = model_parts[0]
            dataset = model_parts[1]
            prompt_type = model_parts[2] if len(model_parts) > 2 else "box"
        elif model_parts[0] == "medsam2":
            model_type = "MedSAM2"
            # Verwende nur den Organtyp als Checkpoint-Namen anstelle des kompletten Namens
            checkpoint = model_parts[1].capitalize()  # z.B. "kopf" zu "Kopf"
            dataset = model_parts[3] if len(model_parts) > 3 else ""
            prompt_type = model_parts[4] if len(model_parts) > 4 else "box"
        else:
            print(f"Unbekanntes Modellformat: {model_info}, überspringe...")
            continue
        
        # Matching bestimmen
        is_matching = (model_parts[0] == "medsam2" and model_parts[1] == dataset)
        
        # Ermittle Fallanzahl aus der metriken.csv im selben Verzeichnis
        num_cases = 0
        metrics_csv = os.path.join(os.path.dirname(summary_csv), "metrics.csv")
        if os.path.exists(metrics_csv):
            try:
                df_metrics = pd.read_csv(metrics_csv)
                num_cases = len(df_metrics)
            except Exception as e:
                print(f"Fehler beim Zählen der Fälle in {metrics_csv}: {e}")
        
        # Lese die Summary-CSV-Datei (enthält Durchschnittswerte)
        try:
            # Erwartetes Format: ,0
            #                    1_DSC,0.79075
            df_summary = pd.read_csv(summary_csv, header=None)
            avg_dsc = None
            
            for _, row in df_summary.iterrows():
                if str(row.iloc[0]).strip() == "1_DSC":
                    avg_dsc = float(row.iloc[1])
                    break
            
            if avg_dsc is not None:
                avg_row = {
                    "Model Type": model_type,
                    "Checkpoint": checkpoint,
                    "Dataset": dataset.capitalize(),  # Auch Dataset-Namen kapitalisieren
                    "Prompt Type": prompt_type,
                    "Matching": is_matching,
                    "Num Cases": num_cases,
                    "DICE (Avg)": avg_dsc
                }
                all_combinations_data.append(avg_row)
            else:
                print(f"DICE (Avg) nicht gefunden in {summary_csv}")
                
        except Exception as e:
            print(f"Fehler beim Lesen von {summary_csv}: {e}")
    
    # Verarbeite die metrics.csv Dateien für Einzelfälle
    for model_id, metrics_csv in all_metric_files:
        parts = model_id.split(os.sep)
        if len(parts) != 2:
            continue
        
        model_info = parts[0]
        model_parts = model_info.split('_')
        
        if model_parts[0].lower().startswith("sam2"):
            model_type = "SAM2"
            if model_parts[0].lower() == "sam2":
                checkpoint = "sam2"  # → klein geschrieben!
            else:
                checkpoint = model_parts[0]
            dataset = model_parts[1]
            prompt_type = model_parts[2] if len(model_parts) > 2 else "box"
        elif model_parts[0] == "medsam2":
            model_type = "MedSAM2"
            # Verwende nur den Organtyp als Checkpoint-Namen anstelle des kompletten Namens
            checkpoint = model_parts[1].capitalize()  # z.B. "kopf" zu "Kopf"
            dataset = model_parts[3] if len(model_parts) > 3 else ""
            prompt_type = model_parts[4] if len(model_parts) > 4 else "box"
        else:
            continue
        
        is_matching = (model_parts[0] == "medsam2" and model_parts[1] == dataset)
        
        # Lese die CSV-Datei mit Einzelfallmetriken
        try:
            # Erwartetes Format: case,1_DSC,1_NSD
            #                    Mamma_case_27_Mamma.npz,0.7625,0.6223
            df_metrics = pd.read_csv(metrics_csv)
            
            # Verarbeite jeden Fall
            for _, row in df_metrics.iterrows():
                case_id = row.iloc[0]  # erste Spalte ist case_id
                
                # Versuche, die 1_DSC-Spalte zu finden
                dsc_column = None
                for col in df_metrics.columns:
                    if '1_DSC' in col or 'dice' in col.lower():
                        dsc_column = col
                        break
                
                if dsc_column is None:
                    print(f"Keine DICE-Spalte gefunden in {metrics_csv}")
                    continue
                    
                dsc_value = row[dsc_column]
                
                # Fall zur Liste aller Fälle hinzufügen
                case_row = {
                    "Case ID": case_id,
                    "Model Type": model_type,
                    "Checkpoint": checkpoint,
                    "Dataset": dataset.capitalize(),
                    "Prompt Type": prompt_type,
                    "Matching": is_matching,
                    "DICE": dsc_value
                }
                all_cases_data.append(case_row)
                
        except Exception as e:
            print(f"Fehler beim Lesen von {metrics_csv}: {e}")
    
    # Erstelle DataFrame für alle Kombinationen
    df_all = pd.DataFrame(all_combinations_data)
    
    # Sortiere nach Modelltyp, Dataset, Checkpoint
    if not df_all.empty:
        df_all = df_all.sort_values(by=['Model Type', 'Dataset', 'Checkpoint', 'Prompt Type'])
    
    # Erstelle DataFrame für alle Einzelfälle
    df_cases = pd.DataFrame(all_cases_data)
    if not df_cases.empty:
        df_cases = df_cases.sort_values(by=['Dataset', 'Case ID', 'Model Type', 'Checkpoint', 'Prompt Type'])
    
    # Mapping für sprechendere Namen
    checkpoint_mapping = {
        'sam2': 'SAM2.0-T',
        'SAM2.1-Tiny': 'SAM2.1-T',
        'SAM2.1-Small': 'SAM2.1-S',
        'SAM2.1-BasePlus': 'SAM2.1-B'
    }

    dataset_mapping = {
        'Kopf': 'Head',
        'Lunge': 'Lung',
        'Rest': 'Others',
        'Mix': 'Mix',
        'Mix2': 'Mix (noHead)',
        'MR-Glio': 'MR-Glio',
        'MR-Mix': 'MR-Head',
        'MR+CT': 'MR+CT'
    }

    # Wende Mapping auf die DataFrames an
    df_all['Checkpoint'] = df_all['Checkpoint'].map(checkpoint_mapping).fillna(df_all['Checkpoint'])
    df_all['Dataset'] = df_all['Dataset'].map(dataset_mapping).fillna(df_all['Dataset'])

    df_cases['Checkpoint'] = df_cases['Checkpoint'].map(checkpoint_mapping).fillna(df_cases['Checkpoint'])
    df_cases['Dataset'] = df_cases['Dataset'].map(dataset_mapping).fillna(df_cases['Dataset'])


    # Erstelle Excel-Datei
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Alle Kombinationen
        df_all.to_excel(writer, sheet_name="All Combinations", index=False)
        
        # Durchschnittswerte pro Dataset/Modell als Pivot-Tabelle
        if not df_all.empty:
            try:
                pivot = df_all.pivot_table(
                    values='DICE (Avg)', 
                    index=['Dataset', 'Prompt Type'],
                    columns=['Model Type', 'Checkpoint'],
                    aggfunc='mean'
                )
                pivot.to_excel(writer, sheet_name="Avg by Dataset")

                pivot_median = df_cases.pivot_table(
                    values='DICE',
                    index=['Dataset', 'Prompt Type'],
                    columns=['Model Type', 'Checkpoint'],
                    aggfunc='median'
                )
                # Optional: nach Model Type und Checkpoint-Namen umsortieren wie im Avg-Sheet
                pivot_median.columns.name = None
                pivot_median.to_excel(writer, sheet_name="Median by Dataset")
                # DICE-Median wie in boxplots – zuerst pro Case mitteln, dann pro Dataset+Checkpoint median aggregieren
                for prompt in df_cases["Prompt Type"].unique():
                    df_prompt = df_cases[df_cases["Prompt Type"] == prompt]
                    pivot = df_prompt.pivot_table(
                        values="DICE",
                        index="Dataset",
                        columns="Checkpoint",
                        aggfunc="median"
                    )
                    pivot.to_excel(writer, sheet_name=f"DICE Median ({prompt})")


            except Exception as e:
                print(f"Fehler beim Erstellen der Pivot-Tabelle: {e}")
        
        # Alle Einzelfälle in einem Sheet
        if not df_cases.empty:
            df_cases.to_excel(writer, sheet_name="All Cases", index=False)
            
        # Durchschnitt und Summe pro Fall über alle Kombinationen
        if not df_cases.empty:
            try:
                # Gruppiere nach Case ID und berechne Durchschnitt und Summe der DICE-Werte
                case_summary = df_cases.groupby('Case ID').agg(
                    Avg_DICE=('DICE', 'mean'),
                    Sum_DICE=('DICE', 'sum'),
                    Count=('DICE', 'count')
                ).reset_index()
                
                # Sortiere nach Durchschnitt (absteigend)
                case_summary = case_summary.sort_values(by='Avg_DICE', ascending=False)
                
                # Füge Gesamtdurchschnitt und -summe am Ende hinzu
                total_row = pd.DataFrame({
                    'Case ID': ['TOTAL'],
                    'Avg_DICE': [df_cases['DICE'].mean()],
                    'Sum_DICE': [df_cases['DICE'].sum()],
                    'Count': [len(df_cases)]
                })
                case_summary = pd.concat([case_summary, total_row], ignore_index=True)
                
                # Speichere im Excel
                case_summary.to_excel(writer, sheet_name="Case Summary", index=False)
            except Exception as e:
                print(f"Fehler beim Erstellen der Case Summary: {e}")
    
    # Erstelle Histogramm-Plots
    plots_dir = os.path.join(os.path.dirname(output_path), "plots")
    create_histogram_plots(df_all, df_cases, os.path.dirname(output_path))
    
    print(f"Kombinierter Excel-Bericht gespeichert unter: {output_path}")

def main():
    vanilla_checkpoints = [
        {
            "name": "SAM2.1-Tiny",
            "path": "checkpoints/sam2.1_hiera_tiny.pt",
            "config": "C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/sam2_configs/sam2.1_hiera_t.yaml"
        },
        {
            "name": "SAM2.1-Small",
            "path": "checkpoints/sam2.1_hiera_small.pt",
            "config": "C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/sam2_configs/sam2.1_hiera_s.yaml"
        },
        {
            "name": "SAM2.1-BasePlus",
            "path": "checkpoints/sam2.1_hiera_base_plus.pt",
            "config": "C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/sam2_configs/sam2.1_hiera_b+.yaml"
        },
        {
            "name": "SAM2.1-FT-tiny",  # Dein Fine-Tuned Modell
            "path": "checkpoints/sam2.1_hiera_tiny.pt",  # Basis-SAM2.1-Modell
            "config": "C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/sam2_configs/sam2.1_hiera_t.yaml",
            "ft_checkpoint": "work_dir/MedSAM2-Tiny-STARDUST_MixMix.1t/best_checkpoint.pth"
        },
        {
            "name": "SAM2.1-FT-small",  # Dein Fine-Tuned Modell
            "path": "checkpoints/sam2.1_hiera_small.pt",  # Basis-SAM2.1-Modell
            "config": "C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/sam2_configs/sam2.1_hiera_s.yaml",
            "ft_checkpoint": "work_dir/MedSAM2-Tiny-STARDUST_MixMix.1s/best_checkpoint.pth"
        }
        ,
        {
            "name": "SAM2-FT-small",  # Dein Fine-Tuned Modell
            "path": "checkpoints/sam2_hiera_small.pt",  # Basis-SAM2.1-Modell
            "config": "C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/sam2_configs/sam2_hiera_s.yaml",
            "ft_checkpoint": "work_dir/MedSAM2-Tiny-STARDUST_MixMix.s/best_checkpoint.pth"
        }
    ]
    
    # Fester Ausgabeordner ohne Zeitstempel für einfacheres Wiederverwenden
    base_output_dir = os.path.join("results", "checkpoint_evaluation")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Lade GUI-Settings für SAM2-Checkpoint
    settings = load_gui_settings()
    sam2_checkpoint = settings.get("sam2_checkpoint", "C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/checkpoints/sam2_hiera_tiny.pt")
    config_path = settings.get("config", "C:/Users/MG/Desktop/STARDUST2/MedSAM-MedSAM2_MG2/sam2_configs/sam2_hiera_t.yaml")
    
    # Prompt-Typ aus Settings oder als Benutzerabfrage
    print("Wähle den Prompt-Typ für die Evaluation:")
    print("1. Box-Prompts")
    print("2. Point-Prompts")
    print("3. Beide (Box und Point)")
    
    prompt_choice = input("Wähle (1/2/3, Standard ist 1): ").strip()
    
    if prompt_choice == "2":
        prompt_types = ["point"]
    elif prompt_choice == "3":
        prompt_types = ["box", "point"]
    else:
        prompt_types = ["box"]
    
    # Lade Prompt-Parameter aus den Einstellungen oder verwende Standardwerte
    box_shift = settings.get("box_shift", 5)
    num_pos_points = settings.get("num_pos_points", 3)
    num_neg_points = settings.get("num_neg_points", 4)
    min_dist_from_mask_edge = settings.get("min_dist_from_mask_edge", 3)
    
    if not sam2_checkpoint:
        print("Warnung: SAM2-Checkpoint nicht in den Einstellungen gefunden.")
        sam2_checkpoint = input("Bitte geben Sie den Pfad zum SAM2-Checkpoint ein: ")
        
    # Initialisiere Ergebnisdictionaries
    sam2_results = {}
    medsam2_results = {}
    
    # ----- SAM2 TESTS (nur einmal pro Datensatz) -----
    print("\n--- Teste SAM2 (Vanilla) auf allen Datensätzen ---")
    
    # Definiere den Standard-SAM2-Checkpoint und die Config
    ##sam2_checkpoint = "checkpoints/sam2_base.pt"
    #config_path = "sam2_configs/sam2_base.yaml"

    for vanilla_ckpt in vanilla_checkpoints:
        print(f"\n==> Teste {vanilla_ckpt['name']} auf allen Datensätzen")        
        is_ft_model = "ft_checkpoint" in vanilla_ckpt
        print(f"[DEBUG] is_ft_model = {is_ft_model}, verwende Modelltyp: {'medsam2' if is_ft_model else 'sam2'}")

        for dataset_key, dataset_info in CHECKPOINTS.items():
            dataset_path = dataset_info["test_data"]
            if not os.path.exists(dataset_path):
                continue

            print(f"\nTestdatensatz: {dataset_path}")
            num_cases = get_test_cases_count(dataset_path)
            if num_cases == 0:
                print(f"Keine Testfälle gefunden in: {dataset_path}")
                continue

            if dataset_key not in sam2_results:
                sam2_results[dataset_key] = {}

            for prompt_type in prompt_types:
                prompt_suffix = f"_{prompt_type}"
                result_key = f"{vanilla_ckpt['name']}_{dataset_key}{prompt_suffix}"
                pred_dir = os.path.join(base_output_dir, result_key, "predictions")
                metrics_dir = os.path.join(base_output_dir, result_key, "metrics")
                metrics_csv = os.path.join(metrics_dir, "metrics.csv")

                if os.path.exists(metrics_csv):
                    print(f"Inferenz bereits durchgeführt für: {result_key}")
                    sam2_results[dataset_key][result_key] = read_metrics(metrics_csv)
                    continue

                print(f"Starte Inferenz für: {result_key}")
                if is_ft_model:
                    success = run_inference(
                        "medsam2",
                        vanilla_ckpt["ft_checkpoint"],
                        dataset_path,
                        pred_dir,
                        vanilla_ckpt["config"],
                        prompt_type=prompt_type,
                        sam2_base_checkpoint=vanilla_ckpt["path"]
                    )
                else:
                    success = run_inference(
                        "sam2",
                        vanilla_ckpt["path"],
                        dataset_path,
                        pred_dir,
                        vanilla_ckpt["config"],
                        prompt_type=prompt_type,
                        sam2_base_checkpoint=None
                    )

                if success:
                    metrics_csv = compute_metrics(result_key, dataset_key, prompt_type, metrics_dir, pred_dir)
                    if metrics_csv:
                        sam2_results[dataset_key][result_key] = read_metrics(metrics_csv)

    
    
    # ----- MEDSAM2 TESTS (Matrix von Checkpoints x Datensätzen) -----
    # Verarbeite jeden Checkpoint (MedSAM2 auf unterschiedlichen Organen trainiert)
    for ckpt_key, ckpt_info in CHECKPOINTS.items():
        ckpt_name = ckpt_info["name"]
        print(f"\n--- Verarbeite MedSAM2 Checkpoint: {ckpt_name} ---")
        
        # Bestimme Checkpoint-Pfad
        checkpoint_path = ckpt_info["path"]
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint nicht gefunden: {checkpoint_path}")
            checkpoint_path = input(f"Bitte geben Sie den Pfad zum {ckpt_name} Checkpoint ein: ")
        
        # Initialisiere Ergebnisdictionary für diesen Checkpoint
        if ckpt_key not in medsam2_results:
            medsam2_results[ckpt_key] = {}
        
        # Verarbeite jeden Testdatensatz mit diesem Checkpoint
        for dataset_key, dataset_info in CHECKPOINTS.items():
            dataset_path = dataset_info["test_data"]
            print(f"\nTestdatensatz: {dataset_path}")
            
            # Zähle Testfälle
            num_cases = get_test_cases_count(dataset_path)
            if num_cases == 0:
                print(f"Keine Testfälle gefunden in: {dataset_path}")
                continue
            
            print(f"Gefundene Testfälle: {num_cases}")
            
            # Verarbeite jeden Prompt-Typ
            for prompt_type in prompt_types:
                prompt_suffix = f"_{prompt_type}"
                
                # MedSAM2 Inferenz und Metriken
                # Klarer Name: medsam2_trainedOn_on_testedOn_prompttype
                medsam2_output_dir = os.path.join(base_output_dir, f"medsam2_{ckpt_key}_on_{dataset_key}{prompt_suffix}")
                medsam2_pred_dir = os.path.join(medsam2_output_dir, "predictions")
                medsam2_metrics_dir = os.path.join(medsam2_output_dir, "metrics")
                metrics_csv = os.path.join(medsam2_metrics_dir, "metrics.csv")
                
                # Überprüfe, ob dieser Task bereits abgeschlossen ist
                if os.path.exists(metrics_csv):
                    print(f"MedSAM2 ({ckpt_key}) Inferenz für {dataset_key} mit {prompt_type} bereits vorhanden, überspringe...")
                    result_key = f"{dataset_key}{prompt_suffix}"
                    medsam2_results[ckpt_key][result_key] = read_metrics(metrics_csv)
                    continue
                
                print(f"\nStarting MedSAM2 ({ckpt_key}) Inferenz auf {dataset_key} mit {prompt_type}-Prompts...")
                if run_inference("medsam2", checkpoint_path, dataset_path, medsam2_pred_dir, config_path, prompt_type=prompt_type):
                    metrics_csv = compute_metrics(ckpt_name, dataset_key, prompt_type, medsam2_metrics_dir, medsam2_pred_dir)
                    if metrics_csv:
                        result_key = f"{dataset_key}{prompt_suffix}"
                        medsam2_results[ckpt_key][result_key] = read_metrics(metrics_csv)
    
    # Erstelle Berichte
    combined_report_path = os.path.join(base_output_dir, "combined_results.xlsx")
    
    create_combined_excel(sam2_results, medsam2_results, combined_report_path)
    
    print(f"\nEvaluation abgeschlossen. Ergebnisse wurden in {base_output_dir} gespeichert.")
    print(f"Kombinierter Bericht: {combined_report_path}")

if __name__ == "__main__":
    main()
