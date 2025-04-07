#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPZ File Creator from DICOM Data using STARDUST Excel Sheet
"""

import os
import sys
import pandas as pd
from datetime import datetime
import argparse
import random
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dicom_converter import convert_dicom_to_npz

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(base_dir, 'data', 'DICOM-All', 'STARDUST_MR-Worklist_MG.xlsx')
    export_dir = os.path.join(base_dir, 'data', 'DICOM-All', 'MR_Export')
    npz_output_dir = os.path.join(base_dir, 'data', 'npz_files_MR')
    os.makedirs(npz_output_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description='Create NPZ files from DICOM data')
    parser.add_argument('--train_split', type=float, default=1)
    args = parser.parse_args()

    df = pd.read_excel(excel_path, sheet_name=0)

    # Nur Zeilen mit Patient ID und Label weiterverarbeiten
    df = df[df["Patient ID"].notna() & df["Label"].notna()]

    # Label-Kürzel ggf. umwandeln
    label_map = {
        'men': 'Meningeom',
        'glio': 'Glioblastom',
        'astro': 'Astrozytom'
    }

    # Nur Labels umwandeln, die im Mapping enthalten sind
    df["Label"] = df["Label"].apply(lambda x: label_map.get(x, x))

    # Jetzt kannst du wie gewohnt weitermachen
    cases = []

    for _, row in df.iterrows():
        patient_id = str(row["Patient ID"]).strip()
        raw_label = str(row["Label"]).strip()
        label = label_map.get(raw_label.lower(), raw_label)  # Fallback falls Label nicht gemappt wird
        label2 = str(row["Label2"]).strip() if "Label2" in row and pd.notna(row["Label2"]) else "unknown"
        cases.append({'Patient_ID': patient_id, 'Label': label, 'Label2': label2})

    random.shuffle(cases)
    train_size = int(len(cases) * args.train_split)
    train_cases, test_cases = cases[:train_size], cases[train_size:]

    train_output_dir = os.path.join(npz_output_dir, 'train')
    test_output_dir = os.path.join(npz_output_dir, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    pseudo_mapping = []
    modality = 'MR'

    for split, case_list, output_dir in [('train', train_cases, train_output_dir), ('test', test_cases, test_output_dir)]:
        for idx, case in enumerate(case_list, 1):
            patient_id = case['Patient_ID']
            label = case['Label']
            label2 = case['Label2']

            patient_dir = os.path.join(export_dir, label, patient_id)
            if not os.path.isdir(patient_dir):
                status = 'Skipped - Patient folder not found'
                logger.warning(f"{status}: {patient_dir}")
                pseudo_mapping.append({
                    'Case_Number': idx,
                    'Patient_ID': patient_id,
                    'Label': label,
                    'Label2': label2,
                    'Status': status,
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue

            subdirs = [os.path.join(patient_dir, d) for d in os.listdir(patient_dir)
                       if os.path.isdir(os.path.join(patient_dir, d))]
            if not subdirs:
                status = 'Skipped - No series folder found'
                logger.warning(f"{status} in {patient_dir}")
                pseudo_mapping.append({
                    'Case_Number': idx,
                    'Patient_ID': patient_id,
                    'Label': label,
                    'Label2': label2,
                    'Status': status,
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue

            series_dir = subdirs[0]
            npz_filename = f"{modality}_case_{idx}_{label}_{patient_id}_{label2}"

            try:
                npz_path = convert_dicom_to_npz(
                    dicom_dir=series_dir,
                    output_dir=output_dir,
                    case_name=npz_filename,
                    save_nii=False
                )
                status = 'Success'
            except Exception as e:
                npz_path = 'N/A'
                status = f'Error - {str(e)[:100]}'
                logger.error(f"Fehler bei {patient_id}: {e}")
            
            if isinstance(npz_path, str):
                npz_filename_result = os.path.basename(npz_path)
            else:
                npz_filename_result = 'N/A'

            pseudo_mapping.append({
                'Case_Number': idx,
                'Patient_ID': patient_id,
                'Label': label,
                'Label2': label2,
                'Status': status,
                'NPZ_Filename': npz_filename_result,
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    pseudo_csv_path = os.path.join(npz_output_dir, f"pseudo_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(pseudo_mapping).to_csv(pseudo_csv_path, index=False)
    logger.info(f"Pseudo-Mapping-Datei erstellt: {pseudo_csv_path}")

    success_count = sum(1 for item in pseudo_mapping if item['Status'] == 'Success')
    logger.info(f"Total: {len(pseudo_mapping)}, Success: {success_count}, Failed: {len(pseudo_mapping) - success_count}")

if __name__ == "__main__":
    main()
