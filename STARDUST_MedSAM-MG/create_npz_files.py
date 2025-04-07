#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPZ File Creator from DICOM Data
This script creates NPZ files from DICOM data based on clinical cases in a CSV file.
It uses the existing dicom_converter.py script and creates a pseudonymization mapping.
"""

import os
import sys
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
import random

# Import the dicom_converter module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dicom_converter import convert_dicom_to_npz

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'DICOM-All', 'Ausgabe2_refine_ALL.csv')
    
    # DICOM directories
    dicom_base_dir = os.path.join(base_dir, 'data', 'DICOM-All')
    ct_export_dir = os.path.join(dicom_base_dir, 'CT_Export')
    km_export_dir = os.path.join(dicom_base_dir, 'KM_Export')
    
    # Create output directory for NPZ files
    npz_output_dir = os.path.join(base_dir, 'data', 'npz_files_all')
    os.makedirs(npz_output_dir, exist_ok=True)
    print(f"Created output directory: {npz_output_dir}")
    
    # Initialize pseudonymization mapping
    pseudo_mapping = []
    
    # Read the CSV file with semicolon separator
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Read the first line to get the header
            header_line = f.readline().strip()
            headers = header_line.split(';')
            print(f"CSV Headers: {headers}")
            
            # Reset file pointer
            f.seek(0)
            
            # Read the CSV file
            reader = csv.DictReader(f, delimiter=';')
            cases = list(reader)
            
            # Debug: Print first case to see actual field names
            if cases:
                print("First case field names:")
                for key, value in cases[0].items():
                    print(f"  '{key}': '{value}'")
            
            print(f"Successfully read {len(cases)} cases from CSV.")
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return
    
    # Add train_split argument
    parser = argparse.ArgumentParser(description='Create NPZ files from DICOM data')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion of data to use for training (default: 0.8)')
    args = parser.parse_args()
    
    # Shuffle and split cases
    train_size = int(len(cases) * args.train_split)
    random.shuffle(cases)
    train_cases = cases[:train_size]
    test_cases = cases[train_size:]
    
    # Create output directories
    train_output_dir = os.path.join(npz_output_dir, 'train')
    os.makedirs(train_output_dir, exist_ok=True)
    test_output_dir = os.path.join(npz_output_dir, 'test')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Process training cases
    for idx, case in enumerate(train_cases):
        try:
            # Debug - print case headers again for first item
            if idx == 0:
                print("Available fields in first case:")
                for key in case.keys():
                    print(f"  '{key}'")
            
            # Extract the patient ID - try several possible field names
            patient_id = None
            for field in ['Patient-ID', 'Patient ID', 'PatientID', 'Patient_ID']:
                if field in case and case[field].strip():
                    patient_id = case[field].strip()
                    break
                    
            # If still not found, try to get the first item
            if not patient_id and len(case) > 0:
                patient_id = list(case.values())[0].strip()
                print(f"Using first value as patient_id: {patient_id}")
            
            # Get Label2
            label2 = None
            for field in ['Label2', 'Label 2', 'label2']:
                if field in case and case[field].strip():
                    label2 = case[field].strip()
                    break
            
            if not label2:
                # If Label2 not found, use a default or another field
                label2 = "unknown_label"
            
            if not patient_id:
                print(f"Case {idx+1}: Missing Patient-ID. Skipping.")
                pseudo_mapping.append({
                    'Case_Number': idx+1,
                    'Patient_ID': 'Unknown',
                    'Label2': label2,
                    'Status': 'Skipped - Missing Patient-ID',
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
            
            print(f"\nProcessing case {idx+1}/{len(train_cases)}: Patient {patient_id}")
            
            # Determine which export directory to use based on KM_Series_UID
            km_series_uid = None
            for field in ['KM_Series_UID', 'KM Series UID', 'KMSeriesUID']:
                if field in case and case[field].strip():
                    km_series_uid = case[field].strip()
                    break
            
            if km_series_uid:
                export_dir = km_export_dir
                series_uid = km_series_uid
                print(f"  Using KM_Export with Series_UID: {series_uid}")
            else:
                export_dir = ct_export_dir
                series_uid = None
                for field in ['SeriesUID', 'Series UID', 'Series_UID']:
                    if field in case and case[field].strip():
                        series_uid = case[field].strip()
                        break
                print(f"  Using CT_Export with Series_UID: {series_uid}")
            
            if not series_uid:
                print(f"  Missing Series UID. Skipping.")
                pseudo_mapping.append({
                    'Case_Number': idx+1,
                    'Patient_ID': patient_id,
                    'Label2': label2,
                    'Status': 'Skipped - Missing Series UID',
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
            
            # Locate DICOM data in the directory structure
            patient_dir = os.path.join(export_dir, patient_id)
            
            if not os.path.exists(patient_dir):
                print(f"  Patient directory not found: {patient_dir}. Skipping.")
                pseudo_mapping.append({
                    'Case_Number': idx+1,
                    'Patient_ID': patient_id,
                    'Label2': label2,
                    'Status': 'Skipped - Patient directory not found',
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
            
            # Look for series_uid directory
            series_dir = None
            for root, dirs, files in os.walk(patient_dir):
                dir_name = os.path.basename(root)
                if dir_name == series_uid:
                    series_dir = root
                    break
            
            if not series_dir:
                print(f"  Series directory for UID {series_uid} not found. Skipping.")
                pseudo_mapping.append({
                    'Case_Number': idx+1,
                    'Patient_ID': patient_id,
                    'Label2': label2,
                    'Status': 'Skipped - Series directory not found',
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
            # Extract case_kind
            case_kind = None
            for field in ['CaseKind', 'Case Kind', 'casekind']:
                if field in case and case[field].strip():
                    case_kind = case[field].strip()
                    break
                    
            # Create the NPZ filename
            if case_kind:
                npz_filename = f"{case_kind}_case_{idx+1}_{label2}"
            else:
                npz_filename = f"case_{idx+1}_{label2}"
            
            print(f"  Found DICOM directory: {series_dir}")
            print(f"  Creating NPZ file: {npz_filename}.npz")
            
            # Call the dicom_converter function
            try:
                npz_path = convert_dicom_to_npz(
                    dicom_dir=series_dir,
                    output_dir=train_output_dir,
                    case_name=npz_filename,
                    save_nii=False
                )
                
                print(f"  Successfully created NPZ file: {os.path.basename(npz_path)}")
                pseudo_mapping.append({
                    'Case_Number': idx+1,
                    'Patient_ID': patient_id,
                    'Label2': label2,
                    'Status': 'Success',
                    'NPZ_Filename': os.path.basename(npz_path),
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except Exception as e:
                error_msg = str(e)
                print(f"  Error converting DICOM to NPZ: {error_msg}")
                pseudo_mapping.append({
                    'Case_Number': idx+1,
                    'Patient_ID': patient_id,
                    'Label2': label2,
                    'Status': f'Error - {error_msg[:100]}',
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
        except Exception as e:
            print(f"  Unexpected error processing case {idx+1}: {str(e)}")
            pseudo_mapping.append({
                'Case_Number': idx+1,
                'Patient_ID': patient_id if 'patient_id' in locals() else 'Unknown',
                'Label2': label2 if 'label2' in locals() else 'Unknown',
                'Status': f'Error - {str(e)[:100]}',
                'NPZ_Filename': 'N/A',
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Process test cases
    for idx, case in enumerate(test_cases):
        try:
            # Debug - print case headers again for first item
            if idx == 0:
                print("Available fields in first case:")
                for key in case.keys():
                    print(f"  '{key}'")
            
            # Extract the patient ID - try several possible field names
            patient_id = None
            for field in ['Patient-ID', 'Patient ID', 'PatientID', 'Patient_ID']:
                if field in case and case[field].strip():
                    patient_id = case[field].strip()
                    break
                    
            # If still not found, try to get the first item
            if not patient_id and len(case) > 0:
                patient_id = list(case.values())[0].strip()
                print(f"Using first value as patient_id: {patient_id}")
            
            # Get Label2
            label2 = None
            for field in ['Label2', 'Label 2', 'label2']:
                if field in case and case[field].strip():
                    label2 = case[field].strip()
                    break
            
            if not label2:
                # If Label2 not found, use a default or another field
                label2 = "unknown_label"
            
            if not patient_id:
                print(f"Case {idx+1}: Missing Patient-ID. Skipping.")
                pseudo_mapping.append({
                    'Case_Number': idx+1+len(train_cases),
                    'Patient_ID': 'Unknown',
                    'Label2': label2,
                    'Status': 'Skipped - Missing Patient-ID',
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
            
            print(f"\nProcessing case {idx+1+len(train_cases)}/{len(cases)}: Patient {patient_id}")
            
            # Determine which export directory to use based on KM_Series_UID
            km_series_uid = None
            for field in ['KM_Series_UID', 'KM Series UID', 'KMSeriesUID']:
                if field in case and case[field].strip():
                    km_series_uid = case[field].strip()
                    break
            
            if km_series_uid:
                export_dir = km_export_dir
                series_uid = km_series_uid
                print(f"  Using KM_Export with Series_UID: {series_uid}")
            else:
                export_dir = ct_export_dir
                series_uid = None
                for field in ['SeriesUID', 'Series UID', 'Series_UID']:
                    if field in case and case[field].strip():
                        series_uid = case[field].strip()
                        break
                print(f"  Using CT_Export with Series_UID: {series_uid}")
            
            if not series_uid:
                print(f"  Missing Series UID. Skipping.")
                pseudo_mapping.append({
                    'Case_Number': idx+1+len(train_cases),
                    'Patient_ID': patient_id,
                    'Label2': label2,
                    'Status': 'Skipped - Missing Series UID',
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
            
            # Locate DICOM data in the directory structure
            patient_dir = os.path.join(export_dir, patient_id)
            
            if not os.path.exists(patient_dir):
                print(f"  Patient directory not found: {patient_dir}. Skipping.")
                pseudo_mapping.append({
                    'Case_Number': idx+1+len(train_cases),
                    'Patient_ID': patient_id,
                    'Label2': label2,
                    'Status': 'Skipped - Patient directory not found',
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
            
            # Look for series_uid directory
            series_dir = None
            for root, dirs, files in os.walk(patient_dir):
                dir_name = os.path.basename(root)
                if dir_name == series_uid:
                    series_dir = root
                    break
            
            if not series_dir:
                print(f"  Series directory for UID {series_uid} not found. Skipping.")
                pseudo_mapping.append({
                    'Case_Number': idx+1+len(train_cases),
                    'Patient_ID': patient_id,
                    'Label2': label2,
                    'Status': 'Skipped - Series directory not found',
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
            # Extract case_kind
            case_kind = None
            for field in ['CaseKind', 'Case Kind', 'casekind']:
                if field in case and case[field].strip():
                    case_kind = case[field].strip()
                    break
                    
            # Create the NPZ filename
            if case_kind:
                npz_filename = f"{case_kind}_case_{idx+1+len(train_cases)}_{label2}"
            else:
                npz_filename = f"case_{idx+1+len(train_cases)}_{label2}"
            
            print(f"  Found DICOM directory: {series_dir}")
            print(f"  Creating NPZ file: {npz_filename}.npz")
            
            # Call the dicom_converter function
            try:
                npz_path = convert_dicom_to_npz(
                    dicom_dir=series_dir,
                    output_dir=test_output_dir,
                    case_name=npz_filename,
                    save_nii=False
                )
                
                print(f"  Successfully created NPZ file: {os.path.basename(npz_path)}")
                pseudo_mapping.append({
                    'Case_Number': idx+1+len(train_cases),
                    'Patient_ID': patient_id,
                    'Label2': label2,
                    'Status': 'Success',
                    'NPZ_Filename': os.path.basename(npz_path),
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except Exception as e:
                error_msg = str(e)
                print(f"  Error converting DICOM to NPZ: {error_msg}")
                pseudo_mapping.append({
                    'Case_Number': idx+1+len(train_cases),
                    'Patient_ID': patient_id,
                    'Label2': label2,
                    'Status': f'Error - {error_msg[:100]}',
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
        except Exception as e:
            print(f"  Unexpected error processing case {idx+1+len(train_cases)}: {str(e)}")
            pseudo_mapping.append({
                'Case_Number': idx+1+len(train_cases),
                'Patient_ID': patient_id if 'patient_id' in locals() else 'Unknown',
                'Label2': label2 if 'label2' in locals() else 'Unknown',
                'Status': f'Error - {str(e)[:100]}',
                'NPZ_Filename': 'N/A',
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Write pseudonymization mapping to CSV
    pseudo_csv_path = os.path.join(npz_output_dir, f"pseudonymization_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    try:
        df = pd.DataFrame(pseudo_mapping)
        df.to_csv(pseudo_csv_path, index=False)
        print(f"\nPseudonymization mapping saved to: {pseudo_csv_path}")
    except Exception as e:
        print(f"Error writing pseudonymization mapping: {str(e)}")
        
        # Fallback to simple CSV writing
        try:
            with open(pseudo_csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['Case_Number', 'Patient_ID', 'Label2', 'Status', 'NPZ_Filename', 'Timestamp']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in pseudo_mapping:
                    writer.writerow(item)
            print(f"Pseudonymization mapping saved using fallback method.")
        except Exception as e2:
            print(f"Failed to write mapping even with fallback method: {str(e2)}")
    
    # Print summary
    success_count = sum(1 for item in pseudo_mapping if item['Status'].startswith('Success'))
    print("\nSummary:")
    print(f"Total cases processed: {len(cases)}")
    print(f"Successfully converted: {success_count}")
    print(f"Failed/skipped: {len(cases) - success_count}")
    print(f"NPZ files saved to: {npz_output_dir}")
    print(f"Pseudonymization mapping saved to: {pseudo_csv_path}")

if __name__ == "__main__":
    print("Starting DICOM to NPZ conversion using dicom_converter.py...")
    main()
    print("Process completed.")
