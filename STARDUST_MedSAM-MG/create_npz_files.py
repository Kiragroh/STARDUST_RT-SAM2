#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NPZ File Creator from DICOM Data
This script creates NPZ files from DICOM data based on clinical cases in a CSV file
or by recursively scanning directories for DICOM data.

It uses the existing dicom_converter.py script and creates a pseudonymization mapping.

Usage:
    # Using a CSV file (traditional method):
    python create_npz_files.py --input_dir /path/to/dicom_folder --output_dir /path/to/npz_output --csv_file /path/to/cases.csv
    
    # Recursive scan of directories without CSV:
    python create_npz_files.py --input_dir /path/to/dicom_folder --output_dir /path/to/npz_output --recursive
"""

import os
import sys
import csv
import glob
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path

# Import the dicom_converter module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dicom_converter import convert_dicom_to_npz

def scan_for_dicom_dirs(base_dir):
    """
    Recursively scan a directory for valid DICOM data sets.
    A valid set is a directory containing multiple CT files and at least one RS file.
    
    Args:
        base_dir: The base directory to scan
        
    Returns:
        List of tuples (directory_path, directory_name) containing valid DICOM data sets
    """
    valid_dirs = []
    
    print(f"Scanning {base_dir} for DICOM data...")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Check if this directory contains CT and RS files
        ct_files = [f for f in files if f.startswith("CT") and f.endswith(".dcm")]
        rs_files = [f for f in files if f.startswith("RS") and f.endswith(".dcm")]
        
        # If we have multiple CT files and at least one RS file, consider it a valid DICOM data set
        if len(ct_files) > 2 and len(rs_files) > 0:
            dir_name = os.path.basename(root)
            valid_dirs.append((root, dir_name))
            print(f"  Found valid DICOM data set in: {root}")
    
    print(f"Found {len(valid_dirs)} valid DICOM data sets.")
    return valid_dirs

def process_from_csv(csv_path, dicom_base_dir, ct_export_dir, km_export_dir, npz_output_dir):
    """Process DICOM data using information from a CSV file"""
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
    
    # Process each clinical case
    print("\nProcessing cases:")
    for idx, case in enumerate(cases):
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
            
            print(f"\nProcessing case {idx+1}/{len(cases)}: Patient {patient_id}")
            
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
            
            # Create the NPZ filename
            npz_filename = f"case_{idx+1}_{label2}"
            
            print(f"  Found DICOM directory: {series_dir}")
            print(f"  Creating NPZ file: {npz_filename}.npz")
            
            # Call the dicom_converter function
            try:
                npz_path = convert_dicom_to_npz(
                    dicom_dir=series_dir,
                    output_dir=npz_output_dir,
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
                print(f"  Error converting DICOM to NPZ: {str(e)}")
                pseudo_mapping.append({
                    'Case_Number': idx+1,
                    'Patient_ID': patient_id,
                    'Label2': label2,
                    'Status': f'Error - {str(e)}',
                    'NPZ_Filename': 'N/A',
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        except Exception as e:
            print(f"Error processing case: {str(e)}")
    
    # Save the pseudonymization mapping to CSV
    pseudo_csv_path = os.path.join(npz_output_dir, "pseudonymization_mapping.csv")
    try:
        pd.DataFrame(pseudo_mapping).to_csv(pseudo_csv_path, index=False)
        print(f"Pseudonymization mapping saved to: {pseudo_csv_path}")
    except Exception as e:
        print(f"Error saving pseudonymization mapping: {str(e)}")

def process_recursive(input_dir, output_dir):
    """Process DICOM data by recursively scanning directories"""
    # Find all valid DICOM directories
    valid_dirs = scan_for_dicom_dirs(input_dir)
    
    if not valid_dirs:
        print("No valid DICOM directories found. Please check your input directory.")
        return
    
    # Initialize pseudonymization mapping
    pseudo_mapping = []
    
    # Process each directory
    for idx, (dicom_dir, dir_name) in enumerate(valid_dirs):
        print(f"\nProcessing directory {idx+1}/{len(valid_dirs)}: {dicom_dir}")
        
        # Create the NPZ filename
        npz_filename = f"case_{idx+1}_{dir_name}"
        
        # Call the dicom_converter function
        try:
            npz_path = convert_dicom_to_npz(
                dicom_dir=dicom_dir,
                output_dir=output_dir,
                case_name=npz_filename,
                save_nii=False
            )
            
            print(f"  Successfully created NPZ file: {os.path.basename(npz_path)}")
            pseudo_mapping.append({
                'Case_Number': idx+1,
                'Directory': dicom_dir,
                'DirName': dir_name,
                'Status': 'Success',
                'NPZ_Filename': os.path.basename(npz_path),
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception as e:
            print(f"  Error converting DICOM to NPZ: {str(e)}")
            pseudo_mapping.append({
                'Case_Number': idx+1,
                'Directory': dicom_dir,
                'DirName': dir_name,
                'Status': f'Error - {str(e)}',
                'NPZ_Filename': 'N/A',
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Save the pseudonymization mapping to CSV
    pseudo_csv_path = os.path.join(output_dir, "directory_mapping.csv")
    try:
        pd.DataFrame(pseudo_mapping).to_csv(pseudo_csv_path, index=False)
        print(f"Directory mapping saved to: {pseudo_csv_path}")
    except Exception as e:
        print(f"Error saving directory mapping: {str(e)}")

def main(input_dir=None, output_dir=None, csv_file=None, recursive=False):
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set input and output directories
    if input_dir is None:
        dicom_base_dir = os.path.join(base_dir, 'data', 'DICOM')
    else:
        dicom_base_dir = input_dir
    
    ct_export_dir = os.path.join(dicom_base_dir, 'CT_Export') if os.path.exists(os.path.join(dicom_base_dir, 'CT_Export')) else dicom_base_dir
    km_export_dir = os.path.join(dicom_base_dir, 'KM_Export') if os.path.exists(os.path.join(dicom_base_dir, 'KM_Export')) else dicom_base_dir
    
    if output_dir is None:
        npz_output_dir = os.path.join(base_dir, 'data', 'npz_files')
    else:
        npz_output_dir = output_dir
    
    os.makedirs(npz_output_dir, exist_ok=True)
    print(f"Created output directory: {npz_output_dir}")
    
    # Process differently based on whether we're using CSV or recursive scan
    if recursive:
        process_recursive(dicom_base_dir, npz_output_dir)
    else:
        if csv_file is None:
            csv_file = os.path.join(dicom_base_dir, 'PlansWithGTV_filtered_Mine_final.csv')
            if not os.path.exists(csv_file):
                print(f"CSV file not found at {csv_file}. Switching to recursive mode.")
                process_recursive(dicom_base_dir, npz_output_dir)
                return
        
        process_from_csv(csv_file, dicom_base_dir, ct_export_dir, km_export_dir, npz_output_dir)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert DICOM files to NPZ format for MedSAM2.')
    parser.add_argument('--input_dir', type=str, help='Input directory containing DICOM files')
    parser.add_argument('--output_dir', type=str, help='Output directory for NPZ files')
    parser.add_argument('--csv_file', type=str, help='CSV file with case information (optional)')
    parser.add_argument('--recursive', action='store_true', help='Recursively scan directories for DICOM data without using CSV')
    
    args = parser.parse_args()
    
    print("Starting DICOM to NPZ conversion using dicom_converter.py...")
    main(input_dir=args.input_dir, output_dir=args.output_dir, csv_file=args.csv_file, recursive=args.recursive)
    print("Process completed.")
