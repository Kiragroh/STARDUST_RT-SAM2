# %% move files based on csv file
import numpy as np
from os import listdir, makedirs
from os.path import join, exists
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import multiprocessing as mp
import argparse
import json

# Hier wird das label_dict aus der GUI-Konfiguration geladen
def load_label_dict():
    if exists("sam_gui_settings.json"):
        try:
            with open("sam_gui_settings.json", 'r') as f:
                settings = json.load(f)
                label_dict = {}
                # Labels aus String-Keys in Integer-Keys umwandeln
                for k, v in settings.get("label_names", {}).items():
                    label_dict[int(k)] = v
                return label_dict
        except Exception as e:
            print(f"Fehler beim Laden der Label-Namen aus der GUI: {e}")
    
    # Fallback zu Standard-Labels wenn keine GUI-Konfiguration existiert
    return {
        1: 'Liver',
        2: 'Right Kidney',
        3: 'Spleen',
        4: 'Pancreas',
        5: 'Aorta',
        6: 'Inferior Vena Cava', # IVC
        7: 'Right Adrenal Gland', # RAG
        8: 'Left Adrenal Gland', # LAG
        9: 'Gallbladder',
        10: 'Esophagus',
        11: 'Stomach',
        13: 'Left Kidney'
    }

label_dict = load_label_dict()

def compute_multi_class_dsc(gt, npz_seg, eval_labels_arg):
    dsc = {}
    for i in eval_labels_arg:  # Nur die übergebenen Labels berücksichtigen
        gt_i = gt == i
        organ_name = label_dict.get(i, f'Organ_{i}')  # Fallback, falls Label nicht im Dictionary
        if organ_name in npz_seg.files:
            seg_i = npz_seg[organ_name]
        elif f'Organ_{i}' in npz_seg.files:  # Fallback für alte Segmentierungen
            seg_i = npz_seg[f'Organ_{i}']
        else:
            seg_i = np.zeros_like(gt_i)
        if np.sum(gt_i)==0 and np.sum(seg_i)==0:
            dsc[i] = 1
        elif np.sum(gt_i)==0 and np.sum(seg_i)>0:
            dsc[i] = 0
        else:
            dsc[i] = compute_dice_coefficient(gt_i, seg_i)

    return dsc


def compute_multi_class_nsd(gt, npz_seg, spacing, eval_labels_arg, tolerance=2.0):
    nsd = {}
    for i in eval_labels_arg:  # Nur die übergebenen Labels berücksichtigen
        gt_i = gt == i
        organ_name = label_dict.get(i, f'Organ_{i}')  # Fallback, falls Label nicht im Dictionary
        if organ_name in npz_seg.files:
            seg_i = npz_seg[organ_name]
        elif f'Organ_{i}' in npz_seg.files:  # Fallback für alte Segmentierungen
            seg_i = npz_seg[f'Organ_{i}']
        else:
            seg_i = np.zeros_like(gt_i)
        if np.sum(gt_i)==0 and np.sum(seg_i)==0:
            nsd[i] = 1
        elif np.sum(gt_i)==0 and np.sum(seg_i)>0:
            nsd[i] = 0
        else:
            surface_distance = compute_surface_distances(
                gt_i, seg_i, spacing_mm=spacing
            )
            nsd[i] = compute_surface_dice_at_tolerance(surface_distance, tolerance)

    return nsd


parser = argparse.ArgumentParser()

parser.add_argument('-s', '--seg_dir', default=None, type=str)
parser.add_argument('-g', '--gt_dir',  default=None, type=str)
parser.add_argument('-csv_dir', default='./', type=str)
parser.add_argument('-nw', '--num_workers', default=8, type=int)
parser.add_argument('-nsd', default=True, type=bool, help='set it to False to disable NSD computation and save time')
args = parser.parse_args()

seg_dir = args.seg_dir
gt_dir = args.gt_dir
num_workers = args.num_workers
compute_NSD = args.nsd
csv_dir = args.csv_dir
makedirs(csv_dir, exist_ok=True)

def compute_metrics(args):
    """
    return
    npz_name: str
    dsc: float
    """
    npz_name, eval_labels_arg = args  # Unpack the arguments
    
    try:
        npz_seg = np.load(join(seg_dir, npz_name), allow_pickle=True, mmap_mode='r')
    except FileNotFoundError as e:
        print(f'INFO: File {npz_name} is missing in submission (possibly skipped during inference). Setting metrics to -1.')
        # Rückgabe eines Placeholder-Ergebnisses statt einen Fehler zu werfen
        dsc_dict = {label_id: -1. for label_id in eval_labels_arg}
        nsd_dict = {label_id: -1. for label_id in eval_labels_arg} if compute_NSD else None
        return npz_name, dsc_dict, nsd_dict if compute_NSD else dsc_dict

    try:
        npz_gt = np.load(join(gt_dir, npz_name), allow_pickle=True, mmap_mode='r')
    except FileNotFoundError as e:
        print(e)
        raise FileNotFoundError(f'File {npz_name} is not a valid case')

    gts = npz_gt['gts']
    spacing = npz_gt['spacing']
    
    dsc = compute_multi_class_dsc(gts, npz_seg, eval_labels_arg)
    if compute_NSD:
        nsd = compute_multi_class_nsd(gts, npz_seg, spacing, eval_labels_arg)
    if compute_NSD:
        return npz_name, dsc, nsd
    else:
        return npz_name, dsc

if __name__ == '__main__':
    # Zuerst alle verfügbaren Ground-Truth-Dateien durchsuchen und vorhandene Labels identifizieren
    gt_labels = set()
    seg_labels = set()
    npz_names = listdir(gt_dir)
    npz_names = [npz_name for npz_name in npz_names if npz_name.endswith('.npz')]
    
    # Eine Stichprobe von npz-Dateien untersuchen, um die Labels zu identifizieren
    sample_size = min(10, len(npz_names))  # Maximal 10 Dateien untersuchen
    sample_npz_names = np.random.choice(npz_names, sample_size, replace=False) if sample_size > 0 else []
    
    # Ground-Truth Labels identifizieren
    for npz_name in sample_npz_names:
        try:
            npz_gt = np.load(join(gt_dir, npz_name), allow_pickle=True, mmap_mode='r')
            gts = npz_gt['gts']
            unique_labels = np.unique(gts)
            for label in unique_labels:
                if label > 0:  # Ignoriere Hintergrund (Label 0)
                    gt_labels.add(int(label))
        except Exception as e:
            print(f"Fehler beim Lesen von GT {npz_name}: {e}")
    
    # Segmentierungs-Labels identifizieren
    for npz_name in sample_npz_names:
        try:
            seg_path = join(seg_dir, npz_name)
            if exists(seg_path):
                npz_seg = np.load(seg_path, allow_pickle=True, mmap_mode='r')
                
                # Suche nach Labels in den Segmentierungen
                # 1. Variante: Label als Dateiname in .npz
                for key in npz_seg.files:
                    if key in label_dict.values():  # Organname gefunden
                        for label_id, organ_name in label_dict.items():
                            if organ_name == key:
                                seg_labels.add(label_id)
                    elif key.startswith('Organ_'):  # Format 'Organ_X'
                        try:
                            label_id = int(key.split('_')[1])
                            seg_labels.add(label_id)
                        except:
                            pass
                
                # 2. Variante: Arrays in .npz auswerten, wenn GT-Format direkt verwendet wird
                if 'pred' in npz_seg.files:
                    pred = npz_seg['pred']
                    unique_labels = np.unique(pred)
                    for label in unique_labels:
                        if label > 0:  # Ignoriere Hintergrund (Label 0)
                            seg_labels.add(int(label))
        except Exception as e:
            print(f"Fehler beim Lesen von SEG {npz_name}: {e}")
    
    # Wenn keine Labels in Segmentierungen gefunden wurden, versuchen wir es mit einem anderen Ansatz
    if not seg_labels:
        print("Keine Labels in Segmentierungen gefunden. Versuche alternative Methode...")
        try:
            # Überprüfe alle Dateien in seg_dir
            all_seg_files = listdir(seg_dir)
            for seg_file in all_seg_files:
                if seg_file.endswith('.npz'):
                    seg_path = join(seg_dir, seg_file)
                    npz_seg = np.load(seg_path, allow_pickle=True, mmap_mode='r')
                    for key in npz_seg.files:
                        if key != 'case' and key != 'spacing':  # Keine Metadaten
                            try:
                                if key in label_dict.values():  # Organname gefunden
                                    for label_id, organ_name in label_dict.items():
                                        if organ_name == key:
                                            seg_labels.add(label_id)
                                elif 'GTV' in key or key == '1':  # Speziell für Label 1 / GTV
                                    seg_labels.add(1)
                            except:
                                pass
        except Exception as e:
            print(f"Fehler bei alternativer Methode: {e}")
    
    # Falls keine Labels gefunden wurden, verwende Label 1 als Standard
    if not gt_labels:
        gt_labels = {1}
        print("Keine Labels in Ground-Truth gefunden, verwende Label 1 als Standard.")
    
    if not seg_labels:
        seg_labels = {1}
        print("Keine Labels in Segmentierungen gefunden, verwende Label 1 als Standard.")
    
    # Nur Labels evaluieren, die in beiden Datensätzen vorkommen
    eval_labels = sorted(list(gt_labels.intersection(seg_labels)))
    
    # Fallback, falls keine übereinstimmenden Labels gefunden wurden
    if not eval_labels:
        eval_labels = [1]
        print("Keine übereinstimmenden Labels gefunden, verwende Label 1 als Standard.")
    
    print(f"Erkannte Labels in den Ground-Truth-Daten: {sorted(list(gt_labels))}")
    print(f"Erkannte Labels in den Segmentierungen: {sorted(list(seg_labels))}")
    print(f"Labels, die evaluiert werden: {eval_labels}")
    
    seg_metrics = OrderedDict()
    seg_metrics['case'] = []
    
    for k in eval_labels:
        seg_metrics[f"{k}_DSC"] = []
    if compute_NSD:
        for k in eval_labels:
            seg_metrics[f"{k}_NSD"] = []
    
    npz_names = listdir(gt_dir)
    npz_names = [npz_name for npz_name in npz_names if npz_name.endswith('.npz')]
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(npz_names)) as pbar:
            # Erstelle eine Liste von Argumenten für compute_metrics
            args_list = [(npz_name, eval_labels) for npz_name in npz_names]
            
            if compute_NSD:
                for i, (npz_name, dsc, nsd) in enumerate(pool.imap_unordered(compute_metrics, args_list)):
                    seg_metrics['case'].append(npz_name)
                    for k, v in dsc.items():
                        seg_metrics[f"{k}_DSC"].append(np.round(v, 4))
                    for k, v in nsd.items():
                        seg_metrics[f"{k}_NSD"].append(np.round(v, 4))
                    pbar.update()
            else:
                for i, (npz_name, dsc) in enumerate(pool.imap_unordered(compute_metrics, args_list)):
                    seg_metrics['case'].append(npz_name)
                    for k, v in dsc.items():
                        seg_metrics[f"{k}_DSC"].append(np.round(v, 4))
                    pbar.update()

    df = pd.DataFrame(seg_metrics)
    df.to_csv(join(csv_dir, 'metrics.csv'), index=False)

    ## make summary csv
    df_dsc = df[["case"] + [f"{k}_DSC" for k in eval_labels]].copy()
    df_dsc_mean = df_dsc[[f"{k}_DSC" for k in eval_labels]].mean()
    df_dsc_mean.to_csv(join(csv_dir, 'dsc_summary.csv'))

    if compute_NSD:
        df_nsd = df[["case"] + [f"{k}_NSD" for k in eval_labels]].copy()
        df_nsd_mean = df_nsd[[f"{k}_NSD" for k in eval_labels]].mean()
        df_nsd_mean.to_csv(join(csv_dir, 'nsd_summary.csv'))