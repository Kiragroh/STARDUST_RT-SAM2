"""
Convert the preprocessed .npz files to .npy files for training
This version allows filtering by filename pattern
"""
# %% import packages
import numpy as np
import os
import re
join = os.path.join
listdir = os.listdir
makedirs = os.makedirs
from tqdm import tqdm
import multiprocessing as mp
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-npz_dir", type=str, default='data/npz_train/CT_Abd',
                    help="Path to the directory containing preprocessed .npz data, [default: data/npz/MedSAM_train/CT_Abd]")
parser.add_argument("-npy_dir", type=str, default="data/npy",
                    help="Path to the directory where the .npy files for training will be saved, [default: ./data/npy]")
parser.add_argument("-num_workers", type=int, default=4,
                    help="Number of workers to convert npz to npy in parallel, [default: 4]")
parser.add_argument("-target_label", type=int, default=1,
                    help="Only keep this label in ground truth masks, set to -1 to keep all labels [default: 1]")
parser.add_argument("-filter_pattern", type=str, default="",
                    help="Only process files that match this pattern, e.g., 'Kopf' to process only head cases [default: '' = process all]")
args = parser.parse_args()
# %%
npz_dir = args.npz_dir
npy_dir = args.npy_dir
target_label = args.target_label
filter_pattern = args.filter_pattern
makedirs(join(npy_dir, "imgs"), exist_ok=True)
makedirs(join(npy_dir, "gts"), exist_ok=True)

# Filter the NPZ files based on the pattern
if filter_pattern:
    npz_names = [f for f in listdir(npz_dir) if f.endswith(".npz") and filter_pattern in f]
    print(f"Found {len(npz_names)} NPZ files matching pattern '{filter_pattern}'")
else:
    npz_names = [f for f in listdir(npz_dir) if f.endswith(".npz")]
    print(f"Found {len(npz_names)} total NPZ files")

num_workers = args.num_workers

def process_ground_truth(gt, target_label):
    """Process ground truth to keep only the target label"""
    if target_label < 0:  # Keep all labels
        return gt
    # Create binary mask where target label is 1 and everything else is 0
    return np.where(gt == target_label, 1, 0)

# convert npz files to npy files
def convert_npz_to_npy(npz_name):
    """
    Convert npz files to npy files for training

    Parameters
    ----------
    npz_name : str
        Name of the npz file to be converted
    """
    name = npz_name.split(".npz")[0]
    npz_path = join(npz_dir, npz_name)
    npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    imgs = npz["imgs"]
    gts = npz["gts"]
    if len(gts.shape) > 2: ## 3D image
        for i in range(imgs.shape[0]):
            img_i = imgs[i, :, :]
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
            gt_i = gts[i, :, :]
            # Process ground truth to keep only target label
            gt_i = process_ground_truth(gt_i, target_label)
            gt_i = np.uint8(gt_i)
            gt_i = cv2.resize(gt_i, (256, 256), interpolation=cv2.INTER_NEAREST)
            assert gt_i.shape == (256, 256)
            np.save(join(npy_dir, "imgs", name + "-" + str(i).zfill(3) + ".npy"), img_3c)
            np.save(join(npy_dir, "gts", name + "-" + str(i).zfill(3) + ".npy"), gt_i)
    else: ## 2D image
        if len(imgs.shape) < 3:
            img_3c = np.repeat(imgs[:, :, None], 3, axis=-1)
        else:
            img_3c = imgs

        gt_i = gts
        # Process ground truth to keep only target label
        gt_i = process_ground_truth(gt_i, target_label)
        gt_i = np.uint8(gt_i)
        gt_i = cv2.resize(gt_i, (256, 256), interpolation=cv2.INTER_NEAREST)
        assert gt_i.shape == (256, 256)
        np.save(join(npy_dir, "imgs", name + ".npy"), img_3c)
        np.save(join(npy_dir, "gts", name + ".npy"), gt_i)
# %%
if __name__ == "__main__":
    filter_msg = f" matching '{filter_pattern}'" if filter_pattern else ""
    print(f"Converting {len(npz_names)} NPZ files{filter_msg} to NPY, keeping only label {target_label if target_label >= 0 else 'all'}")
    
    if not npz_names:
        print("No files to process. Check your filter pattern or directory.")
        exit(0)
        
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(npz_names)) as pbar:
            pbar.set_description("Converting npz to npy")
            for i, _ in enumerate(pool.imap_unordered(convert_npz_to_npy, npz_names)):
                pbar.update()
    print("Conversion completed!")
