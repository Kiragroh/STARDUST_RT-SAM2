import numpy as np
import os
from os.path import join, exists
from os import makedirs, listdir
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import SimpleITK as sitk
import cv2
from skimage import measure
from tqdm import tqdm
import argparse
import json
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Ausgabe auf Konsole
)
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms

from torch import multiprocessing as mp

# Import der neuen Prompt-Utilities
import sys
# Verwende einen absoluten Pfad anstelle eines relativen Pfads
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt_utils import (
    get_bbox, generate_negative_masks, generate_random_points, 
    generate_prompts, save_prompt_debug_visualizations
)

#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

def medsam_point_inference(model, features, points, labels, H, W, device):
    img_embed, high_res_features = features["image_embed"], features["high_res_feats"]
    
    points_torch = torch.as_tensor(points, dtype=torch.float32, device=device)[None, ...]
    labels_torch = torch.as_tensor(labels, dtype=torch.int, device=device)[None, ...]
    
    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model.sam2_model.sam_prompt_encoder(
            points=(points_torch, labels_torch),
            boxes=None,
            masks=None,
        )
        
        mask_logits, _, _, _ = model.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
    
    mask_logits = mask_logits[0, 0]
    mask_1024 = (mask_logits > 0).cpu().numpy()
    mask_original = cv2.resize(mask_1024.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    
    return mask_original

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

# Fallback, falls keine Label-Namen geladen werden konnten
if label_names is None:
    label_names = {
        1: 'GTV',  # Erstes Label immer als GTV
        2: 'Target',
        3: 'Spleen',
        4: 'Pancreas',
        5: 'Aorta',
        6: 'Inferior Vena Cava',
        7: 'Right Adrenal Gland',
        8: 'Left Adrenal Gland',
        9: 'Gallbladder',
        10: 'Esophagus',
        11: 'Stomach',
        13: 'Left Kidney'
    }

# Sicherstellen, dass Label 1 immer als GTV bezeichnet wird
if 1 not in label_names:
    label_names[1] = 'GTV'
    #print("Label 1 wurde explizit als GTV gesetzt")

print(f"Geladene Label-Namen: {label_names}")

class MedSAM2(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.sam2_model = model
        # freeze prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
        

    def forward(self, image, box):
        """
        image: (B, 3, 1024, 1024)
        box: (B, 2, 2)
        """
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                box_labels = box_labels.repeat(box_torch.size(0), 1)
            concat_points = (box_coords, box_labels)

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed, # (1, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks_logits


    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]

        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features


def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC.astype(np.uint8)

image_size = 1024
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def get_bbox(mask, bbox_shift=5):
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)
    bboxes = np.array([x_min, y_min, x_max, y_max])
    return bboxes

@torch.no_grad()
def medsam_inference(
    medsam_model,
    features,
    box_1024,
    H, W
    ):
    img_embed, high_res_features = features["image_embed"], features["high_res_feats"]
    box_torch = torch.as_tensor(box_1024, dtype=torch.float32, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=img_embed.device)
        box_labels = box_labels.repeat(box_torch.size(0), 1)
    concat_points = (box_coords, box_labels)

    sparse_embeddings, dense_embeddings = medsam_model.sam2_model.sam_prompt_encoder(
        points=concat_points,
        boxes=None,
        masks=None,
    )
    low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = medsam_model.sam2_model.sam_mask_decoder(
        image_embeddings=img_embed, # (1, 256, 64, 64)
        image_pe=medsam_model.sam2_model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )

    low_res_pred = torch.sigmoid(low_res_masks_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg

# %% load model

parser = argparse.ArgumentParser(
    description="Run inference on validation set with MedSAM2"
)
parser.add_argument(
    "-data_root",
    type=str,
    default=None,
    help="Path to the data folder",
)
parser.add_argument(
    "-pred_save_dir",
    type=str,
    default=None,
    help="Path to save the segmentation results",
)
parser.add_argument(
    "-sam2_checkpoint",
    type=str,
    default='./checkpoints/sam2_hiera_tiny.pt',
    help="SAM2 pretrained model checkpoint",
)
parser.add_argument(
    "-model_cfg",
    type=str,
    default="sam2_hiera_t.yaml",
    help="Model config file"
)
parser.add_argument(
    "-medsam2_checkpoint",
    type=str,
    default="./work_dir/medsam2_t_flare22.pth",
    help="Path to the finetuned MedSAM2 model",
)
parser.add_argument("-device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-bbox_shift",
    type=int,
    default=5,
    help="Bounding box perturbation",
)
parser.add_argument(
    "-num_workers",
    type=int,
    default=16,
    help="Number of workers for multiprocessing",
)
parser.add_argument("--visualize", action="store_true", help="Save the .nii.gz segmentation results")
parser.add_argument("--label", type=str, default=None, help="Specify which label to segment, e.g. '1,2,3' for multiple labels or a single label")
parser.add_argument("--include_ct", action="store_true", help="Include CT scan as background in visualization")
parser.add_argument("--save_nii", action="store_true", help="Save the segmentation results as .nii.gz files")
parser.add_argument("-prompt_type", type=str, default="box", choices=["box", "point"], help="Type of prompt to use: box or point")
parser.add_argument("-num_pos_points", type=int, default=3, help="Number of positive points for point prompts")
parser.add_argument("-num_neg_points", type=int, default=1, help="Number of negative points for point prompts")
parser.add_argument("-min_dist_from_edge", type=int, default=3, help="Minimum distance from mask edge for point prompts")
parser.add_argument("-debug_mode", action="store_true", help="Enable debug mode to save prompt visualizations")
args = parser.parse_args()

visualize = args.visualize
use_point_prompts = args.prompt_type == "point"
debug_mode = args.debug_mode
bbox_shift = args.bbox_shift
num_pos_points = args.num_pos_points
num_neg_points = args.num_neg_points
min_dist_from_edge = args.min_dist_from_edge
save_nii = args.save_nii
include_ct = args.include_ct

data_root = args.data_root
pred_save_dir = args.pred_save_dir
makedirs(pred_save_dir, exist_ok=True)

device = args.device
model_cfg = args.model_cfg
sam2_checkpoint = args.sam2_checkpoint
medsam2_checkpoint = args.medsam2_checkpoint
num_workers = args.num_workers

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, mode="eval", apply_postprocessing=True, weights_only=False)
medsam2_checkpoint = torch.load(medsam2_checkpoint, map_location="cpu", weights_only=False)
medsam_model = MedSAM2(model=sam2_model)
medsam_model.load_state_dict(medsam2_checkpoint["model_state_dict"], strict=True)
medsam_model.eval()
sam2_transforms = SAM2Transforms(resolution=1024, mask_threshold=0)

# load data
_names = sorted(listdir(data_root))
names = [name for name in _names if name.endswith('.npz')]

def save_npz_and_nii(pred_save_dir, name, segs_dict, label_ids, img_3D=None, spacing=None, visualize=False, save_nii=False, include_ct=False):
    """
    Speichert die Segmentierungsergebnisse im NPZ- und optional im NII-Format.
    
    Args:
        pred_save_dir: Ausgabeverzeichnis
        name: Dateiname (ohne Pfad)
        segs_dict: Dictionary mit Label-ID als Schlüssel und Segmentierung als Wert
        label_ids: Liste der Label-IDs
        img_3D: 3D-Array der Originalbilder (optional)
        spacing: Spacing der Daten für NII-Konvertierung (optional)
        visualize: Flag zum Speichern der NII-Dateien
        save_nii: Flag zum Speichern der NII-Dateien
        include_ct: Flag zum Einschließen der CT-Bilder in der NPZ-Datei
    """
    # Globale label_names verwenden, die am Anfang des Skripts geladen wurden
    global label_names
    
    #print(f"In save_npz_and_nii: label_names = {label_names}")
    
    # Speicherformat anpassen, um Kompatibilität zu gewährleisten
    save_dict = {}
    
    # Keine erneute Ladung der Label-Namen - verwende stattdessen die globale Variable
    for label_id in label_ids:
        # Für Label 1 immer GTV verwenden, unabhängig von label_names
        if label_id == 1:
            organ_name = 'GTV'  # Hartkodiert für Label 1
            print(f"Label {label_id} wird als {organ_name} gespeichert")
        elif label_id in label_names and label_names[label_id]:
            organ_name = label_names[label_id]
            print(f"Label {label_id} wird als {organ_name} gespeichert (aus label_names)")
        else:
            organ_name = f'Organ_{label_id}'
            print(f"Label {label_id} wird als {organ_name} gespeichert (Fallback)")
        
        # Wichtig: Sowohl den numerischen Schlüssel (label_id) als auch den beschreibenden Schlüssel (organ_name) speichern
        # damit visualize_segmentation.py beide Formate verarbeiten kann
        save_dict[str(label_id)] = segs_dict[label_id] # Numerischer Schlüssel als String
        save_dict[organ_name] = segs_dict[label_id]    # Beschreibender Schlüssel
    
    print(f"Speichere folgende Schlüssel in NPZ: {list(save_dict.keys())}")
    
    # Bilder hinzufügen, aber nur wenn include_ct aktiviert ist
    if img_3D is not None and include_ct:
        save_dict['imgs'] = img_3D
        print("CT-Bilder wurden zur NPZ-Datei hinzugefügt")
    
    # Segmentierungen im NPZ-Format speichern
    np.savez_compressed(join(pred_save_dir, name), **save_dict)

    # NII-Dateien speichern, wenn gewünscht
    if (visualize or save_nii) and spacing is not None:
        for label_id in label_ids:
            if label_id in label_names:
                organ_name = label_names[label_id]
            else:
                organ_name = f'Organ_{label_id}'
                
            seg_sitk = sitk.GetImageFromArray(segs_dict[label_id])
            seg_sitk.SetSpacing(spacing)
            sitk.WriteImage(seg_sitk, join(pred_save_dir, name.replace('.npz', f'_{organ_name}.nii.gz')))
        
        # Auch das Originalbild speichern
        if img_3D is not None:
            img_sitk = sitk.GetImageFromArray(img_3D)
            img_sitk.SetSpacing(spacing)
            sitk.WriteImage(img_sitk, join(pred_save_dir, name.replace('.npz', '_0000.nii.gz')))

def main(name):
    logging.info(f"Processing {name}")
    name_prefix = name.split('.')[0]
    try:
        data = np.load(join(data_root, name), allow_pickle=True)
        img_3D = data["imgs"]
        gt_3D = data["gts"]
        spacing = data.get("spacing", None)

        # Bestimme die zu segmentierenden Label-IDs
        if args.label:
            label_ids = [int(i) for i in args.label.split(',')]
        else:
            all_labels = np.unique(gt_3D)
            label_ids = [int(label) for label in all_labels if label > 0]

        # Lade Labelnamen oder erstelle Fallback
        label_names_dict = load_label_names() or {}
        for label_id in label_ids:
            label_names_dict[label_id] = label_names_dict.get(label_id, f"Label_{label_id}")

        # Initialisiere Dictionaries
        debug_prompts_dict = {}
        segs_dict = {}

        for label_id in label_ids:
            segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8)
            marker_data_id = (gt_3D == label_id).astype(np.uint8)
            marker_zids = np.sort(np.unique(np.where(marker_data_id > 0)[0]))

            if len(marker_zids) == 0:
                logging.warning(f"No voxels with label {label_id} found in {name}. Skipping.")
                continue

            # Erstelle Label-Dictionary für alle Labels
            label_dict = {f"{l_id}_{label_names_dict[l_id]}": (gt_3D == l_id).astype(np.uint8) for l_id in label_ids}

            if args.prompt_type == "box":
                # Bestehende Box-Logik bleibt unverändert
                bbox_dict = {z: get_bbox(marker_data_id[z], bbox_shift=args.bbox_shift) for z in marker_zids}
                if args.debug_mode:
                    debug_prompts_dict.update({z: {"box": bbox_dict[z]} for z in marker_zids})
                bbox_areas = [np.prod(bbox_dict[z][2:] - bbox_dict[z][:2]) for z in bbox_dict.keys()]
                z_max_area = list(bbox_dict.keys())[np.argmax(bbox_areas)]
                z_min, z_max = min(bbox_dict.keys()), max(bbox_dict.keys())
                mid_slice_bbox_2d = bbox_dict[z_max_area]
                z_middle = int((z_max - z_min) / 2 + z_min)
                z_max = min(z_max + 1, img_3D.shape[0])

                for z in range(z_middle, z_max):
                    img_2d = img_3D[z]
                    img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1) if len(img_2d.shape) == 2 else img_2d
                    H, W, _ = img_3c.shape
                    img_1024_tensor = sam2_transforms(img_3c)[None, ...].to(device)
                    with torch.no_grad():
                        _features = medsam_model._image_encoder(img_1024_tensor)
                    box_1024 = mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024 if z == z_middle else get_bbox(
                        cv2.resize(segs_3d_temp[z - 1], (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    ) if np.max(segs_3d_temp[z - 1]) > 0 else mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024
                    img_2d_seg = medsam_inference(medsam_model, _features, box_1024[None, :], H, W)
                    segs_3d_temp[z, img_2d_seg > 0] = 1

                z_min = max(-1, z_min - 1)
                for z in range(z_middle - 1, z_min, -1):
                    img_2d = img_3D[z]
                    img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1) if len(img_2d.shape) == 2 else img_2d
                    H, W, _ = img_3c.shape
                    img_1024_tensor = sam2_transforms(img_3c)[None, ...].to(device)
                    with torch.no_grad():
                        _features = medsam_model._image_encoder(img_1024_tensor)
                    box_1024 = get_bbox(cv2.resize(segs_3d_temp[z + 1], (1024, 1024), interpolation=cv2.INTER_NEAREST)) if np.max(
                        segs_3d_temp[z + 1]) > 0 else mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024
                    img_2d_seg = medsam_inference(medsam_model, _features, box_1024[None, :], H, W)
                    segs_3d_temp[z, img_2d_seg > 0] = 1

            elif args.prompt_type == "point":
                # Bestimme den mittleren Slice
                z_min, z_max = min(marker_zids), max(marker_zids)
                z_middle = int((z_max - z_min) / 2 + z_min)

                # Generiere Punkt-Prompts für den mittleren Slice
                gt_mask_middle = marker_data_id[z_middle]
                slice_label_dict = {name: mask[z_middle] for name, mask in label_dict.items()}
                img_2d = img_3D[z_middle]
                prompts = generate_prompts(
                    gt_mask_middle,
                    slice_label_dict,
                    image_slice=img_2d,
                    prompt_type='point',
                    num_pos_points=args.num_pos_points,
                    num_neg_points=args.num_neg_points,
                    min_dist_from_edge=args.min_dist_from_edge,
                    threshold=50
                )

                if 'points' not in prompts or len(prompts['points']) == 0:
                    logging.warning(f"Keine Prompts für den mittleren Slice {z_middle} generiert. Überspringe.")
                    continue

                logging.info(f"Generierte Punkte: {len(prompts['points'])}")
                logging.info(f"Positive Punkte: {np.sum(prompts['labels'] == 1)}")
                logging.info(f"Negative Punkte: {np.sum(prompts['labels'] == 0)}")

                # Segmentierung des mittleren Slices mit Punkt-Prompts
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1) if len(img_2d.shape) == 2 else img_2d
                H, W, _ = img_3c.shape
                img_1024_tensor = sam2_transforms(img_3c)[None, ...].to(device)
                with torch.no_grad():
                    _features = medsam_model._image_encoder(img_1024_tensor)
                points_1024 = prompts['points'] / np.array([W, H]) * 1024
                labels = prompts['labels']
                img_2d_seg = medsam_point_inference(medsam_model, _features, points_1024, labels, H, W, device)
                segs_3d_temp[z_middle, img_2d_seg > 0] = 1

                # Speichere Debug-Informationen für den mittleren Slice
                if args.debug_mode:
                    debug_prompts_dict[z_middle] = {"points": prompts['points'], "labels": labels}
                    #logging.info(f"Debug prompts für z_middle {z_middle}: {debug_prompts_dict[z_middle]}")

                # Erstelle Bounding Box aus der Segmentierung des mittleren Slices
                mid_slice_bbox_2d = get_bbox(segs_3d_temp[z_middle], bbox_shift=args.bbox_shift) if np.max(
                    segs_3d_temp[z_middle]) > 0 else get_bbox(gt_mask_middle, bbox_shift=args.bbox_shift)

                # Propagation nach oben
                z_max = min(z_max + 1, img_3D.shape[0])
                for z in range(z_middle + 1, z_max):
                    img_2d = img_3D[z]
                    img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1) if len(img_2d.shape) == 2 else img_2d
                    H, W, _ = img_3c.shape
                    img_1024_tensor = sam2_transforms(img_3c)[None, ...].to(device)
                    with torch.no_grad():
                        _features = medsam_model._image_encoder(img_1024_tensor)
                    pre_seg = segs_3d_temp[z - 1]
                    pre_seg1024 = cv2.resize(pre_seg, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    box_1024 = get_bbox(pre_seg1024) if np.max(pre_seg1024) > 0 else mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024
                    used_box = box_1024 / 1024 * np.array([W, H, W, H]) if np.max(pre_seg1024) > 0 else mid_slice_bbox_2d
                    if args.debug_mode:
                        debug_prompts_dict[z] = {"box": used_box}
                    img_2d_seg = medsam_inference(medsam_model, _features, box_1024[None, :], H, W)
                    segs_3d_temp[z, img_2d_seg > 0] = 1

                # Propagation nach unten
                z_min = max(-1, z_min - 1)
                for z in range(z_middle - 1, z_min, -1):
                    img_2d = img_3D[z]
                    img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1) if len(img_2d.shape) == 2 else img_2d
                    H, W, _ = img_3c.shape
                    img_1024_tensor = sam2_transforms(img_3c)[None, ...].to(device)
                    with torch.no_grad():
                        _features = medsam_model._image_encoder(img_1024_tensor)
                    pre_seg = segs_3d_temp[z + 1]
                    pre_seg1024 = cv2.resize(pre_seg, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    box_1024 = get_bbox(pre_seg1024) if np.max(pre_seg1024) > 0 else mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024
                    used_box = box_1024 / 1024 * np.array([W, H, W, H]) if np.max(pre_seg1024) > 0 else mid_slice_bbox_2d
                    if args.debug_mode:
                        debug_prompts_dict[z] = {"box": used_box}
                    img_2d_seg = medsam_inference(medsam_model, _features, box_1024[None, :], H, W)
                    segs_3d_temp[z, img_2d_seg > 0] = 1

            segs_dict[label_id] = segs_3d_temp.copy()

        # Speichere Ergebnisse
        if segs_dict:
            save_npz_and_nii(pred_save_dir, name, segs_dict, label_ids, img_3D, spacing, visualize, save_nii, include_ct)
            if args.debug_mode and debug_prompts_dict:
                debug_dir = join(pred_save_dir, "debug_prompts")
                makedirs(debug_dir, exist_ok=True)
                save_prompt_debug_visualizations(img_3D, label_dict, debug_prompts_dict, debug_dir, name_prefix, segs_dict, middle_slice=z_middle)
        else:
            logging.warning(f"No segmentations found for {name} with requested labels.")

    except Exception as e:
        logging.error(f"Error processing {name}: {e}")
        return None

if __name__ == '__main__':
    mp.set_start_method('spawn')
    with mp.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(main, names), total=len(names)))