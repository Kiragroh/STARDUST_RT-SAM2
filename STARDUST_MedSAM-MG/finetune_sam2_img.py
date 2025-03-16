# -*- coding: utf-8 -*-
"""
finetune sam2 model on medical image data
only finetune the image encoder and mask decoder
freeze the prompt encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from matplotlib.animation import FuncAnimation

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from sam2.build_sam import build_sam2
from typing import List, Optional, Tuple
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
import cv2
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(2024)

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20, one_label_per_epoch=True):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        self._transform = SAM2Transforms(resolution=1024, mask_threshold=0)
        self.current_epoch = 0
        self.current_label = None
        self.processed_labels = set()  # Track which labels we've already processed
        self.one_label_per_epoch = bool(one_label_per_epoch)  # Ensure boolean type
        
        # Get all possible labels and their frequencies
        self.label_to_files = {}  # Dictionary to store which files contain which labels
        for gt_file in self.gt_path_files:
            gt = np.load(gt_file, "r", allow_pickle=True)
            labels = np.unique(gt)[1:]  # Exclude background
            for label in labels:
                if label not in self.label_to_files:
                    self.label_to_files[label] = []
                self.label_to_files[label].append(gt_file)
        
        self.all_possible_labels = sorted(list(self.label_to_files.keys()))
        print(f"Total number of images: {len(self.gt_path_files)}")
        print(f"Available labels and their frequencies:")
        for label in self.all_possible_labels:
            print(f"Label {label}: {len(self.label_to_files[label])} images")
        
        # Initialize with random label
        self.current_label = random.choice(self.all_possible_labels)
        self.current_files = self.label_to_files[self.current_label]
        
        # Print initial mode
        if self.one_label_per_epoch:
            print(f"Mode: One label per epoch. Starting with label {self.current_label}")
        else:
            print("Mode: Random label per step")
            
        print(f"Number of training samples: {len(self.current_files)}")

    def set_epoch(self, epoch):
        """Set current epoch and filter files for chosen label"""
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            if self.one_label_per_epoch:
                # Only choose random label if not already set (allows resuming with same label)
                if self.current_label is None:
                    # Special case for single label
                    if len(self.all_possible_labels) == 1:
                        self.current_label = self.all_possible_labels[0]
                        self.processed_labels.clear()  # Always clear for single label
                    else:
                        # Get available labels (excluding processed ones)
                        available_labels = [l for l in self.all_possible_labels if l not in self.processed_labels]
                        if not available_labels:
                            print("All labels have been processed! Resetting processed labels to start new cycle.")
                            self.processed_labels.clear()  # Reset processed labels
                            available_labels = self.all_possible_labels  # All labels are available again
                        self.current_label = random.choice(available_labels)
                # Update file list to only include files with current label
                self.current_files = self.label_to_files[self.current_label]
                print(f"Epoch {epoch}: Training on label {self.current_label} with {len(self.current_files)} images")
                if len(self.all_possible_labels) > 1:  # Only show processed/remaining for multiple labels
                    print(f"Processed labels so far: {sorted(self.processed_labels)}")
                    print(f"Remaining labels: {sorted(set(self.all_possible_labels) - self.processed_labels)}")
            else:
                # In random label mode, just initialize with a random label for the epoch
                self.current_label = random.choice(self.all_possible_labels)
                self.current_files = self.label_to_files[self.current_label]
                print(f"Epoch {epoch}: Starting with random label mode")

    def __len__(self):
        if self.one_label_per_epoch:
            return len(self.current_files)
        else:
            # In random mode, use a fixed number of samples per epoch
            # This ensures consistent training regardless of label distribution
            return args.batch_size * 200  # 200 steps per epoch with given batch size

    def __getitem__(self, index):
        if not self.one_label_per_epoch:
            # For random label mode, choose a random label and file
            random_label = random.choice(self.all_possible_labels)
            random_file = random.choice(self.label_to_files[random_label])
            img_file = join(self.img_path, os.path.basename(random_file))
            gt_npy = np.load(random_file, "r", allow_pickle=True)
            current_label = random_label  # Store for return
            
            # Get all masks for the chosen label
            label_masks = gt_npy == random_label
            if not np.any(label_masks):
                # Fallback if no masks found (shouldn't happen due to label_to_files)
                print(f"Warning: No masks found for label {random_label} in {random_file}")
                return self.__getitem__(index)  # Try again with same index
        else:
            # Use current label mode
            img_file = join(self.img_path, os.path.basename(self.current_files[index]))
            gt_npy = np.load(self.current_files[index], "r", allow_pickle=True)
            label_masks = gt_npy == self.current_label
            current_label = self.current_label  # Store for return
        
        # load npy image (1024, 1024, 3), [0,1]
        img = np.load(img_file, "r", allow_pickle=True)  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = self._transform(img.copy())
        
        # Use the epoch's chosen label
        gt2D = np.uint8(label_masks)
        assert gt_npy.shape == (256, 256), "ground truth should be 256x256"
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        
        y_indices, x_indices = np.where(gt2D > 0)
        if len(y_indices) > 0:  # If there are valid segmentation points
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])*4 ## scale bbox from 256 to 1024
        else:
            print("No valid segmentation points found, using full image")
            bboxes = np.array([0, 0, gt2D.shape[1]*4, gt2D.shape[0]*4])
        
        return (
            img_1024,  # [3, 1024, 1024]
            torch.tensor(gt2D[None, :, :]).long(),  # [1, 256, 256]
            torch.tensor(bboxes).float(),
            torch.tensor(current_label).long()  # Add label to return tuple
        )


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default=None,
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM2-Tiny-Flare22")
parser.add_argument(
    "-model_cfg", type=str, default="sam2_hiera_t.yaml", help="model config file"
)
parser.add_argument("-pretrain_model_path",
                    type=str,
                    default=None,
)
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=8)
parser.add_argument("-bbox_shift", type=int, default=5)
parser.add_argument("-num_workers", type=int, default=0)
parser.add_argument("-one_label_per_epoch", type=lambda x: x.lower() == 'true', default=True,
                    help="If True, use one label per epoch. If False, use random label per step. Use 'True' or 'False'")
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float,
    default=6e-5,
    metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-resume", type=str,
    default=None,
    help="Resuming training from checkpoint"
)
parser.add_argument("-device", type=str, default="cuda:0")
parser.add_argument(
    "-generate_validation_images",
    type=str,
    default="False",
    help="Generate validation images after each epoch (True/False)",
)
args, unknown = parser.parse_known_args()

# Convert one_label_per_epoch to boolean
args.one_label_per_epoch = args.one_label_per_epoch == "true"
# Convert generate_validation_images to boolean 
args.generate_validation_images = args.generate_validation_images.lower() == "true"

# Setup paths and device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)

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
            image_embeddings=img_embed, # (B, 256, 64, 64)
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
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features


# %%
def setup_logging(model_save_path):
    log_file = os.path.join(model_save_path, 'training.log')
    # Create a file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    # Remove any existing handlers
    logger.handlers = []
    # Add the handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

class MetricsPlotter:
    def __init__(self, save_path, alpha=0.95):
        self.save_path = save_path
        self.alpha = alpha  # EMA smoothing factor
        self.current_epoch = 0
        self.total_steps = 0
        self.steps = []
        self.losses = []
        self.dices = []
        self.ious = []
        
        # EMA values
        self.ema_loss = None
        self.ema_dice = None
        self.ema_iou = None
        
        # Raw values for plotting
        self.raw_losses = []
        self.raw_dices = []
        self.raw_ious = []
        
        # Store epoch boundaries
        self.epoch_boundaries = []
        
        plt.ioff()  # Disable interactive mode
        self.create_new_figure()
        
    def create_new_figure(self):
        """Create a new figure with all necessary plots and settings"""
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 12))
        self.fig.suptitle('Training Metrics')
        
        # Initialize two lines per plot - one for raw, one for smoothed
        self.raw_line1, = self.ax1.plot([], [], 'b-', alpha=0.3, label='Raw Loss')
        self.smooth_line1, = self.ax1.plot([], [], 'b-', label='Smoothed Loss')
        self.raw_line2, = self.ax2.plot([], [], 'g-', alpha=0.3, label='Raw Dice')
        self.smooth_line2, = self.ax2.plot([], [], 'g-', label='Smoothed Dice')
        self.raw_line3, = self.ax3.plot([], [], 'r-', alpha=0.3, label='Raw IoU')
        self.smooth_line3, = self.ax3.plot([], [], 'r-', label='Smoothed IoU')
        
        # Set labels
        self.ax1.set_ylabel('Loss')
        self.ax2.set_ylabel('Dice Score')
        self.ax3.set_ylabel('IoU')
        self.ax3.set_xlabel('Steps')
        
        # Add legends
        self.ax1.legend()
        self.ax2.legend()
        self.ax3.legend()
        
    def new_epoch(self):
        """Called when a new epoch starts"""
        if len(self.steps) > 0:
            self.epoch_boundaries.append(self.total_steps)
        self.current_epoch += 1
        
    def _update_ema(self, current_value, ema_value):
        if ema_value is None:
            return current_value
        return self.alpha * ema_value + (1 - self.alpha) * current_value
        
    def update(self, step, loss, dice, iou):
        self.total_steps += 1
        self.steps.append(self.total_steps)
        
        # Store raw values
        self.raw_losses.append(loss)
        self.raw_dices.append(dice)
        self.raw_ious.append(iou)
        
        # Update EMAs
        if self.ema_loss is None:
            self.ema_loss = loss
            self.ema_dice = dice
            self.ema_iou = iou
        else:
            self.ema_loss = self._update_ema(loss, self.ema_loss)
            self.ema_dice = self._update_ema(dice, self.ema_dice)
            self.ema_iou = self._update_ema(iou, self.ema_iou)
        
        self.losses.append(self.ema_loss)
        self.dices.append(self.ema_dice)
        self.ious.append(self.ema_iou)
        
        # Plot all data
        plt.close(self.fig)  # Close previous figure
        self.create_new_figure()
        
        # Plot metrics
        self.raw_line1.set_data(self.steps, self.raw_losses)
        self.smooth_line1.set_data(self.steps, self.losses)
        self.raw_line2.set_data(self.steps, self.raw_dices)
        self.smooth_line2.set_data(self.steps, self.dices)
        self.raw_line3.set_data(self.steps, self.raw_ious)
        self.smooth_line3.set_data(self.steps, self.ious)
        
        # Add vertical lines for epoch boundaries
        for boundary in self.epoch_boundaries:
            self.ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            self.ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            self.ax3.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        # Adjust limits
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view()
        
        # Save the figure
        plt.savefig(os.path.join(self.save_path, 'training_metrics.png'))
    
def generate_validation_images(model, dataset, device, save_path, epoch, num_samples=3):
    """
    Generate validation images for a few samples to check segmentation quality.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    dataset : Dataset
        The dataset to sample from
    device : str
        Device to run inference on ('cuda:0' or 'cpu')
    save_path : str
        Directory to save images
    epoch : int
        Current epoch number
    num_samples : int
        Number of validation samples to generate
    """
    # Create validation directory if it doesn't exist
    val_dir = os.path.join(save_path, "validation_images")
    os.makedirs(val_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Prepare figure for saving images
    plt.figure(figsize=(15, 5 * num_samples))
    
    # Get random indices from dataset
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            try:
                # Get sample from dataset
                image, gt2D, boxes, current_label = dataset[idx]
                
                # Process input for model
                image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
                
                # Format boxes correctly for the model
                # Make sure boxes are in the format expected by the model: (B, 4)
                boxes_np = boxes.detach().cpu().numpy().reshape(1, 4)  # Reshape to (1, 4)
                
                # Get prediction
                pred_logits = model(image_tensor, boxes_np)
                pred_mask = (torch.sigmoid(pred_logits) > 0.5).float().cpu().squeeze()
                
                # Convert tensors to numpy for visualization
                image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
                gt_np = gt2D.cpu().numpy()
                pred_np = pred_mask.cpu().numpy()
                
                # Plot the results
                plt.subplot(num_samples, 3, i*3 + 1)
                plt.imshow(image_np)
                plt.title(f"Image (Label {current_label.item()})")
                plt.axis('off')
                
                plt.subplot(num_samples, 3, i*3 + 2)
                plt.imshow(image_np)
                show_mask(gt_np, plt.gca(), random_color=False)
                plt.title("Ground Truth")
                plt.axis('off')
                
                plt.subplot(num_samples, 3, i*3 + 3)
                plt.imshow(image_np)
                show_mask(pred_np, plt.gca(), random_color=False)
                plt.title("Prediction")
                plt.axis('off')
            except Exception as e:
                logging.error(f"Error generating validation image for sample {idx}: {str(e)}")
                # Continue with next sample if there's an error
                continue
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(val_dir, f"validation_epoch_{epoch}.png"))
    plt.close()
    
    # Set model back to training mode
    model.train()
    
    logging.info(f"Saved validation images for epoch {epoch}")

def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        args.model_cfg,
        join(model_save_path, "sam2_model_cfg.yaml"),
    )

    # set up logger
    logger = setup_logging(model_save_path)
    
    # set up model
    device = args.device
    sam2_model = build_sam2(args.model_cfg, args.pretrain_model_path, device=device)
    medsam_model = MedSAM2(model=sam2_model).to(device)
    medsam_model.train()

    # set up optimizer
    optimizer = optim.AdamW(
        medsam_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # set up learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,  # Total number of epochs
        eta_min=args.lr * 0.01  # Minimum learning rate (1% of initial lr)
    )

    # set up training dataloader
    train_dataset = NpyDataset(
        data_root=args.tr_npy_path,  # Using tr_npy_path instead of data_path
        bbox_shift=args.bbox_shift,
        one_label_per_epoch=args.one_label_per_epoch  # Ein Label pro Epoch (True) oder zufälliges Label pro Step (False)
    )
    
    # Print dataset mode
    print(f"Dataset mode: {'One label per epoch' if args.one_label_per_epoch else 'Random label per step'}")
    
    # Initialize first epoch
    train_dataset.set_epoch(0)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches
    )

    print("Number of training samples: ", len(train_dataset))
    
    # set up metrics
    plotter = MetricsPlotter(model_save_path)
    
    num_epochs = args.num_epochs
    
    # Initialize best metrics
    best_loss = 1e10
    start_epoch = 0
    epoch_loss = 0
    epoch_dice = 0
    epoch_iou = 0

    # Load checkpoint if resuming
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        medsam_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        
        if train_dataset.one_label_per_epoch:
            # Only restore label history in one_label_per_epoch mode
            train_dataset.current_label = checkpoint.get('current_label', None)
            train_dataset.processed_labels = set(checkpoint.get('processed_labels', []))
            logger.info(f"Resuming from epoch {start_epoch} with label {train_dataset.current_label} and best loss {best_loss:.4f}")
            logger.info(f"Already processed labels: {sorted(train_dataset.processed_labels)}")
        else:
            # In random label mode, ignore saved label history and force new epoch setup
            logger.info(f"Resuming from epoch {start_epoch} with random label mode and best loss {best_loss:.4f}")
            train_dataset.current_label = None
            train_dataset.processed_labels.clear()
            train_dataset.set_epoch(start_epoch)  # Force new epoch setup

    for epoch in range(start_epoch, num_epochs):
        # Set epoch for consistent labels and recreate dataloader
        train_dataset.set_epoch(epoch)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,  # Drop incomplete batches
        )
        
        # Skip epoch if no images contain the selected label
        if len(train_dataset.current_files) == 0:
            print(f"Skipping epoch {epoch} as no images contain the selected label")
            continue
            
        for step, (image, gt2D, boxes, current_label) in enumerate(tqdm(train_dataloader)):
            # Log batch labels at the start of each step
            batch_labels = current_label.tolist()
            print(f"\nBatch {step}: Using labels {batch_labels}")
            
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            
            medsam_pred = medsam_model(image, boxes_np)
            dice_loss = monai.losses.DiceLoss(sigmoid=True)(medsam_pred, gt2D)
            ce = torch.nn.BCEWithLogitsLoss()(medsam_pred, gt2D.float())
            loss = dice_loss + ce
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Move tensors to CPU for metric calculation
            with torch.no_grad():
                pred_masks = (torch.sigmoid(medsam_pred) > 0.5).float().cpu()
                gt2D_cpu = gt2D.cpu()
                dice_score = 1 - dice_loss.item()
                iou = (pred_masks * gt2D_cpu).sum() / ((pred_masks + gt2D_cpu) > 0).sum()
                current_loss = loss.item()
            
            # Update metrics after each step
            plotter.update(step, current_loss, dice_score, iou.item())
            
            # Update running averages
            epoch_loss += current_loss
            epoch_dice += dice_score
            epoch_iou += iou.item()
            
            # Calculate and log running averages
            avg_loss = epoch_loss / (step + 1)
            avg_dice = epoch_dice / (step + 1)
            avg_iou = epoch_iou / (step + 1)
            
            # Log metrics and current learning rate
            log_msg = f'Epoch {epoch}, Step {step}, Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}'
            logger.info(log_msg)
            for handler in logger.handlers:
                handler.flush()

        # Only update scheduler and calculate averages if we had data
        if step > 0:
            scheduler.step()
            plotter.new_epoch()  # Signal new epoch to plotter
            
            # Calculate epoch averages
            epoch_loss /= step + 1
            epoch_dice /= step + 1
            epoch_iou /= step + 1
            
            log_msg = f'Epoch {epoch} Complete - Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}, IoU: {epoch_iou:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}'
            logger.info(log_msg)
            for handler in logger.handlers:
                handler.flush()
                
            # Save checkpoint at the end of each epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': medsam_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'dice': epoch_dice,
                'iou': epoch_iou,
                'current_label': train_dataset.current_label,
                'processed_labels': list(train_dataset.processed_labels)
            }
            
            # Save last checkpoint
            last_checkpoint_path = os.path.join(model_save_path, 'last_checkpoint.pth')
            torch.save(checkpoint, last_checkpoint_path)
            logger.info(f"Saved last checkpoint to {last_checkpoint_path}")
            
            # Save best model if loss improved
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_checkpoint_path = os.path.join(model_save_path, 'best_checkpoint.pth')
                torch.save(checkpoint, best_checkpoint_path)
                logger.info(f"New best model! Saved checkpoint to {best_checkpoint_path}")
                
            # Generate validation images if enabled
            if args.generate_validation_images:
                logger.info("Generating validation images...")
                generate_validation_images(
                    model=medsam_model, 
                    dataset=train_dataset, 
                    device=device, 
                    save_path=model_save_path, 
                    epoch=epoch,
                    num_samples=3
                )
                
            # Add current label to processed set
            train_dataset.processed_labels.add(train_dataset.current_label)
                
            # Reset metrics for next epoch
            epoch_loss = 0
            epoch_dice = 0
            epoch_iou = 0
            train_dataset.current_label = None  # Reset current label so a new one will be chosen
        
        # %% plot loss
        plt.plot([epoch_loss])
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()

if __name__ == "__main__":
    main()