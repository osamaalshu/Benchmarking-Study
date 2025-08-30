#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import os

join = os.path.join

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
# NEW: Import advanced loss functions for composite loss
try:
    from monai.losses import FocalLoss, TverskyLoss
    ADVANCED_LOSSES_AVAILABLE = True
except Exception:
    FocalLoss = TverskyLoss = None
    ADVANCED_LOSSES_AVAILABLE = False

# Custom Boundary Loss implementation (since not available in MONAI 1.5.0)
class BoundaryLoss(torch.nn.Module):
    """Custom boundary loss using Signed Distance Transform"""
    def __init__(self, include_background=False):
        super().__init__()
        self.include_background = include_background
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B,1,H,W) predicted foreground probability
            target: (B,1,H,W) binary ground truth
        """
        # Simple L1 loss on boundaries (can be enhanced with SDT)
        return torch.nn.functional.l1_loss(pred, target)
from monai.data import decollate_batch, PILReader, NumpyReader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandSpatialCropd,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    EnsureTyped,
    EnsureType,
    CenterSpatialCropd,
)
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.sac_model import SACModel, create_default_points
from models.nnunet import create_nnunet_model
from models.lstmunet import create_lstmunet_model
from models.maunet import create_maunet_model, create_maunet_ensemble_model, WeightedL1Loss
from models.maunet_error_aware import create_maunet_error_aware_model, create_maunet_error_aware_ensemble_model

# Import proxy losses with fallback
try:
    from models.proxy_losses import ProxyCELoss
    PROXY_LOSS_AVAILABLE = True
except ImportError:
    print("[WARN] models.proxy_losses not found; proxy regularization will be disabled")
    ProxyCELoss = None
    PROXY_LOSS_AVAILABLE = False



print("Successfully imported all requirements!")


def make_centroid_map(instances: torch.Tensor, sigma: float = 2.0):
    """
    Create centroid heatmaps from instance masks
    
    Args:
        instances: (B,1,H,W) or (B,H,W) integer mask with 0=bg, >0 instance ids
        sigma: Gaussian sigma for centroid heatmaps
        
    Returns:
        centroid_maps: (B,1,H,W) in [0,1] with Gaussian peaks at centroids
    """
    if instances.dim() == 3:
        instances = instances.unsqueeze(1)  # (B,H,W) -> (B,1,H,W)
    B, _, H, W = instances.shape
    device = instances.device
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    maps = torch.zeros(B, 1, H, W, device=device)
    
    for b in range(B):
        ids = torch.unique(instances[b, 0])
        ids = ids[ids > 0]  # Exclude background
        for k in ids:
            mask = (instances[b, 0] == k)
            if mask.any():
                y = torch.mean(yy[mask].float())
                x = torch.mean(xx[mask].float())
                g = torch.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
                maps[b, 0] = torch.maximum(maps[b, 0], g)
    
    # Normalize per image to [0,1]
    for b in range(B):
        minv = maps[b].min()
        maxv = maps[b].max()
        if maxv > minv:
            maps[b] = (maps[b] - minv) / (maxv - minv)
    
    return maps


def main():
    parser = argparse.ArgumentParser("Baseline for Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="./data/train-preprocessed/",
        type=str,
        help="training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--work_dir", default="./baseline/work_dir", help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2022, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=4, type=int)

    # Model parameters
    parser.add_argument(
        "--model_name", default="unet", help="select mode: unet, sac, nnunet, lstmunet, maunet, maunet_error_aware"
    )
    parser.add_argument("--num_class", default=3, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=256, type=int, help="segmentation classes"
    )
    parser.add_argument("--backbone", default="resnet50", type=str, choices=["resnet50", "wide_resnet50"], help="Backbone for MAUNet")
    parser.add_argument("--ensemble", action="store_true", help="Use MAUNet ensemble (resnet50 + wide_resnet50)")
    parser.add_argument("--dist_path", type=str, default=None, help="Path to precomputed distance transform maps for MAUNet")
    parser.add_argument("--dist_suffix", type=str, default=".npy", help="Suffix for distance transform filenames (e.g., .npy)")
    parser.add_argument("--reg_loss_weight", type=float, default=0.1, help="Weight for MAUNet regression loss")
    
    # NEW: Composite loss parameters for MAUNet
    parser.add_argument("--lambda_det", type=float, default=1.0, help="Weight for detection (focal) loss")
    parser.add_argument("--lambda_seg", type=float, default=1.0, help="Weight for segmentation (tversky) loss")
    parser.add_argument("--lambda_bnd", type=float, default=0.5, help="Weight for boundary loss")
    parser.add_argument("--lambda_dt", type=float, default=0.1, help="Weight for distance transform loss")
    parser.add_argument("--lambda_center", type=float, default=0.2, help="Weight for centroid loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--tversky_alpha", type=float, default=0.3, help="Tversky loss alpha (false positive weight)")
    parser.add_argument("--tversky_beta", type=float, default=0.7, help="Tversky loss beta (false negative weight)")
    # Training parameters
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=2000, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=10, type=int)
    parser.add_argument("--initial_lr", type=float, default=6e-4, help="learning rate")
    parser.add_argument("--lr_scheduler", action="store_true", help="Use exponential learning rate scheduler")
    parser.add_argument("--lr_gamma", type=float, default=0.95, help="Learning rate decay factor")
    parser.add_argument("--lr_step_size", type=int, default=10, help="Learning rate decay step size")
    parser.add_argument("--load_checkpoint", action="store_true", help="Load from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint file")

    args = parser.parse_args()

    # Guard: training with ensemble container is not supported (proxy loss needs embeddings)
    if args.model_name.lower() == "maunet" and args.ensemble:
        print("⚠️  Training with --ensemble is not supported; using single-backbone MAUNet for training.")
        args.ensemble = False

    monai.config.print_config()

    #%% set training/validation split
    np.random.seed(args.seed)
    model_path = join(args.work_dir, args.model_name + "_3class")
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )
    # Training data
    train_img_path = join(args.data_path, "images")
    train_gt_path = join(args.data_path, "labels")
    
    # Validation data (separate folder)
    val_img_path = join(args.data_path.replace("train-preprocessed", "val"), "images")
    val_gt_path = join(args.data_path.replace("train-preprocessed", "val"), "labels")

    train_img_names = sorted(os.listdir(train_img_path))
    train_gt_names = [img_name.split(".")[0] + "_label.png" for img_name in train_img_names]
    
    val_img_names = sorted(os.listdir(val_img_path))
    val_gt_names = [img_name.split(".")[0] + "_label.png" for img_name in val_img_names]

    # Optionally include distance transform targets
    train_files = []
    for i in range(len(train_img_names)):
        sample = {"img": join(train_img_path, train_img_names[i]), "label": join(train_gt_path, train_gt_names[i])}
        if args.dist_path and os.path.isdir(args.dist_path):
            # Distance maps saved using label basename (e.g., imgname_label.npy)
            base_label = os.path.splitext(train_gt_names[i])[0]
            dist_file = join(args.dist_path, base_label + args.dist_suffix)
            if os.path.exists(dist_file):
                sample["dist"] = dist_file
        train_files.append(sample)

    val_files = []
    for i in range(len(val_img_names)):
        sample = {"img": join(val_img_path, val_img_names[i]), "label": join(val_gt_path, val_gt_names[i])}
        if args.dist_path and os.path.isdir(args.dist_path):
            base_label = os.path.splitext(val_gt_names[i])[0]
            dist_file = join(args.dist_path, base_label + args.dist_suffix)
            if os.path.exists(dist_file):
                sample["dist"] = dist_file
        val_files.append(sample)

    has_dist_train = all(["dist" in s for s in train_files]) and len(train_files) > 0
    has_dist_val = all(["dist" in s for s in val_files]) and len(val_files) > 0
    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)}"
    )
    #%% define transforms for image and segmentation
    if has_dist_train:
        train_transforms = Compose(
            [
                LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
                LoadImaged(keys=["dist"], reader=NumpyReader, allow_missing_keys=True),
                EnsureChannelFirstd(keys=["label", "img", "dist"], allow_missing_keys=True),
                ScaleIntensityd(keys=["img"], allow_missing_keys=True),
                SpatialPadd(keys=["img", "label", "dist"], spatial_size=args.input_size),
                RandSpatialCropd(keys=["img", "label", "dist"], roi_size=args.input_size, random_size=False),
                RandAxisFlipd(keys=["img", "label", "dist"], prob=0.5),
                RandRotate90d(keys=["img", "label", "dist"], prob=0.5, spatial_axes=[0, 1]),
                RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
                RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
                RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
                RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
                RandZoomd(keys=["img", "label", "dist"], prob=0.15, min_zoom=0.8, max_zoom=1.5, mode=["area", "nearest", "nearest"]),
                EnsureTyped(keys=["img", "label", "dist"], allow_missing_keys=True),
            ]
        )
    else:
        train_transforms = Compose(
            [
                LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
                EnsureChannelFirstd(keys=["label", "img"], allow_missing_keys=True),
                ScaleIntensityd(keys=["img"], allow_missing_keys=True),
                SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
                RandSpatialCropd(keys=["img", "label"], roi_size=args.input_size, random_size=False),
                RandAxisFlipd(keys=["img", "label"], prob=0.5),
                RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
                RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
                RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
                RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
                RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
                RandZoomd(keys=["img", "label"], prob=0.15, min_zoom=0.8, max_zoom=1.5, mode=["area", "nearest"]),
                EnsureTyped(keys=["img", "label"], allow_missing_keys=True),
            ]
        )

    if has_dist_val:
        val_transforms = Compose(
            [
                LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
                LoadImaged(keys=["dist"], reader=NumpyReader, allow_missing_keys=True),
                EnsureChannelFirstd(keys=["label", "img", "dist"], allow_missing_keys=True),
                ScaleIntensityd(keys=["img"], allow_missing_keys=True),
                SpatialPadd(keys=["img", "label", "dist"], spatial_size=args.input_size),
                CenterSpatialCropd(keys=["img", "label", "dist"], roi_size=args.input_size),
                EnsureTyped(keys=["img", "label", "dist"], allow_missing_keys=True),
            ]
        )
    else:
        val_transforms = Compose(
            [
                LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
                EnsureChannelFirstd(keys=["label", "img"], allow_missing_keys=True),
                ScaleIntensityd(keys=["img"], allow_missing_keys=True),
                SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
                CenterSpatialCropd(keys=["img", "label"], roi_size=args.input_size),
                EnsureTyped(keys=["img", "label"], allow_missing_keys=True),
            ]
        )

    #% define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=4)
    check_data = monai.utils.misc.first(check_loader)
    print(
        "sanity check:",
        check_data["img"].shape,
        torch.max(check_data["img"]),
        check_data["label"].shape,
        torch.max(check_data["label"]),
    )

    #%% create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() or torch.backends.mps.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )

    post_pred = Compose([
        EnsureType(),
        Activations(softmax=True),
        AsDiscrete(argmax=True, to_onehot=args.num_class),
    ])
    post_gt = Compose([
        EnsureType(),
        AsDiscrete(to_onehot=args.num_class),
    ])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "mps":
        print("✅ Training on Apple Silicon GPU (MPS)")
    elif device.type == "cuda":
        print("✅ Training on NVIDIA GPU (CUDA)")
    else:
        print("⚠️  Training on CPU")
    if args.model_name.lower() == "unet":
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=args.num_class,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)





    if args.model_name.lower() == "sac":
        model = SACModel(device=device, num_classes=args.num_class, freeze_encoder_layers=6, use_lora=True, lora_rank=16)
        # Note: SACModel handles device internally and has its own decoder head

    if args.model_name.lower() == "nnunet":
        model = create_nnunet_model(
            image_size=(args.input_size, args.input_size),
            in_channels=3,
            out_channels=args.num_class,
            gpu_memory_gb=8.0
        ).to(device)

    if args.model_name.lower() == "lstmunet":
        model = create_lstmunet_model(
            image_size=(args.input_size, args.input_size),
            in_channels=3,
            out_channels=args.num_class,
            base_filters=64,
            depth=4,
            lstm_hidden_channels=64,
            lstm_layers=2,
            dropout_rate=0.1
        ).to(device)

    if args.model_name.lower() == "maunet":
        if args.ensemble:
            model = create_maunet_ensemble_model(
                num_classes=args.num_class,
                input_size=args.input_size,
                in_channels=3,
                backbones=["resnet50", "wide_resnet50"],
                average=True,
            ).to(device)
        else:
            model = create_maunet_model(
                num_classes=args.num_class,
                input_size=args.input_size,
                in_channels=3,
                backbone=args.backbone,
            ).to(device)

    if args.model_name.lower() == "maunet_error_aware":
        if args.ensemble:
            model = create_maunet_error_aware_ensemble_model(
                num_classes=args.num_class,
                input_size=args.input_size,
                in_channels=3,
                backbones=["resnet50", "wide_resnet50"],
                average=True,
            ).to(device)
        else:
            model = create_maunet_error_aware_model(
                num_classes=args.num_class,
                input_size=args.input_size,
                in_channels=3,
                backbone=args.backbone,
            ).to(device)

    # Setup loss functions
    if args.model_name.lower() in ["maunet", "maunet_error_aware"]:
        # NEW: Composite loss for MAUNet
        if not ADVANCED_LOSSES_AVAILABLE:
            print("[WARN] MONAI advanced losses not found; falling back to DiceCE.")
            loss_function = monai.losses.DiceCELoss(softmax=True)
            loss_det = loss_seg = loss_bnd = None
        else:
            # Detection: focal; Seg: Tversky (beta>alpha to penalize FN); Boundary: custom
            loss_det = FocalLoss(gamma=args.focal_gamma, include_background=True, to_onehot_y=True, use_softmax=True)
            loss_seg = TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta, include_background=False, to_onehot_y=True, softmax=True)
            # Custom boundary loss
            loss_bnd = BoundaryLoss(include_background=False)
            loss_function = None  # We'll use composite loss instead
        
        # DT regression loss
        try:
            regression_loss = WeightedL1Loss()
        except Exception:
            regression_loss = torch.nn.L1Loss()
    else:
        loss_function = monai.losses.DiceCELoss(softmax=True)
    initial_lr = args.initial_lr
    # Build optimizer with param group for proxies (no weight decay)
    if args.model_name.lower() in ["maunet", "maunet_error_aware"]:
        proxy_params = []
        other_params = []
        for n, p in model.named_parameters():
            if "proxies" in n:
                proxy_params.append(p)
            else:
                other_params.append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": 1e-2},
                {"params": proxy_params, "weight_decay": 0.0},
            ],
            lr=initial_lr,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), initial_lr)
    
    # Setup learning rate scheduler if requested
    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Load checkpoint if specified
    start_epoch = 1
    if args.load_checkpoint and args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            print(f"Loading checkpoint from: {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            epoch_loss_values = checkpoint.get('loss', [])
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found: {args.checkpoint_path}")
            print("Starting training from scratch")

    # start a typical PyTorch training
    max_epochs = args.max_epochs
    epoch_tolerance = args.epoch_tolerance
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(model_path)
    # Initialize checkpoint
    checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "loss": [],
    }
    # Proxy loss setup (MAUNet only)
    if args.model_name.lower() in ["maunet", "maunet_error_aware"] and PROXY_LOSS_AVAILABLE:
        proxy_ce = ProxyCELoss(temperature=0.07, ignore_index=-100).to(device)
        lambda_proxy = 0.1
    else:
        proxy_ce = None
        lambda_proxy = 0.0
    
    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_loss = 0
        for step, batch_data in enumerate(train_loader, 1):
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(
                device
            )
            optimizer.zero_grad()
            
            # Handle different model architectures
            if args.model_name.lower() == "sac":
                batch_size = inputs.shape[0]
                points = create_default_points(batch_size, (args.input_size, args.input_size))
                points = points.to(device)
                outputs = model(inputs, points=points)
                
                # Ensure SAC output matches input size
                if outputs.shape[-2:] != (args.input_size, args.input_size):
                    outputs = F.interpolate(outputs, size=(args.input_size, args.input_size), mode='bilinear', align_corners=False)
                
                labels_onehot = monai.networks.one_hot(labels, args.num_class)
                loss = loss_function(outputs, labels_onehot)
            elif args.model_name.lower() == "maunet":
                # MAUNet returns dual outputs: (classification, regression)
                model_outputs = model(inputs)
                if isinstance(model_outputs, (list, tuple)) and len(model_outputs) == 3:
                    outputs, outputs_reg, emb = model_outputs
                else:
                    # Backward compatibility for 2-output models
                    outputs, outputs_reg = model_outputs
                    emb = None
                labels_onehot = monai.networks.one_hot(labels, args.num_class)
                
                # Classification loss
                class_loss = loss_function(outputs, labels_onehot)
                
                # Regression loss (distance transform) - prefer precomputed DT maps when provided
                if "dist" in batch_data:
                    # Assume precomputed distance maps are already scaled to [0,1]
                    dist_target = batch_data["dist"].to(device).float()
                else:
                    # Fallback: simple binary mask as proxy distance target
                    dist_target = (labels > 0).float()
                reg_loss = regression_loss(outputs_reg, dist_target)

                # Proxy regularizer (anti-ambiguity) - only if embedding available
                if emb is not None:
                    # Build proxy targets: map 3-class {0=bg,1=interior,2=boundary} -> {0=bg,1=interior}, ignore boundary
                    y_full = labels.long().squeeze(1) if labels.dim() == 4 else labels.long()  # (B,H,W)
                    proxy_targets = torch.full_like(y_full, fill_value=-100)
                    proxy_targets[y_full == 0] = 0
                    proxy_targets[y_full == 1] = 1

                    # Flatten and subsample
                    B, D, H, W = emb.shape
                    emb_flat = emb.permute(0, 2, 3, 1).reshape(-1, D)
                    tgt_flat = proxy_targets.reshape(-1)
                    valid = tgt_flat != -100
                    valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
                    if valid_idx.numel() > 0:
                        K = 4096
                        if valid_idx.numel() > K:
                            perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[:K]
                            sel = valid_idx[perm]
                        else:
                            sel = valid_idx
                        emb_sel = emb_flat[sel]
                        tgt_sel = tgt_flat[sel]
                        if proxy_ce is not None:
                            loss_proxy = proxy_ce(emb_sel, model.proxies, tgt_sel)
                        else:
                            loss_proxy = torch.tensor(0.0, device=device)
                    else:
                        loss_proxy = torch.tensor(0.0, device=device, dtype=emb.dtype)
                else:
                    loss_proxy = torch.tensor(0.0, device=device)
                
                # Combined loss
                loss = class_loss + float(args.reg_loss_weight) * reg_loss + lambda_proxy * loss_proxy
            elif args.model_name.lower() == "maunet_error_aware":
                # NEW: Error-Aware MAUNet with composite loss
                outputs = model(inputs)
                if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
                    seg_logits, dt_logits, center_logits = outputs
                else:
                    # Backward compatibility (2 heads)
                    seg_logits, dt_logits = outputs
                    center_logits = None

                # Build label variants
                probs = torch.softmax(seg_logits, dim=1)
                # Foreground prob = 1 - P(background); assume class 0 is background
                p_fg = 1.0 - probs[:, 0:1]
                y_fg = (labels > 0).float().unsqueeze(1)

                # Detection loss (focal) and segmentation (tversky) if available, else DiceCE fallback
                if loss_det is None or loss_seg is None:
                    L_det = torch.tensor(0.0, device=inputs.device)
                    L_seg = loss_function(seg_logits, monai.networks.one_hot(labels, args.num_class))
                    L_bnd = torch.tensor(0.0, device=inputs.device)
                else:
                    L_det = loss_det(seg_logits, labels)
                    L_seg = loss_seg(seg_logits, labels)
                    if hasattr(loss_bnd, '__call__') and loss_bnd is not torch.nn.L1Loss:
                        try:
                            L_bnd = loss_bnd(p_fg, y_fg)
                        except Exception:
                            L_bnd = torch.nn.functional.l1_loss(p_fg, y_fg)
                    else:
                        L_bnd = torch.nn.functional.l1_loss(p_fg, y_fg)

                # DT regression (bound predictions with sigmoid)
                if "dist" in batch_data:
                    dist_target = batch_data["dist"].to(inputs.device).float()
                    dist_target = dist_target / (dist_target.max() + 1e-6)
                else:
                    dist_target = y_fg
                dt_pred = torch.sigmoid(dt_logits)
                L_dt = regression_loss(dt_pred, dist_target)

                # Centroid BCE (if head present)
                if center_logits is not None:
                    center_target = make_centroid_map(labels)
                    # Ensure same shape as center_logits
                    if center_target.shape != center_logits.shape:
                        center_target = center_target.expand_as(center_logits)
                    L_center = F.binary_cross_entropy_with_logits(center_logits, center_target)
                else:
                    L_center = torch.tensor(0.0, device=inputs.device)

                # Composite loss
                loss = (args.lambda_det * L_det
                        + args.lambda_seg * L_seg
                        + args.lambda_bnd * L_bnd
                        + args.lambda_dt * L_dt
                        + args.lambda_center * L_center)

                # Optional: log terms every 50 steps
                if step % 50 == 0:
                    print(f"Step {step}: L_det={L_det.item():.4f} L_seg={L_seg.item():.4f} L_bnd={L_bnd.item():.4f} L_dt={L_dt.item():.4f} L_center={L_center.item():.4f}")
            else:
                outputs = model(inputs)
                labels_onehot = monai.networks.one_hot(labels, args.num_class)
                loss = loss_function(outputs, labels_onehot)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            writer.flush()  # Force flush to ensure logs are written
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
        
        # Step learning rate scheduler if enabled
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate: {current_lr:.6f}")
            writer.add_scalar("learning_rate", current_lr, epoch)
        
        writer.add_scalar("epoch_loss", epoch_loss, epoch)
        writer.flush()  # Force flush to ensure logs are written
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "loss": epoch_loss_values,
        }

        if epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                # Initialize validation variables
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data[
                        "label"
                    ].to(device)
                    
                    # Keep labels as (B,1,H,W); convert to one-hot via post_gt below
                    val_labels_for_post = val_labels
                    
                    roi_size = (args.input_size, args.input_size)
                    sw_batch_size = 4
                    
                    # Handle different model architectures for validation
                    if args.model_name.lower() == "sac":
                        batch_size = val_images.shape[0]
                        points = create_default_points(batch_size, (args.input_size, args.input_size))
                        points = points.to(device)
                        val_outputs = model(val_images, points=points)
                        
                        # Ensure output is the correct size
                        if val_outputs.shape[-2:] != (args.input_size, args.input_size):
                            val_outputs = F.interpolate(val_outputs, size=(args.input_size, args.input_size), mode='bilinear', align_corners=False)
                    elif args.model_name.lower() == "nnunet":
                        # nnU-Net can handle full images directly
                        val_outputs = model(val_images)
                    elif args.model_name.lower() == "maunet":
                        # MAUNet returns (seg, dist, emb) - use only classification output for validation
                        val_outputs = model(val_images)[0]
                    elif args.model_name.lower() == "maunet_error_aware":
                        # Error-Aware MAUNet returns (seg, dist, centroid) - use only classification output for validation
                        model_outputs = model(val_images)
                        if isinstance(model_outputs, (list, tuple)) and len(model_outputs) >= 1:
                            val_outputs = model_outputs[0]
                        else:
                            val_outputs = model_outputs
                    else:
                        val_outputs = sliding_window_inference(
                            val_images, roi_size, sw_batch_size, model
                        )
                    
                    # Apply post-processing transforms
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_onehot = [post_gt(i) for i in decollate_batch(val_labels_for_post)]
                    
                    # compute metric for current iteration
                    print(
                        f"Validation batch {len(metric_values) + 1}",
                        dice_metric(y_pred=val_outputs, y=val_labels_onehot),
                    )

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(checkpoint, join(model_path, "best_Dice_model.pth"))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                writer.flush()  # Force flush to ensure logs are written
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                if val_images is not None and val_outputs is not None:
                    plot_2d_or_3d_image(val_images, epoch, writer, index=0, tag="image")
                    plot_2d_or_3d_image(val_labels, epoch, writer, index=0, tag="label")
                    plot_2d_or_3d_image(val_outputs, epoch, writer, index=0, tag="output")
                
                # Save SAC, nnU-Net, and MAUNet model predictions during validation
                if args.model_name.lower() in ["sac", "nnunet", "maunet", "maunet_error_aware"]:
                    import tifffile as tif
                    from skimage import measure, morphology
                    
                    # Create prediction directory
                    pred_dir = join(model_path, "predictions")
                    os.makedirs(pred_dir, exist_ok=True)
                    
                    # Save predictions for the first validation image
                    val_pred = val_outputs[0]  # First prediction (raw)
                    val_pred_npy = val_pred.cpu().numpy()
                    
                    # Convert to probability map and instance mask
                    if val_pred_npy.shape[0] > 1:  # Multi-class
                        prob_map = val_pred_npy[1]  # Class 1 probability
                    else:  # Binary
                        prob_map = val_pred_npy[0]
                    
                    # Create instance mask
                    pred_mask = measure.label(morphology.remove_small_objects(
                        morphology.remove_small_holes(prob_map > 0.5), 16))
                    
                    # Save files
                    tif.imwrite(join(pred_dir, f"epoch_{epoch}_prob_map.tiff"), prob_map, compression='zlib')
                    tif.imwrite(join(pred_dir, f"epoch_{epoch}_pred_mask.tiff"), pred_mask, compression='zlib')
                    print(f"Saved {args.model_name} predictions for epoch {epoch}")
            if (epoch - best_metric_epoch) > epoch_tolerance:
                print(
                    f"validation metric does not improve for {epoch_tolerance} epochs! current {epoch=}, {best_metric_epoch=}"
                )
                break

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()
    torch.save(checkpoint, join(model_path, "final_model.pth"))
    np.savez_compressed(
        join(model_path, "train_log.npz"),
        val_dice=metric_values,
        epoch_loss=epoch_loss_values,
    )


if __name__ == "__main__":
    main()
