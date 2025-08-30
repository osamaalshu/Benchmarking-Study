#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for Error-Aware MAUNet with Multi-Objective Loss Function

This script trains the enhanced MAUNet model with:
1. Multi-objective composite loss function
2. Auxiliary task supervision (centroid heatmaps)
3. Dual backbone training (ResNet50 and Wide-ResNet50)
4. Enhanced error-aware architecture

The training pipeline addresses systematic error patterns identified in comprehensive
error analysis, focusing on reducing false negatives and improving instance detection
in crowded cellular regions.
"""

import argparse
import os
import sys

join = os.path.join

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
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

# Import our enhanced models and losses
from error_aware_maunet import (
    create_error_aware_maunet_model, 
    create_error_aware_maunet_ensemble_model
)
from composite_losses import CompositeLoss

# Import proxy losses for embedding regularization
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.proxy_losses import ProxyCELoss

print("Successfully imported all requirements!")


def create_instance_masks_from_labels(labels):
    """
    Convert semantic segmentation labels to instance masks for centroid supervision
    
    Args:
        labels: (B, 1, H, W) semantic segmentation labels
        
    Returns:
        instance_masks: (B, H, W) instance masks with connected components
    """
    from skimage import measure
    
    batch_size = labels.shape[0]
    height, width = labels.shape[2], labels.shape[3]
    device = labels.device
    
    instance_masks = torch.zeros((batch_size, height, width), dtype=torch.long, device=device)
    
    for b in range(batch_size):
        # Convert to numpy for connected components analysis
        label_np = labels[b, 0].cpu().numpy()
        
        # Create instance mask using connected components
        # For 3-class: 0=background, 1=interior, 2=boundary
        # We'll treat interior+boundary as foreground for instance detection
        foreground_mask = (label_np > 0).astype(np.uint8)
        
        if foreground_mask.sum() > 0:
            # Label connected components
            instance_mask = measure.label(foreground_mask, connectivity=2)
            instance_masks[b] = torch.from_numpy(instance_mask).to(device)
    
    return instance_masks


def main():
    parser = argparse.ArgumentParser("Error-Aware MAUNet Training")
    
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="./data/train-preprocessed/",
        type=str,
        help="training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--work_dir", 
        default="./final_model/work_dir", 
        help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    # Model parameters
    parser.add_argument("--num_class", default=3, type=int, help="segmentation classes")
    parser.add_argument("--input_size", default=256, type=int, help="input image size")
    parser.add_argument(
        "--backbone", 
        default="resnet50", 
        type=str, 
        choices=["resnet50", "wide_resnet50"], 
        help="Backbone for Error-Aware MAUNet"
    )
    parser.add_argument(
        "--enable_auxiliary_tasks", 
        action="store_true", 
        default=True,
        help="Enable auxiliary centroid prediction task"
    )
    parser.add_argument(
        "--centroid_sigma", 
        type=float, 
        default=2.0, 
        help="Gaussian sigma for centroid heatmap generation"
    )
    
    # Loss function parameters
    parser.add_argument("--lambda_focal", type=float, default=1.0, help="Weight for focal loss")
    parser.add_argument("--lambda_tversky", type=float, default=1.0, help="Weight for tversky loss")
    parser.add_argument("--lambda_boundary", type=float, default=0.5, help="Weight for boundary loss")
    parser.add_argument("--lambda_centroid", type=float, default=0.3, help="Weight for centroid loss")
    parser.add_argument("--lambda_proxy", type=float, default=0.1, help="Weight for proxy regularization")
    
    # Focal loss parameters
    parser.add_argument("--focal_alpha", type=float, default=1.0, help="Focal loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    
    # Tversky loss parameters
    parser.add_argument("--tversky_alpha", type=float, default=0.3, help="Tversky loss alpha (FP weight)")
    parser.add_argument("--tversky_beta", type=float, default=0.7, help="Tversky loss beta (FN weight)")

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

    monai.config.print_config()

    # Set training/validation split
    np.random.seed(args.seed)
    model_path = join(args.work_dir, f"error_aware_maunet_{args.backbone}_3class")
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

    # Create file lists
    train_files = []
    for i in range(len(train_img_names)):
        sample = {
            "img": join(train_img_path, train_img_names[i]), 
            "label": join(train_gt_path, train_gt_names[i])
        }
        train_files.append(sample)

    val_files = []
    for i in range(len(val_img_names)):
        sample = {
            "img": join(val_img_path, val_img_names[i]), 
            "label": join(val_gt_path, val_gt_names[i])
        }
        val_files.append(sample)

    print(f"training image num: {len(train_files)}, validation image num: {len(val_files)}")

    # Define transforms for image and segmentation
    train_transforms = Compose([
        LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
        EnsureChannelFirstd(keys=["label", "img"]),
        ScaleIntensityd(keys=["img"]),
        SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
        RandSpatialCropd(keys=["img", "label"], roi_size=args.input_size, random_size=False),
        RandAxisFlipd(keys=["img", "label"], prob=0.5),
        RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        RandZoomd(keys=["img", "label"], prob=0.15, min_zoom=0.8, max_zoom=1.5, mode=["area", "nearest"]),
        EnsureTyped(keys=["img", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
        EnsureChannelFirstd(keys=["label", "img"]),
        ScaleIntensityd(keys=["img"]),
        SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
        CenterSpatialCropd(keys=["img", "label"], roi_size=args.input_size),
        EnsureTyped(keys=["img", "label"]),
    ])

    # Create datasets and data loaders
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

    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() or torch.backends.mps.is_available(),
    )
    
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    # Metrics
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

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "mps":
        print("✅ Training on Apple Silicon GPU (MPS)")
    elif device.type == "cuda":
        print("✅ Training on NVIDIA GPU (CUDA)")
    else:
        print("⚠️  Training on CPU")

    model = create_error_aware_maunet_model(
        num_classes=args.num_class,
        input_size=args.input_size,
        in_channels=3,
        backbone=args.backbone,
        enable_auxiliary_tasks=args.enable_auxiliary_tasks,
        centroid_sigma=args.centroid_sigma
    ).to(device)

    # Setup composite loss function
    composite_loss = CompositeLoss(
        lambda_focal=args.lambda_focal,
        lambda_tversky=args.lambda_tversky,
        lambda_boundary=args.lambda_boundary,
        lambda_centroid=args.lambda_centroid if args.enable_auxiliary_tasks else 0.0,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        centroid_sigma=args.centroid_sigma,
    ).to(device)

    # Proxy loss for embedding regularization
    proxy_ce = ProxyCELoss(temperature=0.07, ignore_index=-100).to(device)

    # Setup optimizer with separate parameter groups
    proxy_params = []
    other_params = []
    for n, p in model.named_parameters():
        if "proxies" in n:
            proxy_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": other_params, "weight_decay": 1e-2},
        {"params": proxy_params, "weight_decay": 0.0},
    ], lr=args.initial_lr)
    
    # Setup learning rate scheduler
    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )

    # Load checkpoint if specified
    start_epoch = 1
    if args.load_checkpoint and args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            print(f"Loading checkpoint from: {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            epoch_loss_values = checkpoint.get('loss', [])
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found: {args.checkpoint_path}")
            print("Starting training from scratch")

    # Training loop
    max_epochs = args.max_epochs
    epoch_tolerance = args.epoch_tolerance
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter(model_path)
    
    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_loss = 0
        epoch_losses = {'total': 0, 'focal': 0, 'tversky': 0, 'boundary': 0, 'centroid': 0, 'proxy': 0}
        
        for step, batch_data in enumerate(train_loader, 1):
            inputs = batch_data["img"].to(device)
            labels = batch_data["label"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Prepare targets
            labels_onehot = monai.networks.one_hot(labels, args.num_class)
            
            # Create instance masks for centroid supervision
            instance_masks = None
            if args.enable_auxiliary_tasks:
                instance_masks = create_instance_masks_from_labels(labels)
            
            # Compute composite loss
            loss_dict = composite_loss(outputs, labels_onehot, instance_masks)
            
            # Proxy regularization loss
            if args.lambda_proxy > 0:
                emb = outputs['embedding']
                y_full = labels.long().squeeze(1) if labels.dim() == 4 else labels.long()
                proxy_targets = torch.full_like(y_full, fill_value=-100)
                proxy_targets[y_full == 0] = 0
                proxy_targets[y_full == 1] = 1

                # Flatten and subsample for efficiency
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
                    loss_proxy = proxy_ce(emb_sel, model.proxies, tgt_sel)
                else:
                    loss_proxy = torch.tensor(0.0, device=device, dtype=emb.dtype)
                
                total_loss = loss_dict['total_loss'] + args.lambda_proxy * loss_proxy
                epoch_losses['proxy'] += loss_proxy.item()
            else:
                total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_losses['total'] += total_loss.item()
            if 'focal_loss' in loss_dict:
                epoch_losses['focal'] += loss_dict['focal_loss'].item()
            if 'tversky_loss' in loss_dict:
                epoch_losses['tversky'] += loss_dict['tversky_loss'].item()
            if 'boundary_loss' in loss_dict:
                epoch_losses['boundary'] += loss_dict['boundary_loss'].item()
            if 'centroid_loss' in loss_dict:
                epoch_losses['centroid'] += loss_dict['centroid_loss'].item()
            
            # Log step losses
            epoch_len = len(train_ds) // train_loader.batch_size
            writer.add_scalar("train_loss", total_loss.item(), epoch_len * epoch + step)
            writer.flush()

        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= step
        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
        print(f"  focal: {epoch_losses['focal']:.4f}, tversky: {epoch_losses['tversky']:.4f}")
        print(f"  boundary: {epoch_losses['boundary']:.4f}, centroid: {epoch_losses['centroid']:.4f}")
        print(f"  proxy: {epoch_losses['proxy']:.4f}")
        
        # Step learning rate scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate: {current_lr:.6f}")
            writer.add_scalar("learning_rate", current_lr, epoch)
        
        # Log epoch losses
        writer.add_scalar("epoch_loss", epoch_loss, epoch)
        for key, value in epoch_losses.items():
            writer.add_scalar(f"epoch_loss_{key}", value, epoch)
        writer.flush()
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "loss": epoch_loss_values,
        }

        # Validation
        if epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                
                for val_data in val_loader:
                    val_images = val_data["img"].to(device)
                    val_labels = val_data["label"].to(device)
                    
                    # Forward pass (use only segmentation output for validation)
                    model_outputs = model(val_images)
                    val_outputs = model_outputs['segmentation']
                    
                    # Apply post-processing transforms
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_onehot = [post_gt(i) for i in decollate_batch(val_labels)]
                    
                    # Compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels_onehot)

                # Aggregate final mean dice result
                metric = dice_metric.aggregate().item()
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
                writer.flush()
                
                # Plot validation results
                if val_images is not None and val_outputs is not None:
                    plot_2d_or_3d_image(val_images, epoch, writer, index=0, tag="image")
                    plot_2d_or_3d_image(val_labels, epoch, writer, index=0, tag="label")
                    plot_2d_or_3d_image(val_outputs, epoch, writer, index=0, tag="output")
                
                # Save predictions
                import tifffile as tif
                from skimage import measure, morphology
                
                pred_dir = join(model_path, "predictions")
                os.makedirs(pred_dir, exist_ok=True)
                
                val_pred = val_outputs[0]  # First prediction
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
                print(f"Saved Error-Aware MAUNet predictions for epoch {epoch}")
            
            # Early stopping check
            if (epoch - best_metric_epoch) > epoch_tolerance:
                print(
                    f"validation metric does not improve for {epoch_tolerance} epochs! "
                    f"current {epoch=}, {best_metric_epoch=}"
                )
                break

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    torch.save(checkpoint, join(model_path, "final_model.pth"))
    np.savez_compressed(
        join(model_path, "train_log.npz"),
        val_dice=metric_values,
        epoch_loss=epoch_loss_values,
    )


if __name__ == "__main__":
    main()
