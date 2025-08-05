#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAUNet Training Script with Distance Transform
Adapted for the existing benchmarking framework
"""

import argparse
import os
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
    ResizeWithPadOrCropd,
)
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.maunet import MAUNet, create_maunet_model, WeightedL1Loss

print("Successfully imported all requirements!")


def main():
    parser = argparse.ArgumentParser("MAUNet Training Script")
    # Dataset parameters
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="path to preprocessed data; should contain images/ and labels/ folders"
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        required=True,
        help="path to distance transform weights; should contain labels/ folder"
    )
    parser.add_argument(
        "--work_dir", 
        default="./baseline/work_dir/maunet_3class",
        help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    # Model parameters
    parser.add_argument(
        "--backbone", 
        default="resnet50",
        choices=["resnet50", "wide_resnet50"],
        help="backbone architecture"
    )
    parser.add_argument("--num_class", default=3, type=int, help="segmentation classes")
    parser.add_argument("--input_size", default=512, type=int, help="input image size")
    
    # Training parameters
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--val_interval", default=5, type=int)
    parser.add_argument("--lr", default=6e-4, type=float, help="learning rate")
    parser.add_argument("--model_path", type=str, default=None, help="pretrained model path")
    parser.add_argument("--patience", default=10, type=int, help="early stopping patience")
    
    args = parser.parse_args()

    # Set seeds for reproducibility
    monai.utils.set_determinism(seed=args.seed)
    
    # Create working directory
    model_path = os.path.join(args.work_dir, f"maunet_{args.backbone}")
    os.makedirs(model_path, exist_ok=True)
    
    # Setup data paths
    train_images = sorted([
        os.path.join(args.data_path, "images", x) 
        for x in os.listdir(os.path.join(args.data_path, "images"))
        if x.endswith((".png", ".jpg", ".tif", ".tiff"))
    ])
    train_labels = sorted([
        os.path.join(args.data_path, "labels", x) 
        for x in os.listdir(os.path.join(args.data_path, "labels"))
        if x.endswith((".png", ".jpg", ".tif", ".tiff"))
    ])
    # Get weight files (distance transform .npy files)
    train_weights = sorted([
        os.path.join(args.weight_path, x) 
        for x in os.listdir(args.weight_path)
        if x.endswith(".npy")
    ])
    
    # Split data (80% train, 20% val)
    val_split = 0.2
    val_size = int(len(train_images) * val_split)
    
    val_files = [
        {"img": img, "seg": seg, "weight": weight}
        for img, seg, weight in zip(train_images[:val_size], train_labels[:val_size], train_weights[:val_size])
    ]
    train_files = [
        {"img": img, "seg": seg, "weight": weight}
        for img, seg, weight in zip(train_images[val_size:], train_labels[val_size:], train_weights[val_size:])
    ]
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Define transforms based on original saltfish implementation
    train_transforms = Compose([
        LoadImaged(keys=["img", "seg"], reader=PILReader, dtype=np.uint8),
        LoadImaged(keys=["weight"], reader=NumpyReader, dtype=np.float32),
        EnsureChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
        EnsureChannelFirstd(keys=["seg", "weight"], allow_missing_keys=True),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        RandZoomd(
            keys=["img", "seg", "weight"],
            prob=1,
            min_zoom=0.25,
            max_zoom=2,
            mode=["area", "nearest", "bilinear"],
            keep_size=False,
        ),
        SpatialPadd(keys=["img", "seg", "weight"], spatial_size=args.input_size, mode="constant"),
        RandSpatialCropd(
            keys=["img", "seg", "weight"], 
            roi_size=(args.input_size, args.input_size), 
            random_size=False
        ),
        RandAxisFlipd(keys=["img", "seg", "weight"], prob=0.5),
        RandRotate90d(keys=["img", "seg", "weight"], prob=0.5, spatial_axes=[0, 1]),
        RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
        RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
        RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
        RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
        EnsureTyped(keys=["img", "seg", "weight"]),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["img", "seg"], reader=PILReader, dtype=np.uint8),
        LoadImaged(keys=["weight"], reader=NumpyReader, dtype=np.float32),
        EnsureChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
        EnsureChannelFirstd(keys=["seg", "weight"], allow_missing_keys=True),
        ScaleIntensityd(keys=["img"], allow_missing_keys=True),
        SpatialPadd(keys=["img", "seg", "weight"], spatial_size=args.input_size, mode="constant"),
        EnsureTyped(keys=["img", "seg", "weight"]),
    ])
    
    # Create datasets and dataloaders
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_maunet_model(
        num_classes=args.num_class,  # Use num_class directly to match checkpoint
        input_size=args.input_size,
        in_channels=3,  # RGB images
        backbone=args.backbone
    ).to(device)
    
    # Load pretrained weights if provided
    if args.model_path:
        print(f"Loading pretrained model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Setup losses and optimizer based on original implementation
    loss_function = monai.losses.DiceFocalLoss(softmax=False)
    loss_function2 = WeightedL1Loss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95, last_epoch=-1)
    
    # Setup metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=args.num_class)])
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(model_path, "logs"))
    
    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    patience = args.patience  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(args.max_epochs):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{args.max_epochs}")
        
        model.train()
        epoch_loss = 0
        step = 0
        
        for step, batch_data in enumerate(train_loader, 1):
            inputs, labels, weights = (
                batch_data["img"].to(device),
                batch_data["seg"].to(device),
                batch_data["weight"].to(device),
            )
            
            optimizer.zero_grad()
            outputs_seg, outputs_dt = model(inputs)
            
            # Apply softmax and take first 3 channels (matching original implementation)
            outputs_seg = torch.softmax(outputs_seg, dim=1)
            outputs_seg = outputs_seg[:, 0:3, :, :]
            
            # Convert labels to one-hot encoding
            labels_onehot = monai.networks.one_hot(labels, args.num_class)
            
            # Compute combined loss
            loss1 = loss_function(outputs_seg, labels_onehot)
            loss2 = loss_function2(outputs_dt, weights)
            loss = loss1 + loss2
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            
            writer.add_scalar("train_loss", loss1.item(), epoch_len * epoch + step)
            writer.add_scalar("dis_loss", loss2.item(), epoch_len * epoch + step)
        
        epoch_loss /= step
        writer.add_scalar("avg_train_loss", epoch_loss, epoch)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        scheduler.step()
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print("lr:", optimizer.param_groups[0]['lr'])
        
        # Save checkpoint every 10 epochs (matching original)
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            }
            torch.save(checkpoint, os.path.join(model_path, "best_Dice_model.pth"))
    
    writer.close()
    print(f"Training completed!")
    
    # Save final model
    torch.save(checkpoint, os.path.join(model_path, "final_model.pth"))


if __name__ == "__main__":
    main() 