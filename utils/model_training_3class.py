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
from monai.data import decollate_batch, PILReader
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
)
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.unetr2d import UNETR2D
from models.sac_model import SACModel, create_default_points
from models.nnunet import create_nnunet_model
from models.lstmunet import create_lstmunet_model



print("Successfully imported all requirements!")


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
        "--model_name", default="unet", help="select mode: unet, unetr, swinunetr, sac, nnunet, lstmunet"
    )
    parser.add_argument("--num_class", default=3, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=256, type=int, help="segmentation classes"
    )
    # Training parameters
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=2000, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=10, type=int)
    parser.add_argument("--initial_lr", type=float, default=6e-4, help="learning rate")

    args = parser.parse_args()

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

    train_files = [
        {"img": join(train_img_path, train_img_names[i]), "label": join(train_gt_path, train_gt_names[i])}
        for i in range(len(train_img_names))
    ]
    val_files = [
        {"img": join(val_img_path, val_img_names[i]), "label": join(val_gt_path, val_gt_names[i])}
        for i in range(len(val_img_names))
    ]
    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)}"
    )
    #%% define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["img", "label"], reader=PILReader, dtype=np.uint8
            ),  # image three channels (H, W, 3); label: (H, W)
            EnsureChannelFirstd(keys=["label"], allow_missing_keys=True),  # label: (1, H, W)
            EnsureChannelFirstd(keys=["img"], allow_missing_keys=True),  # image: (3, H, W)
            ScaleIntensityd(
                keys=["img"], allow_missing_keys=True
            ),  # Do not scale label
            SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
            RandSpatialCropd(
                keys=["img", "label"], roi_size=args.input_size, random_size=False
            ),
            RandAxisFlipd(keys=["img", "label"], prob=0.5),
            RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
            # # intensity transform
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            RandZoomd(
                keys=["img", "label"],
                prob=0.15,
                min_zoom=0.8,
                max_zoom=1.5,
                mode=["area", "nearest"],
            ),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
            EnsureChannelFirstd(keys=["label"], allow_missing_keys=True),
            EnsureChannelFirstd(keys=["img"], allow_missing_keys=True),
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            # Ensure validation images are resized to expected input size
            SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
            RandSpatialCropd(
                keys=["img", "label"], roi_size=args.input_size, random_size=False
            ),
            EnsureTyped(keys=["img", "label"]),
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

    post_pred = Compose(
        [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
    )
    post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
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

    if args.model_name.lower() == "unetr":
        model = UNETR2D(
            in_channels=3,
            out_channels=args.num_class,
            img_size=(args.input_size, args.input_size),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)

    if args.model_name.lower() == "swinunetr":
        model = monai.networks.nets.SwinUNETR(
            img_size=(args.input_size, args.input_size),
            in_channels=3,
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=2,
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

    loss_function = monai.losses.DiceCELoss(softmax=True)
    initial_lr = args.initial_lr
    optimizer = torch.optim.AdamW(model.parameters(), initial_lr)

    # start a typical PyTorch training
    max_epochs = args.max_epochs
    epoch_tolerance = args.epoch_tolerance
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(model_path)
    for epoch in range(1, max_epochs):
        model.train()
        epoch_loss = 0
        for step, batch_data in enumerate(train_loader, 1):
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(
                device
            )
            optimizer.zero_grad()
            
            # Handle SAC model differently (requires points)
            if args.model_name.lower() == "sac":
                batch_size = inputs.shape[0]
                points = create_default_points(batch_size, (args.input_size, args.input_size))
                points = points.to(device)
                outputs = model(inputs, points=points)
            else:
                outputs = model(inputs)
                
            labels_onehot = monai.networks.one_hot(
                labels, args.num_class
            )  # (b,cls,256,256)
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
        writer.add_scalar("epoch_loss", epoch_loss, epoch)
        writer.flush()  # Force flush to ensure logs are written
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss_values,
        }

        if epoch > 20 and epoch % val_interval == 0:
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
                    
                    val_labels_onehot = monai.networks.one_hot(
                        val_labels, args.num_class
                    )
                    
                    roi_size = (args.input_size, args.input_size)
                    sw_batch_size = 4
                    
                    # Handle SAC model differently for validation
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
                    else:
                        val_outputs = sliding_window_inference(
                            val_images, roi_size, sw_batch_size, model
                        )
                    
                    # Apply post-processing transforms
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_onehot = [post_gt(i) for i in decollate_batch(val_labels_onehot)]
                    
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
                
                # Save SAC and nnU-Net model predictions during validation
                if args.model_name.lower() in ["sac", "nnunet"]:
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
