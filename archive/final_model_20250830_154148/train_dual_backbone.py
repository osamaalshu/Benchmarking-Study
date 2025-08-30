#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Backbone Training Script for Error-Aware MAUNet

This script trains two separate Error-Aware MAUNet models with different backbones:
1. ResNet50 backbone
2. Wide-ResNet50 backbone

After training both models, they can be combined into an ensemble for improved
performance, following the approach used in the original MAUNet paper.

Usage:
    python train_dual_backbone.py --data_path ./data/train-preprocessed/ --work_dir ./final_model/work_dir
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime
from typing import List

def train_single_backbone(backbone: str, base_args: dict, work_dir: str):
    """
    Train a single backbone model
    
    Args:
        backbone: Either "resnet50" or "wide_resnet50"
        base_args: Base training arguments
        work_dir: Working directory for outputs
    """
    print(f"\n{'='*60}")
    print(f"Training Error-Aware MAUNet with {backbone} backbone")
    print(f"{'='*60}")
    
    # Construct command
    cmd = ["python", "train_error_aware_maunet.py"]
    
    # Add base arguments
    for key, value in base_args.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{key}", str(value)])
    
    # Add backbone-specific arguments
    cmd.extend(["--backbone", backbone])
    cmd.extend(["--work_dir", work_dir])
    
    # Run training
    print(f"Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    
    if result.returncode != 0:
        print(f"❌ Training failed for {backbone} backbone")
        return False
    else:
        print(f"✅ Training completed successfully for {backbone} backbone")
        return True


def main():
    parser = argparse.ArgumentParser("Dual Backbone Training for Error-Aware MAUNet")
    
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
        help="base path where to save models and logs"
    )
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    # Model parameters
    parser.add_argument("--num_class", default=3, type=int, help="segmentation classes")
    parser.add_argument("--input_size", default=256, type=int, help="input image size")
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
    
    # Training control
    parser.add_argument(
        "--train_backbones", 
        nargs="+", 
        default=["resnet50", "wide_resnet50"],
        choices=["resnet50", "wide_resnet50"],
        help="Which backbones to train"
    )
    parser.add_argument("--sequential", action="store_true", help="Train backbones sequentially (default: parallel)")

    args = parser.parse_args()

    # Create base work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Prepare base arguments for training
    base_args = {
        "data_path": args.data_path,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "num_class": args.num_class,
        "input_size": args.input_size,
        "enable_auxiliary_tasks": args.enable_auxiliary_tasks,
        "centroid_sigma": args.centroid_sigma,
        "lambda_focal": args.lambda_focal,
        "lambda_tversky": args.lambda_tversky,
        "lambda_boundary": args.lambda_boundary,
        "lambda_centroid": args.lambda_centroid,
        "lambda_proxy": args.lambda_proxy,
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "tversky_alpha": args.tversky_alpha,
        "tversky_beta": args.tversky_beta,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "val_interval": args.val_interval,
        "epoch_tolerance": args.epoch_tolerance,
        "initial_lr": args.initial_lr,
        "lr_scheduler": args.lr_scheduler,
        "lr_gamma": args.lr_gamma,
        "lr_step_size": args.lr_step_size,
    }
    
    print(f"Starting dual backbone training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backbones to train: {args.train_backbones}")
    print(f"Training mode: {'Sequential' if args.sequential else 'Parallel'}")
    
    if args.sequential:
        # Train backbones sequentially
        success_count = 0
        for backbone in args.train_backbones:
            success = train_single_backbone(backbone, base_args, args.work_dir)
            if success:
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"Training Summary:")
        print(f"Successfully trained: {success_count}/{len(args.train_backbones)} models")
        print(f"{'='*60}")
        
    else:
        # Train backbones in parallel (using subprocess)
        print("⚠️  Parallel training not implemented yet. Falling back to sequential training.")
        # For now, fall back to sequential training
        # In a production environment, you might use multiprocessing or job schedulers
        success_count = 0
        for backbone in args.train_backbones:
            success = train_single_backbone(backbone, base_args, args.work_dir)
            if success:
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"Training Summary:")
        print(f"Successfully trained: {success_count}/{len(args.train_backbones)} models")
        print(f"{'='*60}")
    
    # Create ensemble combination script
    if success_count >= 2:
        print("\n🎯 Creating ensemble combination script...")
        create_ensemble_script(args.work_dir, args.train_backbones)


def create_ensemble_script(work_dir: str, trained_backbones: List[str]):
    """
    Create a script to combine trained models into an ensemble
    
    Args:
        work_dir: Working directory containing trained models
        trained_backbones: List of successfully trained backbones
    """
    ensemble_script = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Ensemble Creation Script for Error-Aware MAUNet

This script combines the trained {' and '.join(trained_backbones)} models into an ensemble.
Generated automatically by train_dual_backbone.py
'''

import torch
import os
from error_aware_maunet import create_error_aware_maunet_model, ErrorAwareMAUNetEnsemble

def create_ensemble():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {{device}}")
    
    # Model configuration
    model_config = {{
        'num_classes': 3,
        'input_size': 256,
        'in_channels': 3,
        'enable_auxiliary_tasks': True,
        'centroid_sigma': 2.0
    }}
    
    # Load trained models
    models = []
    model_paths = []
    
"""
    
    for backbone in trained_backbones:
        ensemble_script += f"""    
    # Load {backbone} model
    {backbone}_path = os.path.join("{work_dir}", "error_aware_maunet_{backbone}_3class", "best_Dice_model.pth")
    if os.path.exists({backbone}_path):
        {backbone}_model = create_error_aware_maunet_model(backbone="{backbone}", **model_config)
        checkpoint = torch.load({backbone}_path, map_location=device)
        {backbone}_model.load_state_dict(checkpoint['model_state_dict'])
        models.append({backbone}_model)
        model_paths.append({backbone}_path)
        print(f"✅ Loaded {backbone} model from {{checkpoint['epoch']}} epochs")
    else:
        print(f"❌ {backbone} model not found at {{{backbone}_path}}")
"""
    
    ensemble_script += f"""
    
    # Create ensemble
    if len(models) >= 2:
        ensemble = ErrorAwareMAUNetEnsemble(models, average=True)
        ensemble.to(device)
        
        # Save ensemble
        ensemble_path = os.path.join("{work_dir}", "error_aware_maunet_ensemble.pth")
        torch.save({{
            'ensemble_state_dict': ensemble.state_dict(),
            'model_paths': model_paths,
            'backbones': {trained_backbones},
            'model_config': model_config
        }}, ensemble_path)
        
        print(f"🎯 Ensemble saved to {{ensemble_path}}")
        print(f"Ensemble contains {{len(models)}} models: {', '.join(trained_backbones)}")
        
        return ensemble, ensemble_path
    else:
        print("❌ Need at least 2 models to create ensemble")
        return None, None

if __name__ == "__main__":
    create_ensemble()
"""
    
    script_path = os.path.join(work_dir, "create_ensemble.py")
    with open(script_path, 'w') as f:
        f.write(ensemble_script)
    
    print(f"📝 Ensemble creation script saved to: {script_path}")
    print("   Run this script after training completes to create the ensemble model.")


if __name__ == "__main__":
    main()
