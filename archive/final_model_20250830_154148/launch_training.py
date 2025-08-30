#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch Training Script for Error-Aware MAUNet

Simple launcher script to start training with optimized hyperparameters
based on the error analysis findings.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the current directory (final_model)
    current_dir = Path(__file__).parent
    
    # Set up paths relative to the project root
    project_root = current_dir.parent
    data_path = project_root / "data" / "train-preprocessed"
    work_dir = current_dir / "work_dir"
    
    print("🚀 Launching Error-Aware MAUNet Training")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Data path: {data_path}")
    print(f"Work directory: {work_dir}")
    print("=" * 50)
    
    # Check if data exists
    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        print("Please ensure the preprocessed training data is available.")
        return
    
    # Create work directory
    work_dir.mkdir(exist_ok=True)
    
    # Optimized hyperparameters based on error analysis
    cmd = [
        sys.executable, "train_dual_backbone.py",
        "--data_path", str(data_path),
        "--work_dir", str(work_dir),
        "--train_backbones", "resnet50", "wide_resnet50",
        
        # Model configuration
        "--num_class", "3",
        "--input_size", "256",
        "--enable_auxiliary_tasks",
        "--centroid_sigma", "2.0",
        
        # Loss function weights (optimized for false negative reduction)
        "--lambda_focal", "1.0",      # Focus on hard examples
        "--lambda_tversky", "1.2",    # Slightly higher weight for FN reduction
        "--lambda_boundary", "0.5",   # Boundary precision
        "--lambda_centroid", "0.3",   # Instance detection
        "--lambda_proxy", "0.1",      # Embedding regularization
        
        # Focal loss parameters
        "--focal_alpha", "1.0",
        "--focal_gamma", "2.0",       # Standard focusing
        
        # Tversky loss parameters (β > α for FN reduction)
        "--tversky_alpha", "0.3",     # Lower weight for FP
        "--tversky_beta", "0.7",      # Higher weight for FN
        
        # Training parameters
        "--batch_size", "8",
        "--max_epochs", "2000",
        "--val_interval", "2",
        "--epoch_tolerance", "15",    # Slightly more patience
        "--initial_lr", "6e-4",
        "--lr_scheduler",
        "--lr_gamma", "0.95",
        "--lr_step_size", "10",
        
        # Training control
        "--sequential",  # Train sequentially for stability
    ]
    
    print("🔧 Training Configuration:")
    print("  • Dual backbone training (ResNet50 + Wide-ResNet50)")
    print("  • Multi-objective loss with FN reduction focus")
    print("  • Auxiliary tasks enabled (centroid + distance)")
    print("  • Optimized hyperparameters from error analysis")
    print("  • Sequential training for stability")
    print()
    
    # Ask for confirmation
    response = input("Start training? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    print("🎯 Starting training...")
    print("Command:", " ".join(cmd))
    print()
    
    # Change to final_model directory and run training
    os.chdir(current_dir)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        print(f"Models saved in: {work_dir}")
        print("\n📦 Next steps:")
        print("1. Check training logs in the work_dir")
        print("2. Run create_ensemble.py to combine models")
        print("3. Use predict_error_aware_maunet.py for inference")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code: {e.returncode}")
        print("Check the logs for more details.")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
