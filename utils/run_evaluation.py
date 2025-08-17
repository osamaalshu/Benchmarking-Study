#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run predictions on all models and compute metrics
"""

import os
import subprocess
import sys
from pathlib import Path
import shutil

def run_prediction(model_name, model_path, input_path, output_path):
    """Run prediction for a specific model"""
    print(f"\n{'='*60}")
    print(f"Running prediction for {model_name}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "utils/predict.py",
        "--model_name", model_name,
        "--model_path", model_path,
        "--input_path", input_path,
        "--output_path", output_path,
        "--num_class", "3",
        "--input_size", "256",
        "--show_overlay"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {model_name} prediction completed successfully")
        return True
    else:
        print(f"❌ {model_name} prediction failed")
        print(f"Error: {result.stderr}")
        return False

def run_metrics(gt_path, seg_path, output_path, model_name):
    """Run metrics computation for a specific model"""
    print(f"\n{'='*60}")
    print(f"Computing metrics for {model_name}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "utils/compute_metric.py",
        "--gt_path", gt_path,
        "--seg_path", seg_path,
        "--output_path", output_path,
        "--save_name", f"{model_name}_metrics",
        "--gt_suffix", "_label.tiff",
        "--seg_suffix", "_label.tiff",
        "--thresholds", "0.5", "0.7", "0.9"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {model_name} metrics computed successfully")
        print(f"Output: {result.stdout}")
        return True
    else:
        print(f"❌ {model_name} metrics computation failed")
        print(f"Error: {result.stderr}")
        return False

def main():
    # Configuration
    test_input_path = "./data/test/images"
    test_gt_path = "./data/test/labels"
    base_output_path = "./test_predictions"
    
    # Clean up output directory
    if os.path.exists(base_output_path):
        shutil.rmtree(base_output_path)
    os.makedirs(base_output_path, exist_ok=True)
    
    # Models to evaluate
    models = [
        {
            "name": "unet",
            "path": "./baseline/work_dir/unet_3class"
        },
        {
            "name": "nnunet", 
            "path": "./baseline/work_dir/nnunet_3class"
        },
        {
            "name": "sac",
            "path": "./baseline/work_dir/sac_3class"
        },
        {
            "name": "lstmunet",
            "path": "./baseline/work_dir/lstmunet_3class"
        },
        {
            "name": "maunet",
            "path": "./baseline/work_dir/maunet_3class/maunet_resnet50"
        }
    ]
    
    # Check if test data exists
    if not os.path.exists(test_input_path):
        print(f"❌ Test input path not found: {test_input_path}")
        return
    
    if not os.path.exists(test_gt_path):
        print(f"❌ Test ground truth path not found: {test_gt_path}")
        return
    
    print(f"Test input path: {test_input_path}")
    print(f"Test ground truth path: {test_gt_path}")
    print(f"Number of test images: {len(os.listdir(test_input_path))}")
    
    # Run predictions for each model
    successful_predictions = []
    
    for model in models:
        model_name = model["name"]
        model_path = model["path"]
        output_path = os.path.join(base_output_path, model_name)
        
        # Check if model checkpoint exists
        checkpoint_path = os.path.join(model_path, "best_Dice_model.pth")
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  No checkpoint found for {model_name}: {checkpoint_path}")
            continue
        
        # Run prediction
        if run_prediction(model_name, model_path, test_input_path, output_path):
            successful_predictions.append(model)
    
    print(f"\n{'='*60}")
    print(f"PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Successful predictions: {len(successful_predictions)}/{len(models)}")
    
    # Run metrics for successful predictions
    print(f"\n{'='*60}")
    print(f"COMPUTING METRICS")
    print(f"{'='*60}")
    
    for model in successful_predictions:
        model_name = model["name"]
        seg_path = os.path.join(base_output_path, model_name)
        
        # Run metrics
        run_metrics(test_gt_path, seg_path, base_output_path, model_name)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved in: {base_output_path}")
    print(f"Check the following files for metrics:")
    for model in successful_predictions:
        model_name = model["name"]
        print(f"  - {model_name}_metrics.csv")

if __name__ == "__main__":
    main() 