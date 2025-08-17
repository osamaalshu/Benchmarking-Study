#!/usr/bin/env python3
"""
Evaluate MAUNet model after training
"""

import os
import subprocess
import sys
from pathlib import Path

def run_maunet_evaluation():
    """Run complete MAUNet evaluation pipeline"""
    
    print("="*60)
    print("MAUNet Evaluation Pipeline")
    print("="*60)
    
    # Configuration
    model_path = "./baseline/work_dir/maunet_3class/maunet_resnet50"
    test_input_path = "./data/test"
    predictions_path = "./test_predictions/maunet"
    gt_path = "./data/val/labels"  # Assuming validation labels as ground truth
    
    # Create predictions directory
    os.makedirs(predictions_path, exist_ok=True)
    
    # Check if model exists
    best_model_path = os.path.join(model_path, "best_model.pth")
    final_model_path = os.path.join(model_path, "final_model.pth")
    
    if os.path.exists(best_model_path):
        model_file = best_model_path
        print(f"✅ Using best model: {best_model_path}")
    elif os.path.exists(final_model_path):
        model_file = final_model_path
        print(f"✅ Using final model: {final_model_path}")
    else:
        print("❌ No trained model found!")
        return False
    
    # Step 1: Run predictions
    print("\n1. Running MAUNet predictions...")
    cmd = [
        "python", "utils/predict.py",
        "--model_name", "maunet",
        "--model_path", model_path,
        "--input_path", test_input_path,
        "--output_path", predictions_path,
        "--num_class", "3",
        "--input_size", "256",
        "--show_overlay"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ MAUNet predictions completed successfully")
    else:
        print(f"❌ MAUNet predictions failed: {result.stderr}")
        return False
    
    # Step 2: Run metrics computation
    print("\n2. Computing evaluation metrics...")
    cmd = [
        "python", "utils/compute_metric.py",
        "--gt_path", gt_path,
        "--seg_path", predictions_path,
        "--output_path", f"{predictions_path}/metrics.json"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Metrics computation completed successfully")
    else:
        print(f"❌ Metrics computation failed: {result.stderr}")
        return False
    
    # Step 3: Display results
    print("\n3. Evaluation Results:")
    metrics_file = f"{predictions_path}/metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            print(f.read())
    
    print("\n" + "="*60)
    print("MAUNet Evaluation Complete!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    run_maunet_evaluation() 