#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize segmentation results: predictions vs ground truth
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io, measure, segmentation
import tifffile as tif
from pathlib import Path
import random

def load_image(path):
    """Load image from various formats"""
    if path.endswith('.tif') or path.endswith('.tiff'):
        return tif.imread(path)
    else:
        return io.imread(path)

def create_colored_mask(label_mask):
    """Convert label mask to colored visualization"""
    # Create random colors for each instance
    max_label = label_mask.max()
    if max_label == 0:
        return np.zeros((*label_mask.shape, 3), dtype=np.uint8)
    
    # Generate distinct colors for each instance
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(50, 255, size=(max_label + 1, 3))
    colors[0] = [0, 0, 0]  # Background is black
    
    # Apply colors to mask
    colored_mask = colors[label_mask]
    return colored_mask.astype(np.uint8)

def plot_comparison(img, gt_mask, pred_mask, model_name, img_name, save_path):
    """Create side-by-side comparison plot"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Ground truth
    gt_colored = create_colored_mask(gt_mask)
    axes[1].imshow(gt_colored)
    axes[1].set_title(f'Ground Truth\n({gt_mask.max()} cells)', fontsize=14)
    axes[1].axis('off')
    
    # Prediction
    pred_colored = create_colored_mask(pred_mask)
    axes[2].imshow(pred_colored)
    axes[2].set_title(f'{model_name} Prediction\n({pred_mask.max()} cells)', fontsize=14)
    axes[2].axis('off')
    
    # Overlay
    overlay = img.copy()
    # Find boundaries
    gt_boundaries = segmentation.find_boundaries(gt_mask, mode='outer')
    pred_boundaries = segmentation.find_boundaries(pred_mask, mode='outer')
    
    # Color boundaries: green for GT, red for predictions
    overlay[gt_boundaries] = [0, 255, 0]
    overlay[pred_boundaries] = [255, 0, 0]
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay\n(Green: GT, Red: Pred)', fontsize=14)
    axes[3].axis('off')
    
    plt.suptitle(f'{model_name} - {img_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def calculate_metrics_summary(csv_files, output_path):
    """Create a summary table of all metrics"""
    summary_data = []
    
    # Look for metrics files with threshold suffixes
    models = ['unet', 'nnunet', 'sac', 'lstmunet', 'maunet', 'maunet_ensemble']
    
    for model in models:
        # Try to find the 0.5 threshold file
        csv_file = os.path.join(os.path.dirname(csv_files[0]), f"{model}_metrics-0.5.csv")
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            summary_data.append({
                'Model': model.upper(),
                'Mean F1 (0.5)': df['F1'].mean(),
                'Median F1 (0.5)': df['F1'].median(),
                'Mean Dice (0.5)': df['dice'].mean(),
                'Mean Precision (0.5)': df['precision'].mean(),
                'Mean Recall (0.5)': df['recall'].mean(),
            })
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Mean F1 (0.5)', ascending=False)
    
    # Save summary
    summary_df.to_csv(os.path.join(output_path, 'models_summary.csv'), index=False)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary_df))
    width = 0.35
    
    ax.bar(x - width/2, summary_df['Mean F1 (0.5)'], width, label='Mean F1', color='skyblue')
    ax.bar(x + width/2, summary_df['Mean Dice (0.5)'], width, label='Mean Dice', color='lightcoral')
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison (Threshold=0.5)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Model'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'models_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return summary_df

def main():
    # Configuration
    test_images_path = "./data/test/images"
    test_labels_path = "./data/test/labels"
    predictions_base = "./test_predictions"
    output_base = "./visualization_results"
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Models to visualize
    models = ['unet', 'nnunet', 'sac', 'lstmunet', 'maunet', 'maunet_ensemble']
    
    # Get list of test images
    test_images = sorted([f for f in os.listdir(test_images_path) 
                         if f.endswith(('.tif', '.tiff', '.png', '.jpg'))])
    
    # Select a subset of images to visualize (e.g., 10 random images)
    random.seed(42)
    selected_images = random.sample(test_images, min(10, len(test_images)))
    
    print(f"Generating visualizations for {len(selected_images)} images...")
    
    # Create visualizations for each model
    for model in models:
        model_output_dir = os.path.join(output_base, model)
        os.makedirs(model_output_dir, exist_ok=True)
        
        pred_path = os.path.join(predictions_base, model)
        if not os.path.exists(pred_path):
            print(f"⚠️  Predictions not found for {model}")
            continue
        
        print(f"\nProcessing {model}...")
        
        for img_name in selected_images:
            # Load original image
            img = load_image(os.path.join(test_images_path, img_name))
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] > 3:
                img = img[:, :, :3]
            
            # Load ground truth
            base_name = os.path.splitext(img_name)[0]
            gt_path = os.path.join(test_labels_path, f"{base_name}_label.tiff")
            if not os.path.exists(gt_path):
                print(f"  ⚠️  Ground truth not found for {img_name}")
                continue
            gt_mask = load_image(gt_path)
            
            # Load prediction
            pred_file = os.path.join(pred_path, f"{base_name}_label.tiff")
            if not os.path.exists(pred_file):
                print(f"  ⚠️  Prediction not found for {img_name}")
                continue
            pred_mask = load_image(pred_file)
            
            # Create comparison plot
            save_path = os.path.join(model_output_dir, f"{base_name}_comparison.png")
            plot_comparison(img, gt_mask, pred_mask, model.upper(), base_name, save_path)
            print(f"  ✅ Saved {base_name}_comparison.png")
    
    # Create metrics summary
    print("\nGenerating metrics summary...")
    csv_files = [os.path.join(predictions_base, f"{model}_metrics.csv") for model in models]
    summary_df = calculate_metrics_summary(csv_files, output_base)
    
    print("\nMetrics Summary (Threshold=0.5):")
    print(summary_df.to_string(index=False))
    
    print(f"\n✅ All visualizations saved in: {output_base}")
    print(f"✅ Metrics summary saved as: {output_base}/models_summary.csv")
    print(f"✅ Comparison plot saved as: {output_base}/models_comparison.png")

if __name__ == "__main__":
    main() 