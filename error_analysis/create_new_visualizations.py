#!/usr/bin/env python3
"""
Create new comprehensive visualizations from existing error analysis results
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
import tifffile as tiff
import fastremap
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Add paths
framework_dir = Path(__file__).parent
sys.path.insert(0, str(framework_dir / 'src'))
sys.path.insert(0, str(framework_dir))
sys.path.insert(0, str(framework_dir / 'config'))

try:
    from config.analysis_config import *
except ImportError:
    from analysis_config import *

# Import functions from main pipeline
from error_analysis_pipeline import (
    ErrorAnalysisDataLoader, 
    create_comprehensive_visualization, 
    create_success_failure_analysis,
    enhance_image_for_display,
    _iou_matrix
)

def main():
    """Generate new visualizations from existing results"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Creating new comprehensive visualizations...")
    
    # Initialize data loader
    data_loader = ErrorAnalysisDataLoader(logger)
    
    # Load existing results
    results_dir = Path(__file__).parent / "results" / "error_analysis"
    summary_path = results_dir / "error_summary.csv"
    
    if not summary_path.exists():
        logger.error(f"Results file not found: {summary_path}")
        return
    
    summary_df = pd.read_csv(summary_path)
    logger.info(f"Loaded {len(summary_df)} results from existing analysis")
    
    # Create visualization directories
    viz_dir = results_dir / "new_visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    comprehensive_viz_dir = results_dir / "comprehensive_visualizations"
    comprehensive_viz_dir.mkdir(exist_ok=True)
    
    # Organize results by image for comprehensive visualization
    all_results = {}
    for _, row in summary_df.iterrows():
        image_name = row['image']
        model_name = row['model']
        
        if image_name not in all_results:
            all_results[image_name] = {}
        
        # Load prediction instances for this model/image
        pred_instances = data_loader.load_prediction_instances(model_name, image_name)
        if pred_instances is not None:
            all_results[image_name][model_name] = {
                'results': row.to_dict(),
                'predictions': pred_instances
            }
    
    # Create comprehensive visualizations for images 70-100 (the white/blank ones)
    all_image_names = list(all_results.keys())
    
    # Filter for a few test images first (70-75) to check fixes
    target_images = []
    for img_name in all_image_names:
        # Extract number from image name
        import re
        match = re.search(r'(\d+)', img_name)
        if match:
            img_num = int(match.group(1))
            if 70 <= img_num <= 100:  # All challenging images
                target_images.append(img_name)
    
    logger.info(f"Creating enhanced visualizations for {len(target_images)} challenging images (70-100)...")
    
    successful_visualizations = 0
    failed_visualizations = 0
    
    for i, image_name in enumerate(target_images, 1):
        logger.info(f"Processing {i}/{len(target_images)}: {image_name}")
        
        try:
            gt_instances = data_loader.load_ground_truth_instances(image_name)
            if gt_instances is not None:
                save_path = create_comprehensive_visualization(image_name, gt_instances, all_results[image_name], comprehensive_viz_dir)
                logger.info(f"✓ Created comprehensive visualization: {save_path.name}")
                successful_visualizations += 1
            else:
                logger.warning(f"Could not load ground truth for {image_name}")
                failed_visualizations += 1
        except Exception as e:
            logger.error(f"Error creating comprehensive visualization for {image_name}: {str(e)}")
            failed_visualizations += 1
    
    logger.info(f"Visualization summary: {successful_visualizations} successful, {failed_visualizations} failed")
    
    # Create success/failure analysis
    logger.info("Creating success/failure analysis...")
    try:
        create_success_failure_analysis(summary_df, data_loader, viz_dir)
        logger.info("Created success/failure analysis")
    except Exception as e:
        logger.error(f"Error creating success/failure analysis: {str(e)}")
    
    # Create additional analysis: Best vs Worst images overall
    logger.info("Creating best vs worst overall analysis...")
    try:
        create_best_worst_overall_analysis(summary_df, data_loader, viz_dir)
        logger.info("Created best vs worst overall analysis")
    except Exception as e:
        logger.error(f"Error creating best vs worst analysis: {str(e)}")
    
    logger.info("New visualization generation complete!")

def create_best_worst_overall_analysis(summary_df, data_loader, save_dir):
    """Create analysis showing best and worst performing images across all models"""
    
    # Calculate average F1 score per image across all models
    image_performance = summary_df.groupby('image')['f1_score'].agg(['mean', 'std', 'count']).reset_index()
    image_performance = image_performance.sort_values('mean')
    
    # Get worst and best images
    worst_images = image_performance.head(5)  # Bottom 5
    best_images = image_performance.tail(5)   # Top 5
    
    # Create visualization
    fig, axes = plt.subplots(2, 10, figsize=(30, 6))
    fig.suptitle('Best vs Worst Performing Images (Average F1 Across All Models)', fontsize=16)
    
    # Process worst images
    for i, (_, img_data) in enumerate(worst_images.iterrows()):
        if i >= 5:
            break
            
        image_name = img_data['image']
        gt_instances = data_loader.load_ground_truth_instances(image_name)
        
        if gt_instances is not None:
            # Load original image
            try:
                original_image_path = TEST_IMAGES_PATH / image_name
                if original_image_path.exists():
                    if original_image_path.suffix.lower() in {'.tif', '.tiff'}:
                        original_image = tiff.imread(str(original_image_path))
                    else:
                        original_image = np.array(Image.open(original_image_path))
                    
                    # Convert to RGB if grayscale
                    if len(original_image.shape) == 2:
                        original_image = np.stack([original_image] * 3, axis=-1)
                    
                    # Enhance for better visibility
                    original_image = enhance_image_for_display(original_image)
                else:
                    original_image = np.zeros_like(np.stack([gt_instances] * 3, axis=-1))
            except:
                original_image = np.zeros_like(np.stack([gt_instances] * 3, axis=-1))
            
            # Original image
            axes[0, i*2].imshow(original_image)
            axes[0, i*2].set_title(f'{Path(image_name).stem}\nAvg F1: {img_data["mean"]:.3f}')
            axes[0, i*2].axis('off')
            
            # Ground truth
            axes[0, i*2+1].imshow(gt_instances, cmap='tab20')
            axes[0, i*2+1].set_title(f'GT: {len(np.unique(gt_instances))-1} cells')
            axes[0, i*2+1].axis('off')
            
            # Add worst case label
            axes[0, i*2].text(0.5, -0.1, 'CHALLENGING', 
                             ha='center', va='top', transform=axes[0, i*2].transAxes,
                             fontsize=10, fontweight='bold', color='red')
    
    # Process best images
    for i, (_, img_data) in enumerate(best_images.iterrows()):
        if i >= 5:
            break
            
        image_name = img_data['image']
        gt_instances = data_loader.load_ground_truth_instances(image_name)
        
        if gt_instances is not None:
            # Load original image
            try:
                original_image_path = TEST_IMAGES_PATH / image_name
                if original_image_path.exists():
                    if original_image_path.suffix.lower() in {'.tif', '.tiff'}:
                        original_image = tiff.imread(str(original_image_path))
                    else:
                        original_image = np.array(Image.open(original_image_path))
                    
                    # Convert to RGB if grayscale
                    if len(original_image.shape) == 2:
                        original_image = np.stack([original_image] * 3, axis=-1)
                    
                    # Enhance for better visibility
                    original_image = enhance_image_for_display(original_image)
                else:
                    original_image = np.zeros_like(np.stack([gt_instances] * 3, axis=-1))
            except:
                original_image = np.zeros_like(np.stack([gt_instances] * 3, axis=-1))
            
            # Original image
            axes[1, i*2].imshow(original_image)
            axes[1, i*2].set_title(f'{Path(image_name).stem}\nAvg F1: {img_data["mean"]:.3f}')
            axes[1, i*2].axis('off')
            
            # Ground truth
            axes[1, i*2+1].imshow(gt_instances, cmap='tab20')
            axes[1, i*2+1].set_title(f'GT: {len(np.unique(gt_instances))-1} cells')
            axes[1, i*2+1].axis('off')
            
            # Add best case label
            axes[1, i*2].text(0.5, -0.1, 'EASY', 
                             ha='center', va='top', transform=axes[1, i*2].transAxes,
                             fontsize=10, fontweight='bold', color='green')
    
    plt.tight_layout()
    save_path = save_dir / 'best_worst_images_overall.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

if __name__ == "__main__":
    main()
