#!/usr/bin/env python3
"""
Debug script to examine split detection in detail
"""

import json
import numpy as np
from error_analysis_pipeline import ErrorAnalyzer, _iou_matrix

def debug_split_detection():
    # Load some sample data
    with open('results/error_analysis/detailed_results.json', 'r') as f:
        data = json.load(f)
    
    # Get first few images
    sample_images = list(data.keys())[:5]
    
    print("DEBUGGING SPLIT DETECTION")
    print("=" * 50)
    
    for image_name in sample_images:
        print(f"\nImage: {image_name}")
        
        # Check if this image has any splits in the results
        image_data = data[image_name]
        for model_name in ['maunet_ensemble', 'nnunet', 'unet']:
            if model_name in image_data:
                model_data = image_data[model_name]
                splits = model_data.get('splits', 0)
                merges = model_data.get('merges', 0)
                print(f"  {model_name}: splits={splits}, merges={merges}")
    
    print("\n" + "=" * 50)
    print("ANALYZING SPLIT DETECTION PARAMETERS")
    print("=" * 50)
    
    # Test different parameters
    test_params = [
        (0.10, 0.10, 5),   # More lenient
        (0.15, 0.15, 8),   # Medium
        (0.20, 0.20, 10),  # Current
        (0.25, 0.25, 15),  # More strict
    ]
    
    print("Testing different split detection parameters:")
    print("Format: (alpha_iogt, beta_iop, min_overlap_px)")
    
    for alpha, beta, min_overlap in test_params:
        print(f"\nParameters: ({alpha}, {beta}, {min_overlap})")
        
        # Count how many images would have splits with these parameters
        split_count = 0
        total_images = 0
        
        for image_name in list(data.keys())[:10]:  # Test first 10 images
            image_data = data[image_name]
            for model_name in ['maunet_ensemble', 'nnunet', 'unet']:
                if model_name in image_data:
                    total_images += 1
                    # We can't easily test without the actual instance maps,
                    # but we can check if the current results suggest splits might exist
                    pass
        
        print(f"  Would need actual instance maps to test properly")

def test_split_logic():
    """Test the split detection logic with synthetic data"""
    print("\n" + "=" * 50)
    print("TESTING SPLIT DETECTION LOGIC")
    print("=" * 50)
    
    # Create synthetic data that should produce splits
    # GT: 1 cell, Pred: 2 cells (split)
    gt = np.zeros((100, 100), dtype=np.int32)
    gt[20:80, 20:80] = 1  # One large cell
    
    pred = np.zeros((100, 100), dtype=np.int32)
    pred[20:50, 20:80] = 1  # Left half
    pred[50:80, 20:80] = 2  # Right half
    
    print("Synthetic data:")
    print(f"GT unique values: {np.unique(gt)}")
    print(f"Pred unique values: {np.unique(pred)}")
    
    # Test the split detection
    analyzer = ErrorAnalyzer()
    
    # Get intersection matrix
    gt_ids, pr_ids, iou_matrix, intersections, gt_areas, pr_areas = _iou_matrix(gt, pred)
    
    print(f"\nIntersection matrix shape: {intersections.shape}")
    print(f"GT areas: {gt_areas}")
    print(f"Pred areas: {pr_areas}")
    print(f"Intersections:\n{intersections}")
    
    # Test different parameters
    for alpha, beta, min_overlap in [(0.10, 0.10, 5), (0.20, 0.20, 10), (0.30, 0.30, 15)]:
        split_children, merge_parents, splits_affected, merges_affected = analyzer._count_splits_merges(
            intersections, gt_areas, pr_areas, alpha_iogt=alpha, beta_iop=beta, min_overlap_px=min_overlap
        )
        print(f"\nParameters ({alpha}, {beta}, {min_overlap}):")
        print(f"  Split children: {split_children}")
        print(f"  Merge parents: {merge_parents}")
        print(f"  Splits affected: {splits_affected}")
        print(f"  Merges affected: {merges_affected}")

if __name__ == "__main__":
    debug_split_detection()
    test_split_logic()
