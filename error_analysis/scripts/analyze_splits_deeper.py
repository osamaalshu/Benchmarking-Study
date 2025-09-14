#!/usr/bin/env python3
"""
Deeper analysis of why splits are 0.0
"""

import json
import numpy as np

def analyze_intersection_patterns():
    """Analyze the intersection patterns to understand split detection"""
    
    # Load the detailed results
    with open('results/error_analysis/detailed_results.json', 'r') as f:
        data = json.load(f)
    
    print("DEEP ANALYSIS OF SPLIT DETECTION")
    print("=" * 60)
    
    # Look for any non-zero splits or merges
    total_splits = 0
    total_merges = 0
    images_with_splits = 0
    images_with_merges = 0
    
    for image_name, image_data in data.items():
        for model_name in ['maunet_ensemble', 'nnunet', 'unet', 'maunet_wide', 'maunet_resnet50', 'lstmunet', 'sac']:
            if model_name in image_data:
                model_data = image_data[model_name]
                splits = model_data.get('splits', 0)
                merges = model_data.get('merges', 0)
                
                total_splits += splits
                total_merges += merges
                
                if splits > 0:
                    images_with_splits += 1
                    print(f"SPLIT FOUND: {image_name} - {model_name}: {splits} splits")
                
                if merges > 0:
                    images_with_merges += 1
                    if merges > 5:  # Only show significant merges
                        print(f"MERGE FOUND: {image_name} - {model_name}: {merges} merges")
    
    print(f"\nSUMMARY:")
    print(f"Total splits across all images/models: {total_splits}")
    print(f"Total merges across all images/models: {total_merges}")
    print(f"Images with at least one split: {images_with_splits}")
    print(f"Images with at least one merge: {images_with_merges}")
    
    # Check if this is consistent with the original table
    print(f"\nCOMPARISON WITH ORIGINAL TABLE:")
    original_splits = {
        'MAUNet-Ens': 1.3,
        'MAUNet-Wide': 1.3,
        'MAUNet-R50': 0.9,
        'nnU-Net': 3.6,
        'U-Net': 3.1,
        'LSTM-UNet': 1.7,
        'SAC': 0.1
    }
    
    print("Original table average splits per model:")
    for model, avg_splits in original_splits.items():
        print(f"  {model}: {avg_splits}")
    
    print(f"\nCurrent analysis total splits: {total_splits}")
    print(f"Expected total splits (if same as original): {sum(original_splits.values()) * 100}")
    
    # Check if there might be a methodological difference
    print(f"\nPOSSIBLE REASONS FOR 0.0 SPLITS:")
    print("1. Models have improved significantly since original analysis")
    print("2. Different analysis parameters (thresholds, filtering)")
    print("3. Different data preprocessing (boundary removal, size filtering)")
    print("4. Different evaluation methodology")
    print("5. Bug in current implementation")
    
    # Check the actual data structure
    print(f"\nDATA STRUCTURE ANALYSIS:")
    sample_image = list(data.keys())[0]
    sample_model = 'maunet_ensemble'
    sample_data = data[sample_image][sample_model]
    
    print(f"Sample image: {sample_image}")
    print(f"Sample model: {sample_model}")
    print(f"Available fields: {list(sample_data.keys())}")
    print(f"GT count: {sample_data.get('gt_count', 'N/A')}")
    print(f"Pred count: {sample_data.get('pred_count', 'N/A')}")
    print(f"True positives: {sample_data.get('true_positives', 'N/A')}")
    print(f"False negatives: {sample_data.get('false_negatives', 'N/A')}")
    print(f"False positives: {sample_data.get('false_positives', 'N/A')}")
    print(f"Splits: {sample_data.get('splits', 'N/A')}")
    print(f"Merges: {sample_data.get('merges', 'N/A')}")

def test_parameter_sensitivity():
    """Test how sensitive split detection is to parameter changes"""
    print(f"\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Test different parameter combinations
    test_params = [
        (0.05, 0.05, 1),   # Very lenient
        (0.10, 0.10, 5),   # Lenient
        (0.15, 0.15, 8),   # Medium
        (0.20, 0.20, 10),  # Current
        (0.25, 0.25, 15),  # Strict
        (0.30, 0.30, 20),  # Very strict
    ]
    
    print("Parameter combinations to test:")
    print("Format: (alpha_iogt, beta_iop, min_overlap_px)")
    for alpha, beta, min_overlap in test_params:
        print(f"  ({alpha}, {beta}, {min_overlap})")
    
    print("\nNote: To test these parameters, we would need:")
    print("1. Access to the actual instance maps (GT and predictions)")
    print("2. Re-run the split detection with different parameters")
    print("3. Compare the results across parameter sets")

if __name__ == "__main__":
    analyze_intersection_patterns()
    test_parameter_sensitivity()
