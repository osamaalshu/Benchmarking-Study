#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAUNet Ensemble Prediction Script
Based on the saltfish team's approach from NeurIPS 2022 Cell Segmentation Challenge
Combines ResNet50 and Wide-ResNet50 MAUNet models with multi-scale inference
"""

import sys
import os
join = os.path.join
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
import time
from skimage import io, segmentation, morphology, measure, exposure
try:
    # Newer scikit-image uses feature.peak_local_max
    from skimage.feature import peak_local_max
except Exception:
    # Fallback for older scikit-image versions
    from skimage.morphology import local_maxima as peak_local_max
import tifffile as tif
import cv2

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.maunet import create_maunet_model

def sophisticated_postprocessing(prob_background, prob_interior, distance_map):
    """
    Sophisticated post-processing based on the original saltfish team's approach
    Uses watershed, peak detection, and morphological operations
    """
    # Convert probabilities to binary masks
    interior_mask = prob_interior > 0.7
    background_mask = prob_background < 0.3
    cell_mask = interior_mask & background_mask
    
    # Remove small objects
    cell_mask = morphology.remove_small_objects(cell_mask, 16)
    
    if not cell_mask.any():
        return np.zeros_like(cell_mask, dtype=np.uint16)
    
    # Calculate average cell size for adaptive processing
    labeled_cells = measure.label(cell_mask)
    if labeled_cells.max() == 0:
        return np.zeros_like(cell_mask, dtype=np.uint16)
        
    avg_cell_size = int(np.sum(cell_mask) / labeled_cells.max())
    kernel_size = int(np.sqrt(avg_cell_size))
    kernel_size += kernel_size % 2 + 1  # Make odd
    
    # Normalize and smooth distance map (keep float to preserve detail)
    dmax = np.max(distance_map)
    dm = distance_map / (dmax if dmax > 0 else 1.0)
    blurred_distance = cv2.GaussianBlur(dm.astype(np.float32), (kernel_size, kernel_size), 0)
    
    # Find local maxima (cell centers)
    min_distance = max(1, int(kernel_size / 4))
    # Detect peaks on the smoothed distance (boolean mask)
    try:
        # skimage>=0.20 uses coordinates when indices default changed; request mask via labels output
        local_maxima = peak_local_max(blurred_distance, min_distance=min_distance, threshold_rel=0.6, exclude_border=False)
        peak_mask = np.zeros_like(blurred_distance, dtype=bool)
        if local_maxima.size > 0:
            peak_mask[tuple(local_maxima.T)] = True
    except Exception:
        # Fallback where function returns boolean mask
        peak_mask = peak_local_max(blurred_distance, min_distance=min_distance, threshold_rel=0.6)
    
    # Create markers from peaks within cell mask
    peak_mask = peak_mask & cell_mask
    markers = measure.label(peak_mask).astype(np.int32)

    if markers.max() == 0:
        return np.zeros_like(cell_mask, dtype=np.uint16)

    # Use watershed on negative distance to split instances, constrained by cell_mask
    # Convert mask to uint8 for sure
    mask_uint8 = cell_mask.astype(np.uint8)
    # skimage's watershed expects positive basins on -distance
    ws = segmentation.watershed(-blurred_distance, markers=markers, mask=mask_uint8)
    final_result = ws.astype(np.int32)
    
    # Clean up small objects based on average cell size
    final_mask = final_result > 0
    if final_result.max() > 0:
        avg_size = int(np.sum(final_mask) / final_result.max())
        final_mask = morphology.remove_small_objects(final_mask, max(8, int(avg_size / 10)))
    
    return (final_result * final_mask).astype(np.uint16)

def normalize_channel(img, lower=1, upper=99):
    """Normalize image channel using percentile-based scaling"""
    non_zero_vals = img[np.nonzero(img)]
    if len(non_zero_vals) == 0:
        return img.astype(np.uint8)
    
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(
            img, 
            in_range=(percentiles[0], percentiles[1]), 
            out_range='uint8'
        )
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

def load_maunet_models(device, resnet50_path, wideresnet50_path, num_classes=3, input_size=256):
    """Load both ResNet50 and Wide-ResNet50 MAUNet models"""
    
    # Load ResNet50 MAUNet
    model_resnet50 = create_maunet_model(
        num_classes=num_classes,
        input_size=input_size,
        in_channels=3,
        backbone="resnet50"
    ).to(device)
    
    if os.path.exists(resnet50_path):
        checkpoint = torch.load(resnet50_path, map_location=device, weights_only=False)
        model_resnet50.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded ResNet50 MAUNet from {resnet50_path}")
    else:
        raise FileNotFoundError(f"ResNet50 checkpoint not found: {resnet50_path}")
    
    # Load Wide-ResNet50 MAUNet
    model_wide_resnet50 = create_maunet_model(
        num_classes=num_classes,
        input_size=input_size,
        in_channels=3,
        backbone="wide_resnet50"
    ).to(device)
    
    if os.path.exists(wideresnet50_path):
        checkpoint = torch.load(wideresnet50_path, map_location=device, weights_only=False)
        model_wide_resnet50.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded Wide-ResNet50 MAUNet from {wideresnet50_path}")
    else:
        raise FileNotFoundError(f"Wide-ResNet50 checkpoint not found: {wideresnet50_path}")
    
    return model_resnet50, model_wide_resnet50

def ensemble_predict_single_scale(tensor, models, roi_size, sw_batch_size):
    """Predict using ensemble of models at single scale"""
    ensemble_seg_output = None
    ensemble_dist_output = None
    
    for model in models:
        # MAUNet returns (segmentation, distance_transform)
        # Call SWI twice - once for each output head to avoid stitching issues
        seg_output = sliding_window_inference(
            tensor, roi_size, sw_batch_size, 
            lambda x: model(x)[0],  # Segmentation head only
            padding_mode="reflect"
        )
        dist_output = sliding_window_inference(
            tensor, roi_size, sw_batch_size, 
            lambda x: model(x)[1],  # Distance transform head only
            padding_mode="reflect"
        )
        
        if ensemble_seg_output is None:
            ensemble_seg_output = seg_output
            ensemble_dist_output = dist_output
        else:
            ensemble_seg_output += seg_output
            ensemble_dist_output += dist_output
    
    # Average the ensemble predictions
    ensemble_seg_output /= len(models)
    ensemble_dist_output /= len(models)
    
    return ensemble_seg_output, ensemble_dist_output

def main():
    parser = argparse.ArgumentParser('MAUNet Ensemble Prediction with Multi-Scale Inference')
    
    # Dataset parameters
    parser.add_argument('-i', '--input_path', 
                       default='./data/test/images', 
                       type=str, 
                       help='Input images directory')
    parser.add_argument('-o', '--output_path', 
                       default='./test_predictions/maunet_ensemble', 
                       type=str, 
                       help='Output directory for predictions')
    
    # Model parameters
    parser.add_argument('--num_class', default=3, type=int, 
                       help='Number of segmentation classes')
    parser.add_argument('--input_size', default=256, type=int, 
                       help='Model input size')
    parser.add_argument('--multi_scale', action='store_true', 
                       help='Use multi-scale inference (1.0, 1.25, 1.5)')
    parser.add_argument('--use_sophisticated_postprocessing', action='store_true',
                       help='Use sophisticated post-processing like original saltfish team')
    
    # Ensemble model paths
    parser.add_argument('--resnet50_path', type=str,
                       default='./baseline/work_dir/maunet(Normal)_3class/best_Dice_model.pth',
                       help='Path to ResNet50 MAUNet checkpoint')
    parser.add_argument('--wideresnet50_path', type=str,
                       default='./baseline/work_dir/maunet(Wide)_3class/best_Dice_model.pth',
                       help='Path to Wide-ResNet50 MAUNet checkpoint')
    
    args = parser.parse_args()
    
    # Setup
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load images
    img_names = sorted([f for f in os.listdir(input_path) 
                       if f.endswith(('.tif', '.tiff', '.png', '.jpg'))])
    print(f"Found {len(img_names)} images to process")
    
    # Load ensemble models
    print("Loading MAUNet ensemble models...")
    model_resnet50, model_wide_resnet50 = load_maunet_models(
        device, args.resnet50_path, args.wideresnet50_path, args.num_class, args.input_size
    )
    models = [model_resnet50, model_wide_resnet50]
    
    # Set models to eval mode
    for model in models:
        model.eval()
    
    # Inference settings
    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 4
    scales = [1.0, 1.25, 1.5] if args.multi_scale else [1.0]
    
    print(f"Using scales: {scales}")
    print(f"ROI size: {roi_size}")
    print(f"Sliding window batch size: {sw_batch_size}")
    
    # Process each image
    with torch.no_grad():
        for img_name in img_names:
            print(f"\nProcessing: {img_name}")
            t0 = time.time()
            
            # Load image
            img_path = join(input_path, img_name)
            if img_name.endswith(('.tif', '.tiff')):
                img_data = tif.imread(img_path)
            else:
                img_data = io.imread(img_path)
            
            # Normalize image
            if len(img_data.shape) == 2:
                img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
            elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                img_data = img_data[:, :, :3]
            
            pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:, :, i]
                if len(img_channel_i[np.nonzero(img_channel_i)]) > 0:
                    pre_img_data[:, :, i] = normalize_channel(img_channel_i, lower=1, upper=99)
            
            # Convert to tensor with zero-division guard
            mx = np.max(pre_img_data)
            test_npy01 = pre_img_data / (mx if mx > 0 else 1)
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
            
            b, c, h, w = test_tensor.size()
            
            # Initialize ensemble outputs
            ensemble_seg_output = torch.zeros(b, args.num_class, h, w).to(device)
            ensemble_dist_output = torch.zeros(b, 1, h, w).to(device)
            
            # Multi-scale inference
            total_predictions = 0
            
            # Check if image is too large for multi-scale (like original implementation)
            if h * w < 25000000:  # Use multi-scale for smaller images
                for scale in scales:
                    # Resize input
                    scaled_h, scaled_w = int(scale * h), int(scale * w)
                    scaled_tensor = F.interpolate(
                        test_tensor, (scaled_h, scaled_w), 
                        mode='bilinear', align_corners=True
                    )
                    
                    # Ensemble prediction at this scale
                    seg_out, dist_out = ensemble_predict_single_scale(
                        scaled_tensor, models, roi_size, sw_batch_size
                    )
                    
                    # Resize back to original size
                    seg_out = F.interpolate(seg_out, (h, w), mode='bilinear', align_corners=True)
                    dist_out = F.interpolate(dist_out, (h, w), mode='bilinear', align_corners=True)
                    
                    # Accumulate
                    ensemble_seg_output += seg_out
                    ensemble_dist_output += dist_out
                    total_predictions += 1
            else:
                # Single scale for large images
                seg_out, dist_out = ensemble_predict_single_scale(
                    test_tensor, models, roi_size, sw_batch_size
                )
                ensemble_seg_output += seg_out
                ensemble_dist_output += dist_out
                total_predictions += 1
            
            # Average ensemble predictions
            ensemble_seg_output /= total_predictions
            ensemble_dist_output /= total_predictions
            
            # Apply softmax to segmentation output
            ensemble_seg_output = torch.nn.functional.softmax(ensemble_seg_output, dim=1)
            
            # Convert to numpy
            seg_probs = ensemble_seg_output[0].cpu().numpy()  # (C, H, W)
            dist_map = ensemble_dist_output[0, 0].cpu().numpy()  # (H, W)
            
            # Post-processing
            if args.use_sophisticated_postprocessing and args.num_class >= 3:
                # Use sophisticated post-processing like original saltfish team
                prob_background = seg_probs[0]  # Background
                prob_interior = seg_probs[1]    # Interior
                
                final_mask = sophisticated_postprocessing(
                    prob_background, prob_interior, dist_map
                )
            else:
                # Simple post-processing
                # Use interior class (class 1) for segmentation
                prob_map = seg_probs[1] if args.num_class > 1 else seg_probs[0]
                binary_mask = prob_map > 0.5
                
                # Remove small objects and label connected components
                binary_mask = morphology.remove_small_objects(
                    morphology.remove_small_holes(binary_mask), 16
                )
                final_mask = measure.label(binary_mask).astype(np.uint16)
            
            # Save prediction
            output_filename = join(output_path, img_name.split('.')[0] + '_label.tiff')
            tif.imwrite(output_filename, final_mask, compression='zlib')
            
            t1 = time.time()
            print(f'✅ Prediction saved: {output_filename}')
            print(f'   Image size: {pre_img_data.shape}')
            print(f'   Processing time: {t1-t0:.2f}s')
            print(f'   Unique labels: {len(np.unique(final_mask))-1}')  # -1 for background
    
    print(f"\n🎉 Ensemble prediction completed!")
    print(f"Results saved in: {output_path}")

if __name__ == "__main__":
    main()
