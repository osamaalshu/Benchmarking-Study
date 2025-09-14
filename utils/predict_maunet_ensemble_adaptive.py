#!/usr/bin/env python3
"""
Error-Aware MAUNet Ensemble Prediction with Adaptive Post-Processing
"""

import sys
import os
join = os.path.join
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
import time
from skimage import io, segmentation, morphology, measure, exposure
import tifffile as tif
from models.maunet_error_aware import create_maunet_error_aware_model

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

def load_ensemble_models(device, resnet50_path, wideresnet_path):
    """Load both error-aware models"""
    model1 = create_maunet_error_aware_model(
        num_classes=3, input_size=256, in_channels=3, backbone="resnet50"
    ).to(device)
    ckpt1 = torch.load(resnet50_path, map_location=device)
    model1.load_state_dict(ckpt1['model_state_dict'])
    
    model2 = create_maunet_error_aware_model(
        num_classes=3, input_size=256, in_channels=3, backbone="wide_resnet50"
    ).to(device)
    ckpt2 = torch.load(wideresnet_path, map_location=device)
    model2.load_state_dict(ckpt2['model_state_dict'])
    
    return model1, model2

def estimate_image_density(seg_probs, threshold=0.5):
    """Estimate cell density for adaptive thresholding"""
    interior_prob = seg_probs[0, 1] if seg_probs.shape[1] > 1 else seg_probs[0, 0]
    binary_mask = interior_prob > threshold
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=16)
    
    # Count connected components as rough cell estimate
    labeled = measure.label(binary_mask)
    num_objects = labeled.max()
    image_area = interior_prob.shape[0] * interior_prob.shape[1]
    density = num_objects / (image_area / 1e6)  # cells per megapixel
    
    return density, num_objects

def adaptive_parameters(density):
    """Get adaptive parameters based on estimated density"""
    if density > 300:
        return {'threshold': 0.7, 'min_size': 32, 'approach': 'very_dense'}
    elif density > 150:
        return {'threshold': 0.65, 'min_size': 24, 'approach': 'dense'}
    elif density > 50:
        return {'threshold': 0.55, 'min_size': 16, 'approach': 'moderate'}
    else:
        return {'threshold': 0.5, 'min_size': 16, 'approach': 'sparse'}

def ensemble_predict(models, tensor, roi_size, sw_batch_size):
    """Get ensemble predictions from both models"""
    seg_outputs = []
    for model in models:
        # Error-aware MAUNet returns (seg, dist, centroid)
        output = sliding_window_inference(
            tensor, roi_size, sw_batch_size, 
            lambda x: model(x)[0],  # Only take segmentation output
            padding_mode="reflect"
        )
        seg_outputs.append(output)
    
    # Average the segmentation predictions
    ensemble_output = sum(seg_outputs) / len(seg_outputs)
    return ensemble_output

def main():
    parser = argparse.ArgumentParser('Error-Aware MAUNet Ensemble')
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, required=True)
    parser.add_argument('--resnet50_path', type=str, 
                       default='./baseline/work_dir/maunet_error_aware_resnet50_3class/best_Dice_model.pth')
    parser.add_argument('--wideresnet_path', type=str,
                       default='./baseline/work_dir/maunet_error_aware_wideresnet_3class/best_Dice_model.pth')
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--sw_batch_size', default=4, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    img_names = sorted([f for f in os.listdir(args.input_path) 
                       if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])

    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    models = load_ensemble_models(device, args.resnet50_path, args.wideresnet_path)
    for model in models:
        model.eval()

    roi_size = (args.input_size, args.input_size)
    
    with torch.no_grad():
        for img_name in img_names:
            print(f"Processing: {img_name}")
            
            # Load and preprocess image
            if img_name.endswith(('.tif', '.tiff')):
                img_data = tif.imread(join(args.input_path, img_name))
            else:
                img_data = io.imread(join(args.input_path, img_name))

            if len(img_data.shape) == 2:
                img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
            elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                img_data = img_data[:, :, :3]
                
            pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:, :, i]
                if len(img_channel_i[np.nonzero(img_channel_i)]) > 0:
                    pre_img_data[:, :, i] = normalize_channel(img_channel_i, lower=1, upper=99)

            t0 = time.time()
            test_npy01 = pre_img_data / np.max(pre_img_data) if np.max(pre_img_data) > 0 else pre_img_data
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0, 3, 1, 2).float().to(device)

            # Get ensemble predictions
            ensemble_output = ensemble_predict(models, test_tensor, roi_size, args.sw_batch_size)
            
            # Convert to probabilities
            seg_probs = torch.softmax(ensemble_output, dim=1).cpu().numpy()
            
            # Estimate density for adaptive parameters
            density, _ = estimate_image_density(seg_probs)
            params = adaptive_parameters(density)
            
            # Apply adaptive post-processing
            interior_prob = seg_probs[0, 1] if seg_probs.shape[1] > 1 else seg_probs[0, 0]
            binary_mask = interior_prob > params['threshold']
            binary_mask = morphology.remove_small_objects(binary_mask, min_size=params['min_size'])
            binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=64)
            result = measure.label(binary_mask).astype(np.uint16)
            
            # Save result
            output_name = img_name.split('.')[0] + '_label.tiff'
            tif.imwrite(join(args.output_path, output_name), result, compression='zlib')
            
            t1 = time.time()
            num_cells = result.max()
            
            print(f"  Approach: {params['approach']} (density: {density:.1f} cells/MP)")
            print(f"  Result: {num_cells} cells | time={t1-t0:.2f}s")

    print("Processing complete!")

if __name__ == "__main__":
    main()