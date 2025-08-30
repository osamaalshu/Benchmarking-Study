#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction Script for Error-Aware MAUNet

This script performs inference using trained Error-Aware MAUNet models with
enhanced post-processing including seeded watershed segmentation.

Supports both single model and ensemble inference with test-time augmentation.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import tifffile as tif
from skimage import measure, morphology
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from error_aware_maunet import create_error_aware_maunet_model, ErrorAwareMAUNetEnsemble
from inference_pipeline import ErrorAwareInferencePipeline, create_ensemble_inference_pipeline
from utils.compute_metric import eval_tp_fp_fn, dice


def compute_metrics_for_image(pred_mask, gt_mask, threshold=0.5):
    """
    Simple wrapper to compute metrics for a single image
    
    Args:
        pred_mask: Predicted instance mask
        gt_mask: Ground truth instance mask
        threshold: IoU threshold for matching
        
    Returns:
        Dictionary of computed metrics
    """
    try:
        # Compute basic metrics
        tp, fp, fn = eval_tp_fp_fn(gt_mask, pred_mask, threshold=threshold)
        
        # Compute derived metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Compute Dice coefficient for binary masks
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        dice_score = dice(gt_binary, pred_binary)
        
        # Compute IoU
        intersection = np.sum(pred_binary & gt_binary)
        union = np.sum(pred_binary | gt_binary)
        iou = intersection / union if union > 0 else 0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'dice': dice_score,
            'iou': iou
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {
            'tp': 0, 'fp': 0, 'fn': 0,
            'precision': 0, 'recall': 0, 'f1': 0,
            'dice': 0, 'iou': 0
        }


def load_and_preprocess_image(image_path: str, target_size: int = 256) -> torch.Tensor:
    """
    Load and preprocess image for inference
    
    Args:
        image_path: Path to input image
        target_size: Target size for resizing
        
    Returns:
        preprocessed_image: (C, H, W) tensor ready for inference
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Convert to tensor and add channel dimension
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # (C, H, W)
    
    # Resize if necessary
    if image_tensor.shape[1] != target_size or image_tensor.shape[2] != target_size:
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0), 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
    
    return image_tensor


def save_predictions(
    predictions: dict, 
    output_dir: str, 
    image_name: str, 
    save_all_outputs: bool = False
):
    """
    Save prediction results to files
    
    Args:
        predictions: Dictionary containing prediction arrays
        output_dir: Directory to save outputs
        image_name: Base name for output files
        save_all_outputs: Whether to save all intermediate outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(image_name)[0]
    
    # Always save instance mask
    if 'instance_mask' in predictions:
        instance_path = os.path.join(output_dir, f"{base_name}_instances.tiff")
        tif.imwrite(instance_path, predictions['instance_mask'].astype(np.uint16), compression='zlib')
    
    if save_all_outputs:
        # Save segmentation probabilities
        if 'segmentation' in predictions:
            seg_logits = predictions['segmentation']
            if seg_logits.ndim == 3 and seg_logits.shape[0] > 1:
                # Multi-class: save foreground probability
                probs = np.exp(seg_logits) / np.sum(np.exp(seg_logits), axis=0, keepdims=True)
                foreground_prob = np.sum(probs[1:], axis=0)
            else:
                # Binary: apply sigmoid
                foreground_prob = 1 / (1 + np.exp(-seg_logits.squeeze()))
            
            prob_path = os.path.join(output_dir, f"{base_name}_probabilities.tiff")
            tif.imwrite(prob_path, foreground_prob.astype(np.float32), compression='zlib')
        
        # Save distance transform
        if 'distance_transform' in predictions:
            dist_path = os.path.join(output_dir, f"{base_name}_distance.tiff")
            tif.imwrite(
                dist_path, 
                predictions['distance_transform'].squeeze().astype(np.float32), 
                compression='zlib'
            )
        
        # Save centroid heatmap
        if 'centroid_heatmap' in predictions:
            centroid_path = os.path.join(output_dir, f"{base_name}_centroids.tiff")
            tif.imwrite(
                centroid_path, 
                predictions['centroid_heatmap'].squeeze().astype(np.float32), 
                compression='zlib'
            )


def evaluate_predictions(pred_dir: str, gt_dir: str, output_csv: str):
    """
    Evaluate predictions against ground truth
    
    Args:
        pred_dir: Directory containing prediction masks
        gt_dir: Directory containing ground truth masks
        output_csv: Path to save evaluation results
    """
    results = []
    
    pred_files = list(Path(pred_dir).glob("*_instances.tiff"))
    
    for pred_file in tqdm(pred_files, desc="Evaluating predictions"):
        # Find corresponding ground truth
        base_name = pred_file.stem.replace("_instances", "")
        gt_file = Path(gt_dir) / f"{base_name}_label.png"
        
        if not gt_file.exists():
            print(f"Warning: Ground truth not found for {pred_file}")
            continue
        
        # Load masks
        pred_mask = tif.imread(pred_file)
        gt_image = Image.open(gt_file)
        gt_mask = np.array(gt_image)
        
        # Convert semantic GT to instance mask if needed
        if len(np.unique(gt_mask)) <= 3:  # Semantic segmentation
            # Convert to binary and find connected components
            binary_gt = (gt_mask > 0).astype(np.uint8)
            gt_instances = measure.label(binary_gt, connectivity=2)
        else:
            gt_instances = gt_mask
        
        # Compute metrics
        try:
            metrics = compute_metrics_for_image(pred_mask, gt_instances)
            metrics['image_name'] = base_name
            results.append(metrics)
        except Exception as e:
            print(f"Error computing metrics for {base_name}: {e}")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        # Print summary statistics
        print(f"\nEvaluation Results (n={len(results)}):")
        print(f"Average Dice: {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
        print(f"Average IoU: {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")
        if 'aji' in df.columns:
            print(f"Average AJI: {df['aji'].mean():.4f} ± {df['aji'].std():.4f}")
        print(f"Results saved to: {output_csv}")
    else:
        print("No valid results to save.")


def main():
    parser = argparse.ArgumentParser("Error-Aware MAUNet Prediction")
    
    # Input/Output paths
    parser.add_argument("--input_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save predictions")
    parser.add_argument("--model_path", help="Path to single model checkpoint")
    parser.add_argument("--ensemble_path", help="Path to ensemble model")
    parser.add_argument("--gt_dir", help="Ground truth directory for evaluation")
    
    # Model configuration
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "wide_resnet50"])
    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--enable_auxiliary_tasks", action="store_true", default=True)
    parser.add_argument("--centroid_sigma", type=float, default=2.0)
    
    # Inference configuration
    parser.add_argument("--use_tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--use_watershed", action="store_true", default=True, help="Use seeded watershed post-processing")
    parser.add_argument("--batch_processing", action="store_true", help="Process images in batches")
    parser.add_argument("--save_all_outputs", action="store_true", help="Save all intermediate outputs")
    
    # Watershed configuration
    parser.add_argument("--min_seed_distance", type=int, default=10)
    parser.add_argument("--min_seed_prominence", type=float, default=0.1)
    parser.add_argument("--watershed_threshold", type=float, default=0.5)
    parser.add_argument("--min_object_size", type=int, default=16)
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup inference pipeline
    if args.ensemble_path:
        print("Loading ensemble model...")
        # Load ensemble
        ensemble_data = torch.load(args.ensemble_path, map_location=device)
        model_configs = []
        checkpoint_paths = []
        
        for backbone in ensemble_data['backbones']:
            config = ensemble_data['model_config'].copy()
            config['backbone'] = backbone
            model_configs.append(config)
        
        # Find individual model checkpoints
        for backbone in ensemble_data['backbones']:
            backbone_dir = os.path.dirname(args.ensemble_path)
            checkpoint_path = os.path.join(backbone_dir, f"error_aware_maunet_{backbone}_3class", "best_Dice_model.pth")
            checkpoint_paths.append(checkpoint_path)
        
        # Watershed configuration
        watershed_config = {
            'min_seed_distance': args.min_seed_distance,
            'min_seed_prominence': args.min_seed_prominence,
            'watershed_threshold': args.watershed_threshold,
            'min_object_size': args.min_object_size,
        }
        
        pipeline = create_ensemble_inference_pipeline(
            checkpoint_paths=checkpoint_paths,
            model_configs=model_configs,
            device=device,
            use_tta=args.use_tta,
            use_watershed=args.use_watershed,
            watershed_config=watershed_config
        )
        
    elif args.model_path:
        print("Loading single model...")
        # Load single model
        model_config = {
            'num_classes': args.num_classes,
            'input_size': args.input_size,
            'in_channels': 3,
            'backbone': args.backbone,
            'enable_auxiliary_tasks': args.enable_auxiliary_tasks,
            'centroid_sigma': args.centroid_sigma
        }
        
        model = create_error_aware_maunet_model(**model_config)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        watershed_config = {
            'min_seed_distance': args.min_seed_distance,
            'min_seed_prominence': args.min_seed_prominence,
            'watershed_threshold': args.watershed_threshold,
            'min_object_size': args.min_object_size,
        }
        
        pipeline = ErrorAwareInferencePipeline(
            model=model,
            device=device,
            use_tta=args.use_tta,
            use_watershed=args.use_watershed,
            watershed_config=watershed_config
        )
        
    else:
        raise ValueError("Must provide either --model_path or --ensemble_path")
    
    # Get input images
    input_dir = Path(args.input_dir)
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Load and preprocess image
            image_tensor = load_and_preprocess_image(str(image_file), args.input_size)
            
            # Run inference
            predictions = pipeline.predict(image_tensor)
            
            # Save predictions
            save_predictions(
                predictions, 
                args.output_dir, 
                image_file.name, 
                args.save_all_outputs
            )
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            continue
    
    print(f"Predictions saved to: {args.output_dir}")
    
    # Evaluate if ground truth provided
    if args.gt_dir:
        print("Evaluating predictions...")
        eval_csv = os.path.join(args.output_dir, "evaluation_results.csv")
        evaluate_predictions(args.output_dir, args.gt_dir, eval_csv)


if __name__ == "__main__":
    main()
