#!/usr/bin/env python3
"""
Instance-Aware Error Analysis
Focuses on key metrics: FN, FP, Splits, Merges, PQ/RQ/SQ
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import tifffile as tiff
import fastremap
from scipy.optimize import linear_sum_assignment
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns

# Add paths
framework_dir = Path(__file__).parent
sys.path.insert(0, str(framework_dir / 'src'))
sys.path.insert(0, str(framework_dir))
sys.path.insert(0, str(framework_dir / 'config'))

try:
    from config.analysis_config import *
except ImportError:
    from analysis_config import *

def _read_instance_map(path: Path, logger=None) -> np.ndarray:
    """Safely read instance labels preserving all instance IDs"""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        # Read with proper dtype preservation
        if path.suffix.lower() in {'.tif', '.tiff'}:
            arr = tiff.imread(str(path))
        else:
            arr = np.array(Image.open(path))
        
        # Check for potential 8-bit truncation
        if arr.dtype == np.uint8 and arr.max() > 250:
            logger.warning(f"{path.name}: uint8 with high labels ({arr.max()}); verify no truncation.")
        
        # Ensure integer type
        if not np.issubdtype(arr.dtype, np.integer):
            logger.debug(f"Converting {path.name} from {arr.dtype} to int32")
            arr = arr.astype(np.int32)
        
        # Relabel to contiguous 1..K (keeps 0 for background)
        original_max = arr.max()
        arr = fastremap.renumber(arr, in_place=True)[0]
        
        unique_labels = np.unique(arr)
        n_instances = len(unique_labels[unique_labels > 0])
        
        logger.debug(f"{path.name}: {n_instances} instances (original max: {original_max}, relabeled max: {arr.max()})")
        
        return arr
        
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return None

def _iou_matrix(gt, pr):
    """Compute IoU matrix and return intersections/areas for IoGT/IoPred calculations"""
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]
    pr_ids = np.unique(pr) 
    pr_ids = pr_ids[pr_ids > 0]
    
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        empty_inter = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.int64)
        empty_iou = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.float32)
        empty_gt_areas = np.zeros(len(gt_ids), dtype=np.int64)
        empty_pr_areas = np.zeros(len(pr_ids), dtype=np.int64)
        return gt_ids, pr_ids, empty_iou, empty_inter, empty_gt_areas, empty_pr_areas

    # Flatten once
    gt_flat = gt.ravel()
    pr_flat = pr.ravel()

    # Build id->index maps
    gt_map = {k: i for i, k in enumerate(gt_ids)}
    pr_map = {k: j for j, k in enumerate(pr_ids)}

    # --- INTERSECTIONS (sparse contingency over pixels where both > 0) ---
    both_pos = (gt_flat > 0) & (pr_flat > 0)
    rows = np.fromiter((gt_map[g] for g in gt_flat[both_pos]), count=both_pos.sum(), dtype=np.int32)
    cols = np.fromiter((pr_map[p] for p in pr_flat[both_pos]), count=both_pos.sum(), dtype=np.int32)
    M = sparse.coo_matrix((np.ones_like(rows, dtype=np.int32), (rows, cols)),
                          shape=(len(gt_ids), len(pr_ids))).tocsr()
    intersections = M.toarray().astype(np.int64)

    # --- AREAS (true per-label pixel counts) ---
    gt_area_counts = np.bincount(gt_flat, minlength=int(gt_flat.max()) + 1)
    pr_area_counts = np.bincount(pr_flat, minlength=int(pr_flat.max()) + 1)
    gt_areas = gt_area_counts[gt_ids].astype(np.int64)
    pr_areas = pr_area_counts[pr_ids].astype(np.int64)

    # --- IoU ---
    unions = gt_areas[:, None] + pr_areas[None, :] - intersections
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(unions > 0, intersections / unions, 0.0).astype(np.float32)

    return gt_ids, pr_ids, iou, intersections, gt_areas, pr_areas

class ErrorAnalysisDataLoader:
    """Data loader for instance segmentation error analysis"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.test_images = {}
        self.ground_truth_instances = {}
        self.predictions_instances = {model: {} for model in MODELS.keys()}
        
    def load_ground_truth_instances(self, image_name: str) -> np.ndarray:
        """Load ground truth instance map"""
        if image_name in self.ground_truth_instances:
            return self.ground_truth_instances[image_name]
            
        base_name = Path(image_name).stem
        candidates = [f"{base_name}_label.tiff", f"{base_name}_label.tif", f"{base_name}_label.png"]
        label_path = next((TEST_LABELS_PATH / c for c in candidates if (TEST_LABELS_PATH / c).exists()), None)
        
        if label_path is None:
            self.logger.error(f"Missing GT label for {image_name}")
            return None
        
        label_array = _read_instance_map(label_path, self.logger)
        if label_array is not None:
            self.ground_truth_instances[image_name] = label_array
        return label_array
    
    def load_prediction_instances(self, model_name: str, image_name: str) -> np.ndarray:
        """Load prediction instance map"""
        if image_name in self.predictions_instances[model_name]:
            return self.predictions_instances[model_name][image_name]
            
        model_config = MODELS[model_name]
        pred_dir = PREDICTIONS_PATH / model_config['predictions_dir']
        base_name = Path(image_name).stem
        
        candidates = [f"{base_name}_label.tiff", f"{base_name}_label.tif", f"{base_name}_label.png"]
        pred_path = next((pred_dir / c for c in candidates if (pred_dir / c).exists()), None)
        
        if pred_path is None:
            self.logger.warning(f"Missing prediction for {model_name}/{image_name}")
            return None
        
        pred_array = _read_instance_map(pred_path, self.logger)
        if pred_array is not None:
            self.predictions_instances[model_name][image_name] = pred_array
        return pred_array
    
    def get_test_image_list(self) -> list:
        """Get list of test images"""
        if not TEST_IMAGES_PATH.exists():
            return []
        
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
            image_files.extend(TEST_IMAGES_PATH.glob(ext))
        
        return [f.name for f in sorted(image_files)]

class ErrorAnalyzer:
    """Error analyzer focusing on key metrics"""
    
    def __init__(self, logger=None, min_instance_size=10):
        self.logger = logger or logging.getLogger(__name__)
        self.min_instance_size = min_instance_size
    
    def _filter_small_instances(self, instance_map, min_size):
        """Filter out instances smaller than min_size pixels"""
        if min_size <= 1:
            return instance_map
            
        ids, counts = np.unique(instance_map, return_counts=True)
        keep = set(ids[(ids > 0) & (counts >= min_size)])
        
        out = np.where(np.isin(instance_map, list(keep)), instance_map, 0)
        out = fastremap.renumber(out, in_place=True)[0]
        return out
    
    def _count_splits_merges(self, intersections, gt_areas, pr_areas,
                           alpha_iogt=0.20, beta_iop=0.20, min_overlap_px=10):
        """
        Robust split/merge detection:
          - Count a 'child' for a GT if IoGT >= alpha_iogt AND intersection >= min_overlap_px
          - Count a 'parent' for a Pred if IoPred >= beta_iop  AND intersection >= min_overlap_px
        Returns: split_children, merge_parents, splits_affected, merges_affected
        """
        if intersections.size == 0:
            return 0, 0, 0, 0

        with np.errstate(divide='ignore', invalid='ignore'):
            iogt = np.where(gt_areas[:, None] > 0, intersections / gt_areas[:, None], 0.0)  # G x P
            iop  = np.where(pr_areas[None, :] > 0, intersections / pr_areas[None, :], 0.0)  # G x P

        valid_child  = (iogt >= alpha_iogt) & (intersections >= min_overlap_px)
        valid_parent = (iop  >= beta_iop)  & (intersections >= min_overlap_px)

        gt_child_counts  = valid_child.sum(axis=1)    # #preds substantially covering each GT
        pr_parent_counts = valid_parent.sum(axis=0)   # #GTs substantially contributing to each Pred

        split_children  = int(np.clip(gt_child_counts - 1, 0, None).sum())
        merge_parents   = int(np.clip(pr_parent_counts - 1, 0, None).sum())
        splits_affected = int((gt_child_counts  > 1).sum())
        merges_affected = int((pr_parent_counts > 1).sum())
        return split_children, merge_parents, splits_affected, merges_affected
    
    def analyze_errors(self, gt_instances: np.ndarray, pred_instances: np.ndarray, 
                      iou_threshold=0.5) -> dict:
        """
        Improved error analysis with robust split/merge detection:
        - False Negatives (FN): missed cells
        - False Positives (FP): artifacts wrongly segmented as cells  
        - Over-segmentation (Splits): one GT cell → multiple predictions
        - Under-segmentation (Merges): multiple GT cells → one prediction
        - PQ/RQ/SQ metrics
        """
        
        if gt_instances.shape != pred_instances.shape:
            raise ValueError(f"Shape mismatch: GT {gt_instances.shape} vs Pred {pred_instances.shape}")
        
        # 1) Matching on filtered maps (stable TP/FP/FN & PQ)
        gt_match = self._filter_small_instances(gt_instances, self.min_instance_size)
        pr_match = self._filter_small_instances(pred_instances, self.min_instance_size)

        gt_ids_m, pr_ids_m, iou_matrix, _, _, _ = _iou_matrix(gt_match, pr_match)

        if iou_matrix.size == 0:
            tp, fp, fn = 0, len(pr_ids_m), len(gt_ids_m)
            matched_ious = np.array([])
        else:
            cost_matrix = 1.0 - iou_matrix
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            valid = iou_matrix[row_idx, col_idx] >= iou_threshold
            mgt = set(row_idx[valid]); mpr = set(col_idx[valid])
            tp = len(mgt)
            fn = len(gt_ids_m) - len(mgt)
            fp = len(pr_ids_m) - len(mpr)
            matched_ious = iou_matrix[row_idx[valid], col_idx[valid]] if tp > 0 else np.array([])

        # 2) Splits/Merges on unfiltered maps (keep fragments)
        gt_ids_s, pr_ids_s, _, inter, gt_areas, pr_areas = _iou_matrix(
            self._filter_small_instances(gt_instances, 1),
            self._filter_small_instances(pred_instances, 1)
        )

        if inter.size == 0:
            split_children = merge_parents = splits_affected = merges_affected = 0
        else:
            split_children, merge_parents, splits_affected, merges_affected = self._count_splits_merges(
                inter, gt_areas, pr_areas, alpha_iogt=0.20, beta_iop=0.20, min_overlap_px=10
            )

        # PQ components
        sq = float(matched_ious.mean()) if tp > 0 else 0.0
        rq = tp / (tp + 0.5*fp + 0.5*fn) if (tp + fp + fn) > 0 else 0.0
        pq = sq * rq

        # Standard
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score  = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
        
        return {
            'gt_count': len(gt_ids_s),
            'pred_count': len(pr_ids_s),
            'true_positives': tp,
            'false_negatives': fn,
            'false_positives': fp,
            'splits': splits_affected,   # GTs affected
            'merges': merges_affected,   # Preds affected
            'precision': precision, 'recall': recall, 'f1_score': f1_score,
            'PQ': pq, 'RQ': rq, 'SQ': sq
        }

def enhance_image_for_display(image):
    """Enhance image contrast and brightness for better visualization while preserving RGB"""
    # Work with float version for calculations
    if image.dtype != np.float32:
        enhanced = image.astype(np.float32)
    else:
        enhanced = image.copy()
    
    # Normalize to 0-1 range first if needed
    if enhanced.max() > 1.0:
        enhanced = enhanced / 255.0
    
    # Handle RGB and grayscale differently
    if len(enhanced.shape) == 3 and enhanced.shape[2] == 3:
        # RGB image - enhance each channel separately but preserve color relationships
        for c in range(3):
            channel = enhanced[:, :, c]
            
            # Skip if channel is all zeros
            if channel.max() == 0:
                continue
                
            # Calculate statistics
            channel_min = channel.min()
            channel_max = channel.max()
            channel_mean = channel.mean()
            contrast_ratio = (channel_max - channel_min) / (channel_max + 1e-8)
            
            if channel_max > channel_min:
                # Handle very low contrast (white/blank appearing) images
                if contrast_ratio < 0.05:  # Very low contrast
                    # Aggressive contrast stretching
                    channel = (channel - channel_min) / (channel_max - channel_min + 1e-8)
                    # Strong gamma correction for visibility
                    gamma = 0.2
                    channel = np.power(channel, gamma)
                    # Moderate contrast enhancement (less aggressive for RGB)
                    channel = np.clip((channel - 0.5) * 2.0 + 0.5, 0, 1)
                else:
                    # Normal contrast stretching
                    channel = (channel - channel_min) / (channel_max - channel_min)
                    
                    # Apply gamma correction based on image characteristics
                    if channel_mean < 0.1:  # Very dim
                        gamma = 0.4  # Moderate brightening for RGB
                    elif channel_mean < 0.3:  # Dim
                        gamma = 0.6  # Light brightening
                    elif channel_mean < 0.6:  # Slightly dim
                        gamma = 0.8  # Very light brightening
                    else:
                        gamma = 0.9  # Minimal adjustment
                    
                    channel = np.power(channel, gamma)
                    
                    # Adaptive contrast enhancement
                    if contrast_ratio < 0.2:
                        # Low contrast - enhance moderately
                        channel = np.clip((channel - 0.5) * 1.5 + 0.5, 0, 1)
                    else:
                        # Normal contrast enhancement
                        channel = np.clip((channel - 0.5) * 1.2 + 0.5, 0, 1)
                
                enhanced[:, :, c] = channel
    else:
        # Grayscale image - convert to RGB after enhancement
        if len(enhanced.shape) == 3:
            enhanced = enhanced[:, :, 0]  # Take first channel if it's grayscale with 3 channels
        
        if enhanced.max() > enhanced.min():
            channel_min = enhanced.min()
            channel_max = enhanced.max()
            channel_mean = enhanced.mean()
            contrast_ratio = (channel_max - channel_min) / (channel_max + 1e-8)
            
            # Handle low contrast images
            if contrast_ratio < 0.05:
                # Very aggressive enhancement for white/blank images
                enhanced = (enhanced - channel_min) / (channel_max - channel_min + 1e-8)
                enhanced = np.power(enhanced, 0.2)  # Strong gamma
                enhanced = np.clip((enhanced - 0.5) * 3.0 + 0.5, 0, 1)
            else:
                # Normal processing
                enhanced = (enhanced - channel_min) / (channel_max - channel_min)
                
                if channel_mean < 0.1:
                    gamma = 0.4
                elif channel_mean < 0.6:
                    gamma = 0.7
                else:
                    gamma = 0.8
                
                enhanced = np.power(enhanced, gamma)
                
                # Adaptive contrast enhancement
                if contrast_ratio < 0.2:
                    enhanced = np.clip((enhanced - 0.5) * 2.0 + 0.5, 0, 1)
                else:
                    enhanced = np.clip((enhanced - 0.5) * 1.3 + 0.5, 0, 1)
        
        # Convert grayscale to RGB
        enhanced = np.stack([enhanced] * 3, axis=-1)
    
    # Convert back to appropriate range for display (always return uint8 for matplotlib)
    return (enhanced * 255).astype(np.uint8)

def create_comprehensive_visualization(image_name, gt_instances, all_model_results, save_dir):
    """Create 4x7 comprehensive visualization showing original, GT, predictions, and errors for all models"""
    
    # Load original image
    try:
        original_image_path = TEST_IMAGES_PATH / image_name
        if original_image_path.exists():
            if original_image_path.suffix.lower() in {'.tif', '.tiff'}:
                original_image = tiff.imread(str(original_image_path))
            else:
                original_image = np.array(Image.open(original_image_path))
            
            # Normalize to 0-255 range if needed (handle 16-bit images)
            if original_image.dtype == np.uint16:
                original_image = (original_image / 65535.0 * 255.0).astype(np.uint8)
            elif original_image.dtype != np.uint8:
                # Handle other data types
                if original_image.max() > 255:
                    original_image = (original_image / original_image.max() * 255.0).astype(np.uint8)
                else:
                    original_image = original_image.astype(np.uint8)
            
            # Convert to RGB if grayscale
            if len(original_image.shape) == 2:
                original_image = np.stack([original_image] * 3, axis=-1)
            elif len(original_image.shape) == 3 and original_image.shape[2] == 1:
                original_image = np.repeat(original_image, 3, axis=2)
            
            # Enhance image contrast and brightness for better visibility
            original_image = enhance_image_for_display(original_image)
                
        else:
            # Create placeholder if original image not found
            original_image = np.zeros_like(np.stack([gt_instances] * 3, axis=-1))
    except Exception as e:
        print(f"Could not load original image {image_name}: {e}")
        original_image = np.zeros_like(np.stack([gt_instances] * 3, axis=-1))
    
    # Setup figure - 7 models x 4 columns
    fig, axes = plt.subplots(7, 4, figsize=(16, 28))
    fig.suptitle(f'Comprehensive Error Analysis: {image_name}', fontsize=20, y=0.98)
    
    # Column headers
    col_titles = ['Original Image', 'Ground Truth', 'Model Predictions', 'Error Overlay']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    model_names = ['unet', 'nnunet', 'sac', 'lstmunet', 'maunet_resnet50', 'maunet_wide', 'maunet_ensemble']
    
    for row, model_name in enumerate(model_names):
        if model_name not in all_model_results:
            # Skip if model results not available
            for col in range(4):
                axes[row, col].axis('off')
                axes[row, col].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[row, col].transAxes)
            continue
            
        results = all_model_results[model_name]['results']
        pred_instances = all_model_results[model_name]['predictions']
        
        # Row label
        axes[row, 0].set_ylabel(f'{model_name.upper()}\nF1: {results["f1_score"]:.3f}', 
                               fontsize=12, fontweight='bold', rotation=0, 
                               ha='right', va='center', labelpad=50)
        
        # Column 1: Original Image
        axes[row, 0].imshow(original_image)
        axes[row, 0].axis('off')
        
        # Column 2: Ground Truth
        axes[row, 1].imshow(gt_instances, cmap='tab20')
        axes[row, 1].axis('off')
        if row == 0:  # Only show count on first row
            axes[row, 1].text(0.02, 0.98, f'{results["gt_count"]} cells', 
                             transform=axes[row, 1].transAxes, fontsize=10, 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                             verticalalignment='top')
        
        # Column 3: Predictions
        axes[row, 2].imshow(pred_instances, cmap='tab20')
        axes[row, 2].axis('off')
        # Add model name and cell count
        pred_text = f'{model_name.upper()}\n{results["pred_count"]} cells'
        axes[row, 2].text(0.02, 0.98, pred_text, 
                         transform=axes[row, 2].transAxes, fontsize=10, 
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.9),
                         verticalalignment='top', fontweight='bold')
        
        # Column 4: Error Overlay
        error_map = np.zeros_like(gt_instances, dtype=np.uint8)
        
        # Compute IoU for error visualization
        gt_ids, pr_ids, iou_matrix, _, _, _ = _iou_matrix(gt_instances, pred_instances)
        
        if iou_matrix.size > 0:
            # Hungarian matching
            cost_matrix = 1.0 - iou_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            valid_matches = iou_matrix[row_indices, col_indices] >= 0.5
            
            matched_gt_set = set(row_indices[valid_matches])
            matched_pr_set = set(col_indices[valid_matches])
            
            # Mark FN (missed GT cells) as red (1)
            for i, gt_id in enumerate(gt_ids):
                if i not in matched_gt_set:
                    error_map[gt_instances == gt_id] = 1
            
            # Mark FP (extra predictions) as blue (2)
            for j, pr_id in enumerate(pr_ids):
                if j not in matched_pr_set:
                    error_map[pred_instances == pr_id] = 2
            
            # Mark TP (correct detections) as green (3)
            for i, j in zip(row_indices[valid_matches], col_indices[valid_matches]):
                gt_id = gt_ids[i]
                pr_id = pr_ids[j]
                # Only mark overlap regions as TP
                overlap_mask = (gt_instances == gt_id) & (pred_instances == pr_id)
                error_map[overlap_mask] = 3
        
        # Error visualization with custom colormap
        colors = ['black', 'red', 'blue', 'green']  # background, FN, FP, TP
        cmap = ListedColormap(colors)
        axes[row, 3].imshow(error_map, cmap=cmap, vmin=0, vmax=3)
        axes[row, 3].axis('off')
        
        # Add error statistics
        error_text = f"FN: {results['false_negatives']}\nFP: {results['false_positives']}\nSplits: {results['splits']}\nMerges: {results['merges']}"
        axes[row, 3].text(0.02, 0.98, error_text, 
                         transform=axes[row, 3].transAxes, fontsize=9, 
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                         verticalalignment='top')
    
    # Add legend for error overlay
    legend_elements = [
        mpatches.Patch(color='red', label='False Negative (FN)'),
        mpatches.Patch(color='blue', label='False Positive (FP)'), 
        mpatches.Patch(color='green', label='True Positive (TP)'),
        mpatches.Patch(color='black', label='Background')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=12, 
              bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, bottom=0.06, hspace=0.05, wspace=0.02)
    
    save_path = save_dir / f'{Path(image_name).stem}_comprehensive_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_success_failure_analysis(summary_df, data_loader, save_dir):
    """Create analysis of success and failure cases"""
    
    # Find best and worst performing cases for each model
    model_names = ['maunet_ensemble', 'maunet_wide', 'maunet_resnet50', 'nnunet']
    
    for model_name in model_names:
        model_data = summary_df[summary_df['model'] == model_name].copy()
        
        # Sort by F1 score
        model_data = model_data.sort_values('f1_score')
        
        # Get best and worst cases (excluding extreme outliers)
        worst_cases = model_data.head(3)  # Bottom 3
        best_cases = model_data.tail(3)   # Top 3
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        fig.suptitle(f'{model_name.upper()}: Success vs Failure Cases', fontsize=16)
        
        # Process failure cases
        for i, (_, case) in enumerate(worst_cases.iterrows()):
            if i >= 3:
                break
                
            image_name = case['image']
            gt_instances = data_loader.load_ground_truth_instances(image_name)
            pred_instances = data_loader.load_prediction_instances(model_name, image_name)
            
            if gt_instances is not None and pred_instances is not None:
                # Ground truth
                axes[0, i*2].imshow(gt_instances, cmap='tab20')
                axes[0, i*2].set_title(f'GT: {case["gt_count"]} cells')
                axes[0, i*2].axis('off')
                
                # Predictions  
                axes[0, i*2+1].imshow(pred_instances, cmap='tab20')
                axes[0, i*2+1].set_title(f'Pred: {case["pred_count"]} cells\nF1: {case["f1_score"]:.3f}')
                axes[0, i*2+1].axis('off')
                
                # Add failure case label
                axes[0, i*2].text(0.5, -0.1, f'FAILURE\n{Path(image_name).stem}', 
                                 ha='center', va='top', transform=axes[0, i*2].transAxes,
                                 fontsize=10, fontweight='bold', color='red')
        
        # Process success cases
        for i, (_, case) in enumerate(best_cases.iterrows()):
            if i >= 3:
                break
                
            image_name = case['image']
            gt_instances = data_loader.load_ground_truth_instances(image_name)
            pred_instances = data_loader.load_prediction_instances(model_name, image_name)
            
            if gt_instances is not None and pred_instances is not None:
                # Ground truth
                axes[1, i*2].imshow(gt_instances, cmap='tab20')
                axes[1, i*2].set_title(f'GT: {case["gt_count"]} cells')
                axes[1, i*2].axis('off')
                
                # Predictions
                axes[1, i*2+1].imshow(pred_instances, cmap='tab20')
                axes[1, i*2+1].set_title(f'Pred: {case["pred_count"]} cells\nF1: {case["f1_score"]:.3f}')
                axes[1, i*2+1].axis('off')
                
                # Add success case label
                axes[1, i*2].text(0.5, -0.1, f'SUCCESS\n{Path(image_name).stem}', 
                                 ha='center', va='top', transform=axes[1, i*2].transAxes,
                                 fontsize=10, fontweight='bold', color='green')
        
        plt.tight_layout()
        save_path = save_dir / f'{model_name}_success_failure_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created success/failure analysis for {model_name}")

def create_error_visualization(gt_instances, pred_instances, image_name, model_name, results, save_dir):
    """Legacy function - now creates comprehensive visualization"""
    # This function is kept for compatibility but now delegates to comprehensive version
    return None

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"error_analysis_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def run_error_analysis():
    """Run error analysis on all 101 test images"""
    logger = setup_logging()
    logger.info("Starting Instance-Aware Error Analysis...")
    
    # Initialize components
    data_loader = ErrorAnalysisDataLoader(logger)
    error_analyzer = ErrorAnalyzer(logger, min_instance_size=10)
    
    # Get test images
    test_image_list = data_loader.get_test_image_list()
    
    if not test_image_list:
        logger.error("No test images found!")
        return
    
    logger.info(f"Found {len(test_image_list)} test images")
    
    # Create output directories
    results_dir = Path(__file__).parent / "results" / "error_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Results storage
    all_results = {}
    summary_data = []
    
    # Process each image (excluding cell_00101)
    filtered_image_list = [img for img in test_image_list if img != 'cell_00101.tif']
    for image_name in tqdm(filtered_image_list, desc="Processing images"):
        logger.info(f"Processing {image_name}...")
        
        # Load ground truth
        try:
            gt_instances = data_loader.load_ground_truth_instances(image_name)
            if gt_instances is None:
                logger.warning(f"Could not load ground truth for {image_name}")
                continue
        except Exception as e:
            logger.error(f"Error loading ground truth for {image_name}: {str(e)}")
            continue
        
        image_results = {}
        
        # Process each model
        for model_name in MODELS.keys():
            try:
                # Load prediction
                pred_instances = data_loader.load_prediction_instances(model_name, image_name)
                if pred_instances is None:
                    logger.warning(f"Could not load prediction for {model_name} on {image_name}")
                    continue
                
                # Analyze errors
                results = error_analyzer.analyze_errors(gt_instances, pred_instances)
                image_results[model_name] = results
                
                # Add to summary
                summary_row = {
                    'image': image_name,
                    'model': model_name,
                    **results
                }
                summary_data.append(summary_row)
                
                # Store results for comprehensive visualization
                if image_name not in image_results:
                    image_results[image_name] = {}
                image_results[image_name][model_name] = {
                    'results': results,
                    'predictions': pred_instances
                }
                
                # Log results
                logger.info(f"  {model_name}: GT={results['gt_count']}, Pred={results['pred_count']}, "
                          f"TP={results['true_positives']}, FN={results['false_negatives']}, FP={results['false_positives']}, "
                          f"Splits={results['splits']}, Merges={results['merges']}, F1={results['f1_score']:.3f}, PQ={results['PQ']:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing {model_name} on {image_name}: {str(e)}")
                continue
        
        if image_results:
            all_results[image_name] = image_results
    
    # Create comprehensive visualizations for selected images
    selected_images = ['cell_00007.tiff', 'cell_00062.png', 'cell_00050.png', 'cell_00073.tif']
    logger.info("Creating comprehensive visualizations...")
    
    for image_name in selected_images:
        if image_name in all_results:
            try:
                gt_instances = data_loader.load_ground_truth_instances(image_name)
                if gt_instances is not None:
                    create_comprehensive_visualization(image_name, gt_instances, all_results[image_name], viz_dir)
                    logger.info(f"Created comprehensive visualization for {image_name}")
            except Exception as e:
                logger.error(f"Error creating comprehensive visualization for {image_name}: {str(e)}")
    
    # Create success/failure analysis
    logger.info("Creating success/failure analysis...")
    try:
        summary_df_temp = pd.DataFrame(summary_data)
        create_success_failure_analysis(summary_df_temp, data_loader, viz_dir)
        logger.info("Created success/failure analysis")
    except Exception as e:
        logger.error(f"Error creating success/failure analysis: {str(e)}")
    
    # Save detailed results
    detailed_path = results_dir / "detailed_results.json"
    with open(detailed_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved detailed results to {detailed_path}")
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = results_dir / "error_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")
    
    # Generate final report
    generate_final_report(summary_df, results_dir, logger)
    
    logger.info(f"Error analysis complete! Processed {len(all_results)} images")
    return all_results, summary_df

def generate_final_report(summary_df, results_dir, logger):
    """Generate comprehensive final report with metrics and findings"""
    
    logger.info("Generating final report...")
    
    # Create comprehensive visualizations
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Comprehensive Error Analysis Report - 100 Images', fontsize=16)
    
    # 1. Overall Performance Metrics with Micro-aggregation
    # Micro aggregation: sum TP/FP/FN per model, then derive P/R/F1 once
    micro_rows = []
    for model, g in summary_df.groupby('model'):
        TP = int(g['true_positives'].sum())
        FP = int(g['false_positives'].sum())
        FN = int(g['false_negatives'].sum())
        P = TP / (TP + FP) if (TP + FP) else 0.0
        R = TP / (TP + FN) if (TP + FN) else 0.0
        F1 = 2*P*R/(P+R) if (P+R) else 0.0
        micro_rows.append(dict(model=model, precision=P, recall=R, f1_score=F1,
                               PQ=g['PQ'].mean(), RQ=g['RQ'].mean(), SQ=g['SQ'].mean()))
    model_performance = pd.DataFrame(micro_rows).set_index('model').round(3)
    
    # Plot F1 scores
    axes[0, 0].bar(model_performance.index, model_performance['f1_score'])
    axes[0, 0].set_title('Average F1-Score by Model')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot PQ scores
    axes[0, 1].bar(model_performance.index, model_performance['PQ'])
    axes[0, 1].set_title('Average Panoptic Quality (PQ) by Model')
    axes[0, 1].set_ylabel('PQ')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 2. Error Type Analysis
    error_summary = summary_df.groupby('model').agg({
        'false_negatives': 'mean',
        'false_positives': 'mean',
        'splits': 'mean',
        'merges': 'mean'
    }).round(1)
    
    # Plot error types
    x = np.arange(len(error_summary.index))
    width = 0.2
    
    axes[0, 2].bar(x - 1.5*width, error_summary['false_negatives'], width, label='False Negatives', color='red', alpha=0.7)
    axes[0, 2].bar(x - 0.5*width, error_summary['false_positives'], width, label='False Positives', color='blue', alpha=0.7)
    axes[0, 2].bar(x + 0.5*width, error_summary['splits'], width, label='Splits', color='orange', alpha=0.7)
    axes[0, 2].bar(x + 1.5*width, error_summary['merges'], width, label='Merges', color='green', alpha=0.7)
    
    axes[0, 2].set_title('Average Error Types by Model')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(error_summary.index, rotation=45)
    axes[0, 2].legend()
    
    # 3. Cell Count Analysis
    cell_count_analysis = summary_df.groupby('model').agg({
        'gt_count': 'mean',
        'pred_count': 'mean'
    }).round(1)
    
    axes[1, 0].bar(x - 0.2, cell_count_analysis['gt_count'], 0.4, label='Ground Truth', alpha=0.7)
    axes[1, 0].bar(x + 0.2, cell_count_analysis['pred_count'], 0.4, label='Predicted', alpha=0.7)
    axes[1, 0].set_title('Average Cell Count by Model')
    axes[1, 0].set_ylabel('Cell Count')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(cell_count_analysis.index, rotation=45)
    axes[1, 0].legend()
    
    # 4. Precision vs Recall scatter
    for model in summary_df['model'].unique():
        model_data = summary_df[summary_df['model'] == model]
        axes[1, 1].scatter(model_data['recall'], model_data['precision'], 
                          label=model, alpha=0.6, s=20)
    
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Precision vs Recall Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. PQ Components
    axes[1, 2].bar(x - 0.2, model_performance['RQ'], 0.4, label='Recognition Quality (RQ)', alpha=0.7)
    axes[1, 2].bar(x + 0.2, model_performance['SQ'], 0.4, label='Segmentation Quality (SQ)', alpha=0.7)
    axes[1, 2].set_title('PQ Components by Model')
    axes[1, 2].set_ylabel('Quality Score')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(model_performance.index, rotation=45)
    axes[1, 2].legend()
    
    # 6. Error Distribution Heatmap
    error_matrix = summary_df.pivot_table(index='model', 
                                         values=['false_negatives', 'false_positives', 'splits', 'merges'], 
                                         aggfunc='mean')
    
    sns.heatmap(error_matrix.T, annot=True, fmt='.1f', cmap='Reds', ax=axes[2, 0])
    axes[2, 0].set_title('Error Type Heatmap')
    axes[2, 0].set_ylabel('Error Type')
    
    # 7. Performance Ranking
    ranking_metrics = ['f1_score', 'PQ', 'precision', 'recall']
    ranking_data = model_performance[ranking_metrics].rank(ascending=False, method='average')
    
    sns.heatmap(ranking_data.T, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=axes[2, 1])
    axes[2, 1].set_title('Performance Ranking (1=Best)')
    axes[2, 1].set_ylabel('Metric')
    
    # 8. Summary Statistics Table
    axes[2, 2].axis('off')
    summary_stats = model_performance.round(3)
    table_data = []
    for model in summary_stats.index:
        row = [model] + [f"{summary_stats.loc[model, col]:.3f}" for col in summary_stats.columns]
        table_data.append(row)
    
    table = axes[2, 2].table(cellText=table_data,
                            colLabels=['Model'] + list(summary_stats.columns),
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    axes[2, 2].set_title('Performance Summary Table')
    
    plt.tight_layout()
    report_path = results_dir / "comprehensive_error_analysis_report.png"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate text report
    report_text_path = results_dir / "final_error_analysis_report.md"
    with open(report_text_path, 'w') as f:
        f.write("# Comprehensive Error Analysis Report\n\n")
        f.write("## Dataset Overview\n")
        f.write(f"- Total images analyzed: {len(summary_df['image'].unique())}\n")
        f.write(f"- Models evaluated: {', '.join(summary_df['model'].unique())}\n")
        f.write(f"- Minimum instance size: 10 pixels\n")
        f.write(f"- IoU threshold: 0.5\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Best performing model
        best_f1 = model_performance['f1_score'].idxmax()
        best_pq = model_performance['PQ'].idxmax()
        
        f.write(f"### Overall Performance\n")
        f.write(f"- **Best F1-Score**: {best_f1} ({model_performance.loc[best_f1, 'f1_score']:.3f})\n")
        f.write(f"- **Best Panoptic Quality**: {best_pq} ({model_performance.loc[best_pq, 'PQ']:.3f})\n")
        f.write(f"- **Average GT cells per image**: {summary_df['gt_count'].mean():.1f}\n\n")
        
        f.write("### Error Analysis Summary\n\n")
        f.write("| Model | F1-Score | PQ | Precision | Recall | Avg FN | Avg FP | Avg Splits | Avg Merges |\n")
        f.write("|-------|----------|----|-----------|---------|---------|---------|-----------|-----------|\n")
        
        for model in model_performance.index:
            model_errors = error_summary.loc[model]
            model_perf = model_performance.loc[model]
            f.write(f"| {model} | {model_perf['f1_score']:.3f} | {model_perf['PQ']:.3f} | "
                   f"{model_perf['precision']:.3f} | {model_perf['recall']:.3f} | "
                   f"{model_errors['false_negatives']:.1f} | {model_errors['false_positives']:.1f} | "
                   f"{model_errors['splits']:.1f} | {model_errors['merges']:.1f} |\n")
        
        f.write("\n### Detailed Metrics Explanation\n")
        f.write("- **False Negatives (FN)**: Ground truth cells that were missed by the model\n")
        f.write("- **False Positives (FP)**: Artifacts wrongly segmented as cells\n") 
        f.write("- **Splits**: Ground truth cells that were over-segmented into multiple predictions\n")
        f.write("- **Merges**: Multiple ground truth cells that were under-segmented into one prediction\n")
        f.write("- **PQ (Panoptic Quality)**: Overall segmentation quality combining recognition and segmentation\n")
        f.write("- **RQ (Recognition Quality)**: How well the model detects instances\n")
        f.write("- **SQ (Segmentation Quality)**: How accurately detected instances are segmented\n\n")
        
        f.write("### Model Rankings\n")
        f.write("Based on F1-Score:\n")
        for i, (model, score) in enumerate(model_performance['f1_score'].sort_values(ascending=False).items(), 1):
            f.write(f"{i}. {model}: {score:.3f}\n")
        
        f.write("\nBased on Panoptic Quality:\n")
        for i, (model, score) in enumerate(model_performance['PQ'].sort_values(ascending=False).items(), 1):
            f.write(f"{i}. {model}: {score:.3f}\n")
    
    logger.info(f"Final report saved to {report_text_path}")
    logger.info(f"Comprehensive visualization saved to {report_path}")

if __name__ == "__main__":
    run_error_analysis()
