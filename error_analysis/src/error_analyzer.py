"""
Error categorization and analysis system
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from skimage import morphology, segmentation, measure
import logging

from config.analysis_config import ANALYSIS_CONFIG, IMAGE_CONFIG

class ErrorAnalyzer:
    """Analyzes and categorizes different types of segmentation errors"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = ANALYSIS_CONFIG['error_analysis']
        
    def analyze_errors(self, ground_truth: np.ndarray, prediction: np.ndarray) -> Dict:
        """
        Comprehensive error analysis between ground truth and prediction
        
        Args:
            ground_truth: Ground truth segmentation (0=background, 1=cell, 2=boundary)
            prediction: Model prediction (same format)
            
        Returns:
            Dictionary with detailed error analysis
        """
        # Convert to binary masks for cell detection
        gt_cells = (ground_truth > 0).astype(int)
        pred_cells = (prediction > 0).astype(int)
        
        # Label connected components (individual cells)
        gt_labeled, gt_num_cells = ndimage.label(gt_cells)
        pred_labeled, pred_num_cells = ndimage.label(pred_cells)
        
        # Analyze different error types
        error_analysis = {
            'false_negatives': self._analyze_false_negatives(gt_labeled, pred_labeled, gt_num_cells),
            'false_positives': self._analyze_false_positives(gt_labeled, pred_labeled, pred_num_cells),
            'under_segmentation': self._analyze_under_segmentation(gt_labeled, pred_labeled),
            'over_segmentation': self._analyze_over_segmentation(gt_labeled, pred_labeled),
            'boundary_errors': self._analyze_boundary_errors(ground_truth, prediction),
            'summary': self._compute_error_summary(gt_labeled, pred_labeled, gt_num_cells, pred_num_cells)
        }
        
        return error_analysis
    
    def _analyze_false_negatives(self, gt_labeled: np.ndarray, pred_labeled: np.ndarray, 
                                gt_num_cells: int) -> Dict:
        """Analyze completely missed cells (False Negatives)"""
        missed_cells = []
        
        for gt_cell_id in range(1, gt_num_cells + 1):
            gt_mask = (gt_labeled == gt_cell_id)
            
            # Check if this ground truth cell has any overlap with predictions
            overlap_with_pred = pred_labeled[gt_mask]
            unique_pred_labels = np.unique(overlap_with_pred[overlap_with_pred > 0])
            
            if len(unique_pred_labels) == 0:
                # Completely missed cell
                cell_area = np.sum(gt_mask)
                cell_centroid = ndimage.center_of_mass(gt_mask)
                
                missed_cells.append({
                    'cell_id': gt_cell_id,
                    'area': cell_area,
                    'centroid': cell_centroid,
                    'reason': 'completely_missed'
                })
            else:
                # Check if overlap is significant enough
                max_overlap = 0
                for pred_label in unique_pred_labels:
                    pred_mask = (pred_labeled == pred_label)
                    intersection = np.sum(gt_mask & pred_mask)
                    union = np.sum(gt_mask | pred_mask)
                    iou = intersection / union if union > 0 else 0
                    max_overlap = max(max_overlap, iou)
                
                if max_overlap < self.config['overlap_threshold']:
                    # Poorly detected cell
                    cell_area = np.sum(gt_mask)
                    cell_centroid = ndimage.center_of_mass(gt_mask)
                    
                    missed_cells.append({
                        'cell_id': gt_cell_id,
                        'area': cell_area,
                        'centroid': cell_centroid,
                        'max_iou': max_overlap,
                        'reason': 'poor_detection'
                    })
        
        return {
            'count': len(missed_cells),
            'details': missed_cells,
            'total_missed_area': sum(cell['area'] for cell in missed_cells)
        }
    
    def _analyze_false_positives(self, gt_labeled: np.ndarray, pred_labeled: np.ndarray,
                               pred_num_cells: int) -> Dict:
        """Analyze false positive detections"""
        false_positives = []
        
        for pred_cell_id in range(1, pred_num_cells + 1):
            pred_mask = (pred_labeled == pred_cell_id)
            
            # Check if this predicted cell overlaps with ground truth
            overlap_with_gt = gt_labeled[pred_mask]
            unique_gt_labels = np.unique(overlap_with_gt[overlap_with_gt > 0])
            
            if len(unique_gt_labels) == 0:
                # Complete false positive
                cell_area = np.sum(pred_mask)
                cell_centroid = ndimage.center_of_mass(pred_mask)
                
                false_positives.append({
                    'cell_id': pred_cell_id,
                    'area': cell_area,
                    'centroid': cell_centroid,
                    'reason': 'noise_detection'
                })
            else:
                # Check if it's a significant false positive
                max_overlap = 0
                for gt_label in unique_gt_labels:
                    gt_mask = (gt_labeled == gt_label)
                    intersection = np.sum(gt_mask & pred_mask)
                    union = np.sum(gt_mask | pred_mask)
                    iou = intersection / union if union > 0 else 0
                    max_overlap = max(max_overlap, iou)
                
                if max_overlap < self.config['overlap_threshold']:
                    cell_area = np.sum(pred_mask)
                    cell_centroid = ndimage.center_of_mass(pred_mask)
                    
                    false_positives.append({
                        'cell_id': pred_cell_id,
                        'area': cell_area,
                        'centroid': cell_centroid,
                        'max_iou': max_overlap,
                        'reason': 'poor_localization'
                    })
        
        return {
            'count': len(false_positives),
            'details': false_positives,
            'total_fp_area': sum(cell['area'] for cell in false_positives)
        }
    
    def _analyze_under_segmentation(self, gt_labeled: np.ndarray, pred_labeled: np.ndarray) -> Dict:
        """Analyze cases where multiple cells are merged into one"""
        under_segmentation_cases = []
        
        # For each predicted cell, check how many ground truth cells it overlaps with
        pred_num_cells = np.max(pred_labeled)
        
        for pred_cell_id in range(1, pred_num_cells + 1):
            pred_mask = (pred_labeled == pred_cell_id)
            
            # Find all ground truth cells that overlap with this prediction
            overlap_with_gt = gt_labeled[pred_mask]
            unique_gt_labels = np.unique(overlap_with_gt[overlap_with_gt > 0])
            
            if len(unique_gt_labels) > 1:
                # This predicted cell overlaps with multiple ground truth cells
                overlapping_cells = []
                total_overlap_area = 0
                
                for gt_label in unique_gt_labels:
                    gt_mask = (gt_labeled == gt_label)
                    intersection = np.sum(gt_mask & pred_mask)
                    
                    # Only count significant overlaps
                    if intersection > self.config['min_cell_area']:
                        overlapping_cells.append({
                            'gt_cell_id': gt_label,
                            'overlap_area': intersection,
                            'gt_cell_area': np.sum(gt_mask)
                        })
                        total_overlap_area += intersection
                
                if len(overlapping_cells) > 1:
                    under_segmentation_cases.append({
                        'pred_cell_id': pred_cell_id,
                        'pred_cell_area': np.sum(pred_mask),
                        'merged_gt_cells': overlapping_cells,
                        'num_merged_cells': len(overlapping_cells),
                        'total_overlap_area': total_overlap_area
                    })
        
        return {
            'count': len(under_segmentation_cases),
            'details': under_segmentation_cases,
            'total_merged_cells': sum(case['num_merged_cells'] for case in under_segmentation_cases)
        }
    
    def _analyze_over_segmentation(self, gt_labeled: np.ndarray, pred_labeled: np.ndarray) -> Dict:
        """Analyze cases where single cells are split into multiple predictions"""
        over_segmentation_cases = []
        
        # For each ground truth cell, check how many predictions overlap with it
        gt_num_cells = np.max(gt_labeled)
        
        for gt_cell_id in range(1, gt_num_cells + 1):
            gt_mask = (gt_labeled == gt_cell_id)
            
            # Find all predicted cells that overlap with this ground truth
            overlap_with_pred = pred_labeled[gt_mask]
            unique_pred_labels = np.unique(overlap_with_pred[overlap_with_pred > 0])
            
            if len(unique_pred_labels) > 1:
                # This ground truth cell is split into multiple predictions
                splitting_predictions = []
                total_overlap_area = 0
                
                for pred_label in unique_pred_labels:
                    pred_mask = (pred_labeled == pred_label)
                    intersection = np.sum(gt_mask & pred_mask)
                    
                    # Only count significant overlaps
                    if intersection > self.config['min_cell_area']:
                        splitting_predictions.append({
                            'pred_cell_id': pred_label,
                            'overlap_area': intersection,
                            'pred_cell_area': np.sum(pred_mask)
                        })
                        total_overlap_area += intersection
                
                if len(splitting_predictions) > 1:
                    over_segmentation_cases.append({
                        'gt_cell_id': gt_cell_id,
                        'gt_cell_area': np.sum(gt_mask),
                        'splitting_predictions': splitting_predictions,
                        'num_splits': len(splitting_predictions),
                        'total_overlap_area': total_overlap_area
                    })
        
        return {
            'count': len(over_segmentation_cases),
            'details': over_segmentation_cases,
            'total_split_cells': len(over_segmentation_cases)
        }
    
    def _analyze_boundary_errors(self, ground_truth: np.ndarray, prediction: np.ndarray) -> Dict:
        """Analyze boundary classification errors"""
        # Focus on boundary class (class 2)
        gt_boundaries = (ground_truth == 2)
        pred_boundaries = (prediction == 2)
        
        # Dilate boundaries for analysis
        thickness = self.config['boundary_thickness']
        kernel = morphology.disk(thickness)
        
        gt_boundary_region = morphology.binary_dilation(gt_boundaries, kernel)
        pred_boundary_region = morphology.binary_dilation(pred_boundaries, kernel)
        
        # Calculate boundary metrics
        boundary_tp = np.sum(gt_boundary_region & pred_boundary_region)
        boundary_fp = np.sum(pred_boundary_region & ~gt_boundary_region)
        boundary_fn = np.sum(gt_boundary_region & ~pred_boundary_region)
        
        boundary_precision = boundary_tp / (boundary_tp + boundary_fp) if (boundary_tp + boundary_fp) > 0 else 0
        boundary_recall = boundary_tp / (boundary_tp + boundary_fn) if (boundary_tp + boundary_fn) > 0 else 0
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0
        
        return {
            'boundary_precision': boundary_precision,
            'boundary_recall': boundary_recall,
            'boundary_f1': boundary_f1,
            'boundary_tp': boundary_tp,
            'boundary_fp': boundary_fp,
            'boundary_fn': boundary_fn,
            'gt_boundary_pixels': np.sum(gt_boundaries),
            'pred_boundary_pixels': np.sum(pred_boundaries)
        }
    
    def _compute_error_summary(self, gt_labeled: np.ndarray, pred_labeled: np.ndarray,
                              gt_num_cells: int, pred_num_cells: int) -> Dict:
        """Compute overall error summary statistics"""
        return {
            'gt_num_cells': gt_num_cells,
            'pred_num_cells': pred_num_cells,
            'cell_count_error': pred_num_cells - gt_num_cells,
            'cell_count_error_rate': abs(pred_num_cells - gt_num_cells) / gt_num_cells if gt_num_cells > 0 else float('inf'),
            'total_gt_area': np.sum(gt_labeled > 0),
            'total_pred_area': np.sum(pred_labeled > 0),
            'area_difference': np.sum(pred_labeled > 0) - np.sum(gt_labeled > 0)
        }
    
    def compute_error_rates(self, error_analysis: Dict) -> Dict:
        """Compute normalized error rates from error analysis"""
        summary = error_analysis['summary']
        gt_num_cells = summary['gt_num_cells']
        
        if gt_num_cells == 0:
            return {
                'false_negative_rate': 0,
                'false_positive_rate': float('inf') if error_analysis['false_positives']['count'] > 0 else 0,
                'under_segmentation_rate': 0,
                'over_segmentation_rate': 0
            }
        
        return {
            'false_negative_rate': error_analysis['false_negatives']['count'] / gt_num_cells,
            'false_positive_rate': error_analysis['false_positives']['count'] / gt_num_cells,
            'under_segmentation_rate': error_analysis['under_segmentation']['total_merged_cells'] / gt_num_cells,
            'over_segmentation_rate': error_analysis['over_segmentation']['total_split_cells'] / gt_num_cells
        }
