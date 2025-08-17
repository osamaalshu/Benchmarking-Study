"""
Comprehensive Evaluation Framework for Modality Agnostic Controlled Augmentation Study

Evaluation protocol:
- Test set: always real and untouched
- Metrics: Dice / IoU / Precision / Recall / Boundary F1 (or HD95)
- Statistics: per-image paired test (R vs each RxS@r), report p and effect size
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import logging
from datetime import datetime
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

# Import existing evaluation utilities
# Note: compute_metrics not used in current implementation


class EvaluationMetrics:
    """Comprehensive evaluation metrics for segmentation"""
    
    @staticmethod
    def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """Calculate Dice coefficient"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        intersection = np.sum(pred_flat * target_flat)
        union = np.sum(pred_flat) + np.sum(target_flat)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice
    
    @staticmethod
    def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """Calculate Intersection over Union (IoU)"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        intersection = np.sum(pred_flat * target_flat)
        union = np.sum(pred_flat) + np.sum(target_flat) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou
    
    @staticmethod
    def precision_recall_f1(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Handle edge cases
        if np.sum(target_flat) == 0 and np.sum(pred_flat) == 0:
            return 1.0, 1.0, 1.0
        elif np.sum(target_flat) == 0:
            return 0.0, 1.0, 0.0
        elif np.sum(pred_flat) == 0:
            return 1.0, 0.0, 0.0
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_flat, pred_flat, average='binary', zero_division=0
        )
        
        return precision, recall, f1
    
    @staticmethod
    def boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
        """Calculate boundary F1 score with tolerance"""
        # Extract boundaries
        pred_boundary = EvaluationMetrics._extract_boundary(pred)
        target_boundary = EvaluationMetrics._extract_boundary(target)
        
        if np.sum(target_boundary) == 0 and np.sum(pred_boundary) == 0:
            return 1.0
        elif np.sum(target_boundary) == 0 or np.sum(pred_boundary) == 0:
            return 0.0
        
        # Dilate boundaries for tolerance
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance*2+1, tolerance*2+1))
        target_boundary_dilated = cv2.dilate(target_boundary.astype(np.uint8), kernel)
        pred_boundary_dilated = cv2.dilate(pred_boundary.astype(np.uint8), kernel)
        
        # Calculate precision and recall
        tp_precision = np.sum(pred_boundary * target_boundary_dilated)
        fp_precision = np.sum(pred_boundary) - tp_precision
        
        tp_recall = np.sum(target_boundary * pred_boundary_dilated)
        fn_recall = np.sum(target_boundary) - tp_recall
        
        precision = tp_precision / (tp_precision + fp_precision) if (tp_precision + fp_precision) > 0 else 0
        recall = tp_recall / (tp_recall + fn_recall) if (tp_recall + fn_recall) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    @staticmethod
    def _extract_boundary(mask: np.ndarray) -> np.ndarray:
        """Extract boundary from binary mask"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(mask.astype(np.uint8), kernel)
        boundary = mask.astype(np.uint8) - eroded
        return boundary
    
    @staticmethod
    def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate 95th percentile Hausdorff distance (correct implementation)"""
        try:
            from scipy.spatial.distance import cdist
            
            # Get boundary points
            pred_boundary = EvaluationMetrics._extract_boundary(pred)
            target_boundary = EvaluationMetrics._extract_boundary(target)
            
            pred_points = np.column_stack(np.where(pred_boundary))
            target_points = np.column_stack(np.where(target_boundary))
            
            if len(pred_points) == 0 or len(target_points) == 0:
                return float('inf')
            
            # Compute all pairwise distances
            distances_pred_to_target = cdist(pred_points, target_points)
            distances_target_to_pred = cdist(target_points, pred_points)
            
            # Get minimum distance for each point to the other set
            min_dist_pred_to_target = np.min(distances_pred_to_target, axis=1)
            min_dist_target_to_pred = np.min(distances_target_to_pred, axis=1)
            
            # Combine all minimum distances
            all_distances = np.concatenate([min_dist_pred_to_target, min_dist_target_to_pred])
            
            # Calculate 95th percentile of all distances
            hd95 = np.percentile(all_distances, 95)
            
            return hd95
        except Exception:
            return float('inf')


class ModelEvaluator:
    """Evaluates trained models on test data"""
    
    def __init__(self, 
                 test_data_dir: str,
                 device: str = 'auto'):
        """
        Initialize model evaluator
        
        Args:
            test_data_dir: Directory containing test data (images and labels)
            device: Device to use for evaluation
        """
        self.test_data_dir = Path(test_data_dir)
        self.device = self._get_device(device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load test data
        self.test_data = self._load_test_data()
        
    def _get_device(self, device: str) -> torch.device:
        """Automatically detect best available device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_test_data(self) -> List[Tuple[str, str]]:
        """Load test image-label pairs"""
        images_dir = self.test_data_dir / 'images'
        labels_dir = self.test_data_dir / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            raise ValueError(f"Expected images and labels directories in {self.test_data_dir}")
        
        image_files = sorted(list(images_dir.glob('*')))
        label_files = sorted(list(labels_dir.glob('*')))
        
        # Match image and label files
        matched_pairs = []
        for img_file in image_files:
            # Find corresponding label file
            label_file = None
            for lbl_file in label_files:
                # Handle different naming patterns: 
                # - cell_00001.tiff -> cell_00001_label.tiff
                # - cell_00001.png -> cell_00001_label.png
                base_name = img_file.stem
                if lbl_file.stem == base_name or lbl_file.stem == f"{base_name}_label":
                    label_file = lbl_file
                    break
            
            if label_file:
                matched_pairs.append((str(img_file), str(label_file)))
            else:
                self.logger.warning(f"No matching label found for {img_file}")
        
        self.logger.info(f"Loaded {len(matched_pairs)} test image-label pairs")
        return matched_pairs
    
    def evaluate_model(self, 
                      model_path: str,
                      model_type: str = 'nnunet',
                      input_size: int = 256,
                      num_classes: int = 3) -> Dict[str, List[float]]:
        """
        Evaluate a single model on test data
        
        Args:
            model_path: Path to trained model
            model_type: Type of model ('nnunet' or 'unet')
            input_size: Input image size
            num_classes: Number of classes
            
        Returns:
            Dictionary of per-image metrics
        """
        # Load model
        from .model_wrappers import get_model_creator
        creator = get_model_creator(model_type)
        model = creator(
            input_channels=3,
            num_classes=num_classes,
            input_size=input_size
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        # Evaluate on each test image
        results = {
            'dice': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'boundary_f1': [],
            'hd95': [],
            'image_names': []
        }
        
        with torch.no_grad():
            for img_path, lbl_path in self.test_data:
                try:
                    # Load and preprocess image
                    image = Image.open(img_path).convert('RGB')
                    image = image.resize((input_size, input_size), Image.LANCZOS)
                    image_array = np.array(image) / 255.0
                    image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float().unsqueeze(0)
                    image_tensor = image_tensor.to(self.device)
                    
                    # Load ground truth with proper class handling
                    label = Image.open(lbl_path).convert('L')
                    label = label.resize((input_size, input_size), Image.NEAREST)
                    label_array = np.array(label)
                    
                    # Handle different label formats (same as in dataset)
                    if num_classes == 2:
                        gt_binary = (label_array > 127).astype(np.uint8)
                    elif num_classes == 3:
                        if label_array.max() > 2:
                            if np.unique(label_array).tolist() == [0, 128, 255]:
                                label_array = (label_array / 127.5).astype(np.uint8)
                                label_array = np.clip(label_array, 0, 2)
                            else:
                                label_array = (label_array > 127).astype(np.uint8) * 2
                        gt_binary = (label_array > 0).astype(np.uint8)  # For binary metrics
                    else:
                        gt_binary = (label_array > 0).astype(np.uint8)
                    
                    # Model prediction
                    output = model(image_tensor)
                    pred_probs = F.softmax(output, dim=1)
                    pred_class = torch.argmax(pred_probs, dim=1).cpu().numpy()[0]
                    
                    # Convert prediction to binary for binary metrics
                    pred_binary = (pred_class > 0).astype(np.uint8)
                    
                    # Calculate metrics
                    dice = EvaluationMetrics.dice_coefficient(pred_binary, gt_binary)
                    iou = EvaluationMetrics.iou_score(pred_binary, gt_binary)
                    precision, recall, f1 = EvaluationMetrics.precision_recall_f1(pred_binary, gt_binary)
                    boundary_f1 = EvaluationMetrics.boundary_f1(pred_binary, gt_binary)
                    hd95 = EvaluationMetrics.hausdorff_distance_95(pred_binary, gt_binary)
                    
                    # Store results
                    results['dice'].append(dice)
                    results['iou'].append(iou)
                    results['precision'].append(precision)
                    results['recall'].append(recall)
                    results['f1'].append(f1)
                    results['boundary_f1'].append(boundary_f1)
                    results['hd95'].append(hd95 if hd95 != float('inf') else 100.0)  # Cap HD95
                    results['image_names'].append(Path(img_path).name)
                    
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {img_path}: {e}")
                    continue
        
        self.logger.info(f"Evaluated model on {len(results['dice'])} test images")
        return results


class StatisticalAnalyzer:
    """Performs statistical analysis comparing different arms"""
    
    def __init__(self):
        """Initialize statistical analyzer"""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def paired_comparison(self, 
                         baseline_results: Dict[str, List[float]],
                         treatment_results: Dict[str, List[float]],
                         baseline_name: str = 'R',
                         treatment_name: str = 'Treatment') -> Dict[str, Any]:
        """
        Perform paired statistical comparison between baseline and treatment
        
        Args:
            baseline_results: Results from baseline arm
            treatment_results: Results from treatment arm
            baseline_name: Name of baseline arm
            treatment_name: Name of treatment arm
            
        Returns:
            Statistical comparison results
        """
        comparison_results = {
            'baseline_name': baseline_name,
            'treatment_name': treatment_name,
            'metrics': {}
        }
        
        # Metrics to compare
        metrics = ['dice', 'iou', 'precision', 'recall', 'f1', 'boundary_f1', 'hd95']
        
        for metric in metrics:
            if metric in baseline_results and metric in treatment_results:
                # Align results by image names for proper pairing
                if 'image_names' in baseline_results and 'image_names' in treatment_results:
                    baseline_names = baseline_results['image_names']
                    treatment_names = treatment_results['image_names']
                    
                    # Find common images
                    common_names = list(set(baseline_names) & set(treatment_names))
                    if not common_names:
                        self.logger.warning(f"No common images found for {baseline_name} vs {treatment_name}")
                        continue
                    
                    # Align values by name
                    baseline_values = []
                    treatment_values = []
                    
                    for name in common_names:
                        try:
                            baseline_idx = baseline_names.index(name)
                            treatment_idx = treatment_names.index(name)
                            baseline_values.append(baseline_results[metric][baseline_idx])
                            treatment_values.append(treatment_results[metric][treatment_idx])
                        except (ValueError, IndexError):
                            continue
                    
                    baseline_values = np.array(baseline_values)
                    treatment_values = np.array(treatment_values)
                    
                    self.logger.info(f"Aligned {len(baseline_values)} pairs for {metric}")
                else:
                    # Fallback: truncate to same length (less reliable)
                    baseline_values = np.array(baseline_results[metric])
                    treatment_values = np.array(treatment_results[metric])
                    
                    min_len = min(len(baseline_values), len(treatment_values))
                    baseline_values = baseline_values[:min_len]
                    treatment_values = treatment_values[:min_len]
                    
                    self.logger.warning(f"Using truncation alignment for {metric} (not recommended)")
                
                if len(baseline_values) == 0:
                    continue
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(treatment_values, baseline_values)
                
                # Effect size (Cohen's d for paired samples)
                differences = treatment_values - baseline_values
                effect_size = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0
                
                # Descriptive statistics
                baseline_mean = np.mean(baseline_values)
                baseline_std = np.std(baseline_values)
                treatment_mean = np.mean(treatment_values)
                treatment_std = np.std(treatment_values)
                
                # Confidence interval for difference
                diff_mean = np.mean(differences)
                diff_se = stats.sem(differences)
                ci_95 = stats.t.interval(0.95, len(differences)-1, diff_mean, diff_se)
                
                comparison_results['metrics'][metric] = {
                    'baseline_mean': baseline_mean,
                    'baseline_std': baseline_std,
                    'treatment_mean': treatment_mean,
                    'treatment_std': treatment_std,
                    'difference_mean': diff_mean,
                    'difference_std': np.std(differences),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'effect_size_cohens_d': effect_size,
                    'confidence_interval_95': ci_95,
                    'significant': p_value < 0.05,
                    'n_samples': len(differences)
                }
        
        return comparison_results
    
    def multiple_comparisons(self, 
                           all_results: Dict[str, Dict[str, List[float]]],
                           baseline_arm: str = 'R') -> Dict[str, Dict]:
        """
        Perform multiple pairwise comparisons against baseline
        
        Args:
            all_results: Dictionary mapping arm names to their results
            baseline_arm: Name of baseline arm
            
        Returns:
            Dictionary of all pairwise comparisons
        """
        if baseline_arm not in all_results:
            raise ValueError(f"Baseline arm '{baseline_arm}' not found in results")
        
        baseline_results = all_results[baseline_arm]
        comparisons = {}
        
        for arm_name, arm_results in all_results.items():
            if arm_name != baseline_arm:
                comparison = self.paired_comparison(
                    baseline_results=baseline_results,
                    treatment_results=arm_results,
                    baseline_name=baseline_arm,
                    treatment_name=arm_name
                )
                comparisons[arm_name] = comparison
        
        return comparisons
    
    def bonferroni_correction(self, 
                            comparisons: Dict[str, Dict],
                            alpha: float = 0.05) -> Dict[str, Dict]:
        """
        Apply Bonferroni correction for multiple comparisons
        
        Args:
            comparisons: Results from multiple_comparisons
            alpha: Significance level
            
        Returns:
            Corrected comparison results
        """
        corrected_comparisons = {}
        
        # Count total number of tests
        total_tests = 0
        for comp in comparisons.values():
            total_tests += len(comp['metrics'])
        
        corrected_alpha = alpha / total_tests
        
        for arm_name, comp in comparisons.items():
            corrected_comp = comp.copy()
            corrected_comp['corrected_alpha'] = corrected_alpha
            corrected_comp['total_tests'] = total_tests
            
            for metric in corrected_comp['metrics']:
                original_p = corrected_comp['metrics'][metric]['p_value']
                corrected_comp['metrics'][metric]['significant_bonferroni'] = original_p < corrected_alpha
            
            corrected_comparisons[arm_name] = corrected_comp
        
        return corrected_comparisons


class ComprehensiveEvaluator:
    """Main evaluator that coordinates model evaluation and statistical analysis"""
    
    def __init__(self, 
                 test_data_dir: str,
                 results_output_dir: str):
        """
        Initialize comprehensive evaluator
        
        Args:
            test_data_dir: Directory containing test data
            results_output_dir: Directory to save evaluation results
        """
        self.test_data_dir = test_data_dir
        self.results_output_dir = Path(results_output_dir)
        self.results_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_evaluator = ModelEvaluator(test_data_dir)
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_all_models_all_arms(self, 
                                   training_results_dir: str,
                                   models: List[str] = ['nnunet', 'unet'],
                                   seeds: List[int] = [0, 1, 2]) -> Dict[str, Any]:
        """
        Evaluate all trained models on all arms
        
        Args:
            training_results_dir: Directory containing training results
            models: List of models to evaluate
            seeds: List of seeds used in training
            
        Returns:
            Comprehensive evaluation results
        """
        training_results_dir = Path(training_results_dir)
        
        all_evaluation_results = {}
        
        for model_name in models:
            self.logger.info(f"Evaluating model: {model_name}")
            
            model_results_dir = training_results_dir / model_name
            model_eval_results = {}
            
            # Find all arm directories
            if model_results_dir.exists():
                arm_dirs = [d for d in model_results_dir.iterdir() if d.is_dir()]
                
                for arm_dir in arm_dirs:
                    # Extract arm name and seed from directory name
                    dir_name = arm_dir.name
                    if '_seed' in dir_name:
                        arm_name, seed_part = dir_name.rsplit('_seed', 1)
                        seed = int(seed_part)
                    else:
                        continue
                    
                    if arm_name not in model_eval_results:
                        model_eval_results[arm_name] = {}
                    
                    # Evaluate this specific run
                    model_path = arm_dir / 'best_model.pth'
                    if model_path.exists():
                        try:
                            eval_results = self.model_evaluator.evaluate_model(
                                model_path=str(model_path),
                                model_type=model_name
                            )
                            model_eval_results[arm_name][seed] = eval_results
                        except Exception as e:
                            self.logger.error(f"Failed to evaluate {arm_dir}: {e}")
                            continue
            
            all_evaluation_results[model_name] = model_eval_results
        
        # Aggregate results across seeds for each arm
        aggregated_results = self._aggregate_across_seeds(all_evaluation_results)
        
        # Perform statistical comparisons
        statistical_results = {}
        for model_name in models:
            if model_name in aggregated_results:
                comparisons = self.statistical_analyzer.multiple_comparisons(
                    all_results=aggregated_results[model_name],
                    baseline_arm='R'
                )
                
                # Apply Bonferroni correction
                corrected_comparisons = self.statistical_analyzer.bonferroni_correction(comparisons)
                
                statistical_results[model_name] = corrected_comparisons
        
        # Compile comprehensive results
        comprehensive_results = {
            'evaluation_results': all_evaluation_results,
            'aggregated_results': aggregated_results,
            'statistical_comparisons': statistical_results,
            'models': models,
            'seeds': seeds,
            'test_data_dir': str(self.test_data_dir),
            'evaluation_time': datetime.now().isoformat()
        }
        
        # Save results
        with open(self.results_output_dir / 'comprehensive_evaluation.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2, cls=NumpyEncoder)
        
        # Create summary reports
        self._create_summary_reports(comprehensive_results)
        
        return comprehensive_results
    
    def _aggregate_across_seeds(self, 
                              all_results: Dict[str, Dict[str, Dict[int, Dict]]]) -> Dict[str, Dict[str, Dict]]:
        """Aggregate results across seeds for each model and arm"""
        aggregated = {}
        
        for model_name, model_results in all_results.items():
            aggregated[model_name] = {}
            
            for arm_name, arm_seeds in model_results.items():
                if not arm_seeds:  # Skip empty arms
                    continue
                
                # Aggregate across seeds
                aggregated_arm = {}
                metrics = ['dice', 'iou', 'precision', 'recall', 'f1', 'boundary_f1', 'hd95']
                
                for metric in metrics:
                    all_values = []
                    for seed_results in arm_seeds.values():
                        if metric in seed_results:
                            all_values.extend(seed_results[metric])
                    
                    if all_values:
                        aggregated_arm[metric] = all_values
                
                if aggregated_arm:
                    aggregated[model_name][arm_name] = aggregated_arm
        
        return aggregated
    
    def _create_summary_reports(self, results: Dict[str, Any]):
        """Create human-readable summary reports"""
        # Create summary tables
        summary_data = []
        
        for model_name, comparisons in results['statistical_comparisons'].items():
            for arm_name, comparison in comparisons.items():
                for metric, metric_results in comparison['metrics'].items():
                    summary_data.append({
                        'Model': model_name,
                        'Arm': arm_name,
                        'Metric': metric,
                        'Baseline_Mean': metric_results['baseline_mean'],
                        'Treatment_Mean': metric_results['treatment_mean'],
                        'Difference': metric_results['difference_mean'],
                        'P_Value': metric_results['p_value'],
                        'Effect_Size': metric_results['effect_size_cohens_d'],
                        'Significant': metric_results['significant'],
                        'Significant_Bonferroni': metric_results['significant_bonferroni']
                    })
        
        # Save as CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(self.results_output_dir / 'statistical_summary.csv', index=False)
        
        # Create markdown report
        self._create_markdown_report(results)
        
        self.logger.info(f"Summary reports saved to {self.results_output_dir}")
    
    def _create_markdown_report(self, results: Dict[str, Any]):
        """Create comprehensive markdown report"""
        report_lines = [
            "# Modality Agnostic Controlled Augmentation Study - Evaluation Report",
            "",
            f"**Evaluation Date:** {results['evaluation_time']}",
            f"**Models Evaluated:** {', '.join(results['models'])}",
            f"**Seeds Used:** {results['seeds']}",
            "",
            "## Summary",
            "",
            "This report presents the results of a comprehensive evaluation of synthetic data augmentation",
            "using cascaded diffusion models for cell microscopy image segmentation.",
            "",
            "### Dataset Arms",
            "- **R (Real-only):** Original training set",
            "- **RxS@10/25/50:** Replace 10%/25%/50% of training images with synthetic",
            "- **S (Synthetic-only):** Synthetic pairs equal in size to R", 
            "- **Rmask+SynthTex@25:** 25% real masks with synthetic textures",
            "",
            "## Statistical Results",
            ""
        ]
        
        for model_name, comparisons in results['statistical_comparisons'].items():
            report_lines.extend([
                f"### {model_name.upper()} Model Results",
                "",
                "| Arm | Metric | Baseline Mean | Treatment Mean | Difference | P-Value | Effect Size | Significant | Bonferroni |",
                "|-----|--------|---------------|----------------|------------|---------|-------------|-------------|------------|"
            ])
            
            for arm_name, comparison in comparisons.items():
                for metric, metric_results in comparison['metrics'].items():
                    baseline_mean = f"{metric_results['baseline_mean']:.4f}"
                    treatment_mean = f"{metric_results['treatment_mean']:.4f}"
                    difference = f"{metric_results['difference_mean']:.4f}"
                    p_value = f"{metric_results['p_value']:.4f}"
                    effect_size = f"{metric_results['effect_size_cohens_d']:.4f}"
                    significant = "✓" if metric_results['significant'] else "✗"
                    bonferroni = "✓" if metric_results['significant_bonferroni'] else "✗"
                    
                    report_lines.append(
                        f"| {arm_name} | {metric} | {baseline_mean} | {treatment_mean} | "
                        f"{difference} | {p_value} | {effect_size} | {significant} | {bonferroni} |"
                    )
            
            report_lines.append("")
        
        report_lines.extend([
            "## Interpretation",
            "",
            "- **Positive differences** indicate improvement over baseline (R)",
            "- **Effect sizes** > 0.2 (small), > 0.5 (medium), > 0.8 (large)",
            "- **Bonferroni correction** applied for multiple comparisons",
            "",
            "## Conclusions",
            "",
            "Based on the statistical analysis, the following conclusions can be drawn:",
            "1. [Add specific conclusions based on results]",
            "2. [Add recommendations for best augmentation strategy]",
            "3. [Add insights about synthetic data effectiveness]"
        ])
        
        # Save markdown report
        with open(self.results_output_dir / 'evaluation_report.md', 'w') as f:
            f.write('\n'.join(report_lines))


def main():
    """Test function for evaluation framework"""
    print("Evaluation Framework module loaded successfully")


if __name__ == "__main__":
    main()
