"""
Calibration and confidence analysis for segmentation models
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.calibration import calibration_curve
import logging

try:
    from config.analysis_config import ANALYSIS_CONFIG, IMAGE_CONFIG
except ImportError:
    from analysis_config import ANALYSIS_CONFIG, IMAGE_CONFIG

class CalibrationAnalyzer:
    """Analyzes model calibration and confidence through reliability diagrams and ECE"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = ANALYSIS_CONFIG['calibration']
        
    def compute_ece(self, confidences: np.ndarray, predictions: np.ndarray, 
                   ground_truth: np.ndarray, n_bins: int = None) -> Dict:
        """
        Compute Expected Calibration Error (ECE)
        
        Args:
            confidences: Model confidence scores [0, 1]
            predictions: Binary predictions {0, 1}
            ground_truth: Ground truth binary labels {0, 1}
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary with ECE metrics
        """
        if n_bins is None:
            n_bins = self.config['n_bins']
        
        # Flatten arrays if needed
        confidences = confidences.flatten()
        predictions = predictions.flatten()
        ground_truth = ground_truth.flatten()
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        bin_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = ground_truth[in_bin].mean()
                
                # Average confidence in this bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'avg_confidence': avg_confidence_in_bin,
                    'accuracy': accuracy_in_bin,
                    'proportion': prop_in_bin,
                    'count': in_bin.sum()
                })
            else:
                bin_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'avg_confidence': 0,
                    'accuracy': 0,
                    'proportion': 0,
                    'count': 0
                })
        
        return {
            'ece': ece,
            'n_bins': n_bins,
            'bin_data': bin_data,
            'total_samples': len(confidences)
        }
    
    def compute_reliability_diagram_data(self, confidences: np.ndarray, 
                                       ground_truth: np.ndarray, 
                                       n_bins: int = None) -> Dict:
        """
        Compute data for reliability diagram
        
        Args:
            confidences: Model confidence scores [0, 1]
            ground_truth: Ground truth binary labels {0, 1}
            n_bins: Number of bins
            
        Returns:
            Dictionary with reliability diagram data
        """
        if n_bins is None:
            n_bins = self.config['n_bins']
        
        # Use sklearn's calibration_curve for reliability diagram
        fraction_of_positives, mean_predicted_value = calibration_curve(
            ground_truth.flatten(), 
            confidences.flatten(), 
            n_bins=n_bins
        )
        
        return {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'n_bins': n_bins
        }
    
    def analyze_confidence_distribution(self, confidences: np.ndarray) -> Dict:
        """Analyze the distribution of confidence scores"""
        confidences_flat = confidences.flatten()
        
        return {
            'mean_confidence': np.mean(confidences_flat),
            'std_confidence': np.std(confidences_flat),
            'min_confidence': np.min(confidences_flat),
            'max_confidence': np.max(confidences_flat),
            'median_confidence': np.median(confidences_flat),
            'confidence_percentiles': {
                '25th': np.percentile(confidences_flat, 25),
                '75th': np.percentile(confidences_flat, 75),
                '90th': np.percentile(confidences_flat, 90),
                '95th': np.percentile(confidences_flat, 95)
            }
        }
    
    def compute_confidence_accuracy_correlation(self, confidences: np.ndarray,
                                              predictions: np.ndarray,
                                              ground_truth: np.ndarray) -> Dict:
        """Compute correlation between confidence and accuracy"""
        confidences_flat = confidences.flatten()
        correct_predictions = (predictions.flatten() == ground_truth.flatten()).astype(int)
        
        # Compute correlation
        correlation = np.corrcoef(confidences_flat, correct_predictions)[0, 1]
        
        # Analyze accuracy at different confidence levels
        accuracy_by_confidence = {}
        for threshold in self.config['confidence_thresholds']:
            high_confidence_mask = confidences_flat >= threshold
            if high_confidence_mask.sum() > 0:
                accuracy_at_threshold = correct_predictions[high_confidence_mask].mean()
                proportion_above_threshold = high_confidence_mask.mean()
            else:
                accuracy_at_threshold = 0
                proportion_above_threshold = 0
            
            accuracy_by_confidence[f'threshold_{threshold}'] = {
                'accuracy': accuracy_at_threshold,
                'proportion': proportion_above_threshold,
                'count': high_confidence_mask.sum()
            }
        
        return {
            'confidence_accuracy_correlation': correlation,
            'accuracy_by_confidence_threshold': accuracy_by_confidence
        }
    
    def analyze_overconfidence(self, confidences: np.ndarray, 
                             ground_truth: np.ndarray) -> Dict:
        """Analyze model overconfidence patterns"""
        confidences_flat = confidences.flatten()
        gt_flat = ground_truth.flatten()
        
        # Compute Brier score
        brier_score = np.mean((confidences_flat - gt_flat) ** 2)
        
        # Compute overconfidence: cases where confidence > actual accuracy
        bin_boundaries = np.linspace(0, 1, 11)  # 10 bins
        overconfidence_data = []
        
        for i in range(len(bin_boundaries) - 1):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences_flat >= bin_lower) & (confidences_flat < bin_upper)
            
            if in_bin.sum() > 0:
                avg_confidence = confidences_flat[in_bin].mean()
                actual_accuracy = gt_flat[in_bin].mean()
                overconfidence = avg_confidence - actual_accuracy
                
                overconfidence_data.append({
                    'bin_center': (bin_lower + bin_upper) / 2,
                    'avg_confidence': avg_confidence,
                    'actual_accuracy': actual_accuracy,
                    'overconfidence': overconfidence,
                    'count': in_bin.sum()
                })
        
        return {
            'brier_score': brier_score,
            'overconfidence_by_bin': overconfidence_data,
            'mean_overconfidence': np.mean([item['overconfidence'] for item in overconfidence_data])
        }
    
    def create_reliability_diagram(self, reliability_data: Dict, 
                                 model_name: str, 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Create and optionally save reliability diagram"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot reliability curve
        ax.plot(reliability_data['mean_predicted_value'], 
                reliability_data['fraction_of_positives'], 
                'o-', label=f'{model_name}', linewidth=2, markersize=8)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'Reliability Diagram - {model_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add ECE text
        ece_data = self.compute_ece(
            reliability_data.get('confidences', np.array([])),
            reliability_data.get('predictions', np.array([])),
            reliability_data.get('ground_truth', np.array([]))
        )
        ax.text(0.05, 0.95, f'ECE: {ece_data["ece"]:.4f}', 
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Reliability diagram saved to {save_path}")
        
        return fig
    
    def create_confidence_histogram(self, confidences: np.ndarray, 
                                  model_name: str,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Create confidence distribution histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        ax.hist(confidences.flatten(), bins=50, alpha=0.7, density=True, 
                edgecolor='black', linewidth=0.5)
        
        # Add statistics
        stats = self.analyze_confidence_distribution(confidences)
        ax.axvline(stats['mean_confidence'], color='red', linestyle='--', 
                  label=f'Mean: {stats["mean_confidence"]:.3f}')
        ax.axvline(stats['median_confidence'], color='orange', linestyle='--', 
                  label=f'Median: {stats["median_confidence"]:.3f}')
        
        # Formatting
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Confidence Distribution - {model_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confidence histogram saved to {save_path}")
        
        return fig
    
    def comprehensive_calibration_analysis(self, confidences: np.ndarray,
                                         predictions: np.ndarray,
                                         ground_truth: np.ndarray,
                                         model_name: str) -> Dict:
        """Perform comprehensive calibration analysis"""
        self.logger.info(f"Performing comprehensive calibration analysis for {model_name}")
        
        # Ensure binary ground truth for calibration analysis
        gt_binary = (ground_truth > 0).astype(int)
        pred_binary = (predictions > 0).astype(int)
        
        # Core calibration metrics
        ece_results = self.compute_ece(confidences, pred_binary, gt_binary)
        reliability_data = self.compute_reliability_diagram_data(confidences, gt_binary)
        
        # Confidence analysis
        confidence_stats = self.analyze_confidence_distribution(confidences)
        confidence_accuracy = self.compute_confidence_accuracy_correlation(
            confidences, pred_binary, gt_binary
        )
        overconfidence_analysis = self.analyze_overconfidence(confidences, gt_binary)
        
        return {
            'model_name': model_name,
            'ece': ece_results,
            'reliability_data': reliability_data,
            'confidence_distribution': confidence_stats,
            'confidence_accuracy_correlation': confidence_accuracy,
            'overconfidence_analysis': overconfidence_analysis,
            'summary': {
                'ece_score': ece_results['ece'],
                'mean_confidence': confidence_stats['mean_confidence'],
                'confidence_accuracy_corr': confidence_accuracy['confidence_accuracy_correlation'],
                'brier_score': overconfidence_analysis['brier_score'],
                'mean_overconfidence': overconfidence_analysis['mean_overconfidence']
            }
        }
    
    def compare_model_calibrations(self, calibration_results: Dict[str, Dict]) -> Dict:
        """Compare calibration across multiple models"""
        comparison = {
            'model_rankings': {},
            'metrics_comparison': {},
            'best_calibrated': None,
            'worst_calibrated': None
        }
        
        # Extract key metrics for comparison
        metrics = ['ece_score', 'brier_score', 'confidence_accuracy_corr', 'mean_overconfidence']
        
        for metric in metrics:
            model_scores = {}
            for model_name, results in calibration_results.items():
                score = results['summary'][metric]
                model_scores[model_name] = score
            
            # Rank models (lower is better for ECE and Brier, higher is better for correlation)
            if metric in ['ece_score', 'brier_score', 'mean_overconfidence']:
                ranked = sorted(model_scores.items(), key=lambda x: abs(x[1]))
            else:  # confidence_accuracy_corr
                ranked = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            comparison['model_rankings'][metric] = ranked
            comparison['metrics_comparison'][metric] = model_scores
        
        # Determine overall best and worst calibrated models
        ece_ranking = comparison['model_rankings']['ece_score']
        comparison['best_calibrated'] = ece_ranking[0][0]  # Model with lowest ECE
        comparison['worst_calibrated'] = ece_ranking[-1][0]  # Model with highest ECE
        
        return comparison
