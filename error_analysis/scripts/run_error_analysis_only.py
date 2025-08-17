#!/usr/bin/env python3
"""
Run comprehensive error analysis without multimodal clustering
Focus on error categorization, calibration, and visual inspection
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add src to path
framework_dir = Path(__file__).parent.parent
sys.path.insert(0, str(framework_dir / 'src'))
sys.path.insert(0, str(framework_dir))

from data_loader import DataLoader
from error_analyzer import ErrorAnalyzer
from calibration_analyzer import CalibrationAnalyzer
from visual_inspector import VisualInspector
from config.analysis_config import *

def setup_logging():
    """Setup logging configuration"""
    log_dir = ERROR_ANALYSIS_DIR / "logs"
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

def run_error_categorization(data_loader: DataLoader, logger: logging.Logger):
    """Run error categorization analysis for all models and images"""
    logger.info("Starting error categorization analysis...")
    
    error_analyzer = ErrorAnalyzer(logger)
    test_images = data_loader.get_test_image_list()
    
    # Results storage
    all_error_results = {}
    error_summary_data = []
    detailed_errors_by_image = {}
    
    for model_name in MODELS.keys():
        logger.info(f"Analyzing errors for {model_name}...")
        model_errors = {}
        
        for i, image_name in enumerate(test_images):
            if i % 20 == 0:
                logger.info(f"  Processing {model_name}: {i+1}/{len(test_images)} images")
                
            try:
                # Load data
                ground_truth = data_loader.load_ground_truth(image_name)
                prediction = data_loader.load_prediction(model_name, image_name)
                
                # Analyze errors
                error_analysis = error_analyzer.analyze_errors(ground_truth, prediction)
                error_rates = error_analyzer.compute_error_rates(error_analysis)
                
                model_errors[image_name] = {
                    'error_analysis': error_analysis,
                    'error_rates': error_rates
                }
                
                # Add to summary data
                summary_row = {
                    'model': model_name,
                    'image': image_name,
                    'false_negative_count': error_analysis['false_negatives']['count'],
                    'false_positive_count': error_analysis['false_positives']['count'],
                    'under_segmentation_count': error_analysis['under_segmentation']['count'],
                    'over_segmentation_count': error_analysis['over_segmentation']['count'],
                    'false_negative_rate': error_rates['false_negative_rate'],
                    'false_positive_rate': error_rates['false_positive_rate'],
                    'under_segmentation_rate': error_rates['under_segmentation_rate'],
                    'over_segmentation_rate': error_rates['over_segmentation_rate'],
                    'boundary_f1': error_analysis['boundary_errors']['boundary_f1'],
                    'gt_num_cells': error_analysis['summary']['gt_num_cells'],
                    'pred_num_cells': error_analysis['summary']['pred_num_cells'],
                    'cell_count_error': error_analysis['summary']['cell_count_error']
                }
                error_summary_data.append(summary_row)
                
                # Store detailed errors by image for visualization
                if image_name not in detailed_errors_by_image:
                    detailed_errors_by_image[image_name] = {}
                detailed_errors_by_image[image_name][model_name] = error_analysis
                
            except Exception as e:
                logger.error(f"Error analyzing {model_name} on {image_name}: {str(e)}")
                continue
        
        all_error_results[model_name] = model_errors
    
    # Save results
    error_results_dir = ERROR_CATEGORIZATION_DIR
    error_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary CSV
    error_summary_df = pd.DataFrame(error_summary_data)
    error_summary_df.to_csv(error_results_dir / "error_summary.csv", index=False)
    
    # Create analysis by model
    model_analysis = {}
    for model_name in MODELS.keys():
        model_data = error_summary_df[error_summary_df['model'] == model_name]
        if len(model_data) > 0:
            model_analysis[model_name] = {
                'total_images': len(model_data),
                'avg_false_negative_rate': model_data['false_negative_rate'].mean(),
                'avg_false_positive_rate': model_data['false_positive_rate'].mean(),
                'avg_under_segmentation_rate': model_data['under_segmentation_rate'].mean(),
                'avg_over_segmentation_rate': model_data['over_segmentation_rate'].mean(),
                'avg_boundary_f1': model_data['boundary_f1'].mean(),
                'avg_cell_count_error': model_data['cell_count_error'].mean(),
                'worst_images': model_data.nlargest(5, 'false_negative_rate')[['image', 'false_negative_rate', 'false_positive_rate']].to_dict('records'),
                'best_images': model_data.nsmallest(5, 'false_negative_rate')[['image', 'false_negative_rate', 'false_positive_rate']].to_dict('records')
            }
    
    # Save model analysis
    with open(error_results_dir / "model_error_analysis.json", 'w') as f:
        json.dump(model_analysis, f, indent=2)
    
    # Find most challenging images (high error rates across models)
    challenging_images = error_summary_df.groupby('image').agg({
        'false_negative_rate': 'mean',
        'false_positive_rate': 'mean',
        'under_segmentation_rate': 'mean',
        'over_segmentation_rate': 'mean',
        'boundary_f1': 'mean',
        'gt_num_cells': 'first'
    }).reset_index()
    
    challenging_images['total_error_rate'] = (
        challenging_images['false_negative_rate'] + 
        challenging_images['false_positive_rate'] + 
        challenging_images['under_segmentation_rate'] + 
        challenging_images['over_segmentation_rate']
    )
    
    challenging_images = challenging_images.sort_values('total_error_rate', ascending=False)
    challenging_images.to_csv(error_results_dir / "challenging_images.csv", index=False)
    
    logger.info(f"Error categorization completed. Results saved to {error_results_dir}")
    return all_error_results, detailed_errors_by_image, challenging_images

def run_visual_inspection(data_loader: DataLoader, error_results: dict, 
                         challenging_images: pd.DataFrame, logger: logging.Logger):
    """Run visual inspection focusing on most challenging cases"""
    logger.info("Starting visual inspection...")
    
    visual_inspector = VisualInspector(logger)
    
    visual_dir = VISUAL_INSPECTION_DIR
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    # Select images for detailed analysis
    # Top 10 most challenging + 5 random good ones for comparison
    top_challenging = challenging_images.head(10)['image'].tolist()
    good_performing = challenging_images.tail(10)['image'].tolist()
    selected_images = top_challenging + good_performing[:5]
    
    logger.info(f"Creating detailed visualizations for {len(selected_images)} images...")
    
    for i, image_name in enumerate(selected_images):
        logger.info(f"Processing visualization {i+1}/{len(selected_images)}: {image_name}")
        
        try:
            # Load data
            image = data_loader.load_test_image(image_name)
            ground_truth = data_loader.load_ground_truth(image_name)
            
            # Load all predictions
            predictions = {}
            image_error_analyses = {}
            
            for model_name in MODELS.keys():
                try:
                    prediction = data_loader.load_prediction(model_name, image_name)
                    predictions[model_name] = prediction
                    
                    if model_name in error_results and image_name in error_results[model_name]:
                        image_error_analyses[model_name] = error_results[model_name][image_name]['error_analysis']
                
                except Exception as e:
                    logger.warning(f"Could not load prediction for {model_name} on {image_name}: {str(e)}")
                    continue
            
            if not predictions:
                continue
            
            # Create enhanced side-by-side comparison with error overlays
            comparison_fig = visual_inspector.create_side_by_side_comparison(
                image, ground_truth, predictions, image_error_analyses, image_name,
                save_path=visual_dir / f"{Path(image_name).stem}_comparison.png"
            )
            plt.close(comparison_fig)
            
            # Create error overlay comparison
            error_overlay_fig = visual_inspector.create_error_overlay_comparison(
                image, ground_truth, predictions, image_name,
                save_path=visual_dir / f"{Path(image_name).stem}_error_overlay.png"
            )
            plt.close(error_overlay_fig)
            
            # Create detailed case study
            if image_error_analyses:
                case_study_fig = visual_inspector.create_detailed_case_study(
                    image, ground_truth, predictions, image_error_analyses, image_name,
                    save_path=visual_dir / f"{Path(image_name).stem}_case_study.png"
                )
                plt.close(case_study_fig)
            
        except Exception as e:
            logger.error(f"Error creating visuals for {image_name}: {str(e)}")
            continue
    
    # Create summary visualization showing error patterns
    create_error_pattern_summary(error_results, visual_dir, logger)
    
    logger.info(f"Visual inspection completed. Results saved to {visual_dir}")

def create_error_pattern_summary(error_results: dict, visual_dir: Path, logger: logging.Logger):
    """Create summary plots showing error patterns across models"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    logger.info("Creating error pattern summary plots...")
    
    # Aggregate error data
    model_error_summary = {}
    for model_name, model_data in error_results.items():
        error_counts = {
            'false_negatives': [],
            'false_positives': [],
            'under_segmentation': [],
            'over_segmentation': []
        }
        
        for image_data in model_data.values():
            error_analysis = image_data['error_analysis']
            error_counts['false_negatives'].append(error_analysis['false_negatives']['count'])
            error_counts['false_positives'].append(error_analysis['false_positives']['count'])
            error_counts['under_segmentation'].append(error_analysis['under_segmentation']['count'])
            error_counts['over_segmentation'].append(error_analysis['over_segmentation']['count'])
        
        model_error_summary[model_name] = {
            'avg_false_negatives': np.mean(error_counts['false_negatives']),
            'avg_false_positives': np.mean(error_counts['false_positives']),
            'avg_under_segmentation': np.mean(error_counts['under_segmentation']),
            'avg_over_segmentation': np.mean(error_counts['over_segmentation'])
        }
    
    # Create error pattern comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    error_types = ['avg_false_negatives', 'avg_false_positives', 'avg_under_segmentation', 'avg_over_segmentation']
    titles = ['Average False Negatives', 'Average False Positives', 'Average Under-segmentation', 'Average Over-segmentation']
    
    for idx, (error_type, title) in enumerate(zip(error_types, titles)):
        ax = axes[idx // 2, idx % 2]
        
        models = list(model_error_summary.keys())
        values = [model_error_summary[model][error_type] for model in models]
        colors = [MODELS[model]['color'] for model in models]
        
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Count per Image')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODELS[model]['name'] for model in models], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Error Pattern Comparison Across Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = visual_dir / "error_patterns_summary.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Error pattern summary saved: {save_path}")

def generate_comprehensive_report(error_results: dict, challenging_images: pd.DataFrame, 
                                logger: logging.Logger):
    """Generate comprehensive error analysis report"""
    logger.info("Generating comprehensive report...")
    
    reports_dir = REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Create markdown report
    report_content = f"""# Comprehensive Error Analysis Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report presents a detailed error analysis of {len(MODELS)} segmentation models on {len(challenging_images)} test images, focusing on understanding specific failure modes rather than just overall performance metrics.

## Methodology

### Error Categories Analyzed
1. **False Negatives**: Completely missed cells in ground truth
2. **False Positives**: Incorrectly detected cells (noise/artifacts)  
3. **Under-segmentation**: Multiple adjacent cells merged into single detection
4. **Over-segmentation**: Single cells incorrectly split into multiple detections
5. **Boundary Errors**: Misclassification of cell boundaries

## Key Findings

### Most Challenging Images
The following images showed consistently high error rates across all models:

"""
    
    # Add top 10 challenging images
    for i, row in challenging_images.head(10).iterrows():
        report_content += f"**{i+1}. {row['image']}**\n"
        report_content += f"   - Ground Truth Cells: {row['gt_num_cells']:.0f}\n"
        report_content += f"   - Avg False Negative Rate: {row['false_negative_rate']:.3f}\n"
        report_content += f"   - Avg False Positive Rate: {row['false_positive_rate']:.3f}\n"
        report_content += f"   - Total Error Rate: {row['total_error_rate']:.3f}\n\n"
    
    report_content += """
### Error Pattern Analysis

#### Model-Specific Insights
"""
    
    # Add model-specific analysis
    for model_name in MODELS.keys():
        if model_name in error_results:
            model_data = []
            for image_data in error_results[model_name].values():
                rates = image_data['error_rates']
                model_data.append(rates)
            
            if model_data:
                avg_fn = np.mean([d['false_negative_rate'] for d in model_data])
                avg_fp = np.mean([d['false_positive_rate'] for d in model_data])
                avg_under = np.mean([d['under_segmentation_rate'] for d in model_data])
                avg_over = np.mean([d['over_segmentation_rate'] for d in model_data])
                
                report_content += f"""
**{MODELS[model_name]['name']}**
- Primary failure mode: {"Missing cells" if avg_fn > max(avg_fp, avg_under, avg_over) else "False detections" if avg_fp > max(avg_under, avg_over) else "Under-segmentation" if avg_under > avg_over else "Over-segmentation"}
- False Negative Rate: {avg_fn:.3f}
- False Positive Rate: {avg_fp:.3f}  
- Under-segmentation Rate: {avg_under:.3f}
- Over-segmentation Rate: {avg_over:.3f}
"""
    
    report_content += f"""

## Visual Analysis Available

### Generated Visualizations
- **Comparison Images**: Side-by-side view of all model predictions vs ground truth
- **Error Overlays**: Color-coded error highlighting (Green=Correct, Red=False Positive, Blue=False Negative)
- **Case Studies**: Detailed analysis of challenging images with error statistics
- **Error Pattern Summary**: Aggregate error patterns across all models

### Files for Manual Inspection
All visualizations are saved in `results/visual_inspection/` directory:

- `*_comparison.png`: Direct model comparisons
- `*_error_overlay.png`: Error type highlighting  
- `*_case_study.png`: Detailed error analysis
- `error_patterns_summary.png`: Overall error patterns

## Recommendations for Further Analysis

### Manual Modality Analysis
Based on the visual results, you can manually identify patterns such as:
1. **Image characteristics** where certain models consistently fail
2. **Cell density patterns** that correlate with error types
3. **Morphological features** that distinguish challenging cases
4. **Potential modality clusters** based on visual similarity

### Model Improvement Strategies
1. **Target training** on the most challenging image types identified
2. **Error-specific loss functions** based on dominant failure modes
3. **Ensemble approaches** combining models with complementary strengths
4. **Post-processing rules** to address systematic errors

## Files Generated
- `error_summary.csv`: Complete error metrics for all model-image pairs
- `model_error_analysis.json`: Aggregated analysis by model
- `challenging_images.csv`: Images ranked by error rates
- Visual comparison images in `visual_inspection/` directory

---

*This analysis provides the foundation for understanding model behavior beyond simple performance metrics, enabling targeted improvements and deeper insights into segmentation challenges.*
"""
    
    # Save report
    with open(reports_dir / "error_analysis_report.md", 'w') as f:
        f.write(report_content)
    
    logger.info(f"Comprehensive report generated: {reports_dir / 'error_analysis_report.md'}")

def main():
    """Main execution function"""
    logger = setup_logging()
    logger.info("Starting comprehensive error analysis (without multimodal clustering)...")
    
    try:
        # Initialize data loader
        data_loader = DataLoader(logger)
        
        # Phase 1: Error Categorization
        logger.info("="*60)
        logger.info("PHASE 1: ERROR CATEGORIZATION")
        logger.info("="*60)
        error_results, detailed_errors_by_image, challenging_images = run_error_categorization(data_loader, logger)
        
        # Phase 2: Visual Inspection
        logger.info("="*60)
        logger.info("PHASE 2: VISUAL INSPECTION")
        logger.info("="*60)
        run_visual_inspection(data_loader, error_results, challenging_images, logger)
        
        # Phase 3: Generate Report
        logger.info("="*60)
        logger.info("PHASE 3: COMPREHENSIVE REPORT")
        logger.info("="*60)
        generate_comprehensive_report(error_results, challenging_images, logger)
        
        logger.info("="*60)
        logger.info("✅ ERROR ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"📁 Results available in: {ERROR_ANALYSIS_DIR / 'results'}")
        logger.info(f"📊 Key outputs:")
        logger.info(f"   • error_summary.csv - Detailed error metrics")
        logger.info(f"   • challenging_images.csv - Most problematic images")
        logger.info(f"   • Visual comparisons in visual_inspection/")
        logger.info(f"   • Comprehensive report in reports/")
        logger.info(f"")
        logger.info(f"🔍 NEXT STEPS:")
        logger.info(f"   1. Review challenging_images.csv to identify patterns")
        logger.info(f"   2. Examine visual comparisons for the top challenging images")
        logger.info(f"   3. Look for modality patterns in the visual results")
        logger.info(f"   4. Use insights for targeted model improvements")
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
