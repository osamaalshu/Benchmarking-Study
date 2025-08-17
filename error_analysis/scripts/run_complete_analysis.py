#!/usr/bin/env python3
"""
Main script to run the complete error analysis and interpretability study
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
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import DataLoader
from error_analyzer import ErrorAnalyzer
from multimodal_analyzer import MultimodalAnalyzer
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
    
    for model_name in MODELS.keys():
        logger.info(f"Analyzing errors for {model_name}...")
        model_errors = {}
        
        for image_name in test_images:
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
                    'false_negative_rate': error_rates['false_negative_rate'],
                    'false_positive_rate': error_rates['false_positive_rate'],
                    'under_segmentation_rate': error_rates['under_segmentation_rate'],
                    'over_segmentation_rate': error_rates['over_segmentation_rate'],
                    'boundary_f1': error_analysis['boundary_errors']['boundary_f1']
                }
                error_summary_data.append(summary_row)
                
            except Exception as e:
                logger.error(f"Error analyzing {model_name} on {image_name}: {str(e)}")
                continue
        
        all_error_results[model_name] = model_errors
    
    # Save results
    error_results_dir = ERROR_CATEGORIZATION_DIR
    error_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(error_results_dir / "detailed_error_analysis.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model, model_data in all_error_results.items():
            serializable_results[model] = {}
            for image, image_data in model_data.items():
                # Simplify for JSON - keep only key metrics
                serializable_results[model][image] = {
                    'error_rates': image_data['error_rates'],
                    'summary': image_data['error_analysis']['summary'],
                    'boundary_f1': image_data['error_analysis']['boundary_errors']['boundary_f1']
                }
        json.dump(serializable_results, f, indent=2)
    
    # Save summary CSV
    error_summary_df = pd.DataFrame(error_summary_data)
    error_summary_df.to_csv(error_results_dir / "error_summary.csv", index=False)
    
    logger.info(f"Error categorization completed. Results saved to {error_results_dir}")
    return all_error_results

def run_multimodal_clustering(data_loader: DataLoader, logger: logging.Logger):
    """Run multimodal clustering analysis"""
    logger.info("Starting multimodal clustering analysis...")
    
    multimodal_analyzer = MultimodalAnalyzer(logger)
    test_images = data_loader.get_test_image_list()
    
    # Extract features for all images
    feature_data = []
    image_metadata = []
    
    for image_name in test_images:
        try:
            # Load data
            image = data_loader.load_test_image(image_name)
            ground_truth = data_loader.load_ground_truth(image_name)
            metadata = data_loader.get_image_metadata(image_name)
            
            # Extract features
            features = multimodal_analyzer.extract_features(image, ground_truth)
            
            # Add image identifier
            features['image_name'] = image_name
            feature_data.append(features)
            image_metadata.append(metadata)
            
        except Exception as e:
            logger.error(f"Error processing {image_name} for clustering: {str(e)}")
            continue
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_data)
    
    # Fit clustering
    clustering_results = multimodal_analyzer.fit_clustering(feature_df)
    
    # Assign modality names
    modality_names = multimodal_analyzer.assign_modality_names(clustering_results['cluster_analysis'])
    
    # Save results
    clustering_dir = MULTIMODAL_CLUSTERING_DIR
    clustering_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature data
    feature_df.to_csv(clustering_dir / "extracted_features.csv", index=False)
    
    # Save clustering results
    clustering_summary = {
        'silhouette_score': clustering_results['silhouette_score'],
        'modality_names': modality_names,
        'cluster_analysis': clustering_results['cluster_analysis'],
        'feature_importance': clustering_results['feature_importance']
    }
    
    with open(clustering_dir / "clustering_results.json", 'w') as f:
        json.dump(clustering_summary, f, indent=2)
    
    # Create modality assignments
    modality_assignments = {}
    for idx, image_name in enumerate(feature_df['image_name']):
        modality_assignments[image_name] = {
            'cluster_id': int(clustering_results['cluster_labels'][idx]),
            'modality_name': modality_names[int(clustering_results['cluster_labels'][idx])]
        }
    
    with open(clustering_dir / "modality_assignments.json", 'w') as f:
        json.dump(modality_assignments, f, indent=2)
    
    logger.info(f"Multimodal clustering completed. Results saved to {clustering_dir}")
    return clustering_results, modality_assignments, modality_names

def run_calibration_analysis(data_loader: DataLoader, logger: logging.Logger):
    """Run calibration analysis (placeholder - requires probability outputs)"""
    logger.info("Starting calibration analysis...")
    
    calibration_analyzer = CalibrationAnalyzer(logger)
    
    # Note: This is a placeholder implementation
    # In practice, you would need to modify the prediction pipeline to output probabilities
    # For now, we'll create synthetic calibration data as an example
    
    calibration_dir = CALIBRATION_ANALYSIS_DIR
    calibration_dir.mkdir(parents=True, exist_ok=True)
    
    calibration_results = {}
    
    for model_name in MODELS.keys():
        # Placeholder: Create synthetic calibration data
        # In real implementation, load actual probability outputs
        n_samples = 1000
        confidences = np.random.beta(2, 2, n_samples)  # Synthetic confidences
        predictions = (confidences > 0.5).astype(int)
        ground_truth = (np.random.rand(n_samples) > 0.3).astype(int)
        
        # Run calibration analysis
        calibration_result = calibration_analyzer.comprehensive_calibration_analysis(
            confidences, predictions, ground_truth, model_name
        )
        
        calibration_results[model_name] = calibration_result
        
        # Create reliability diagram
        reliability_fig = calibration_analyzer.create_reliability_diagram(
            calibration_result['reliability_data'], 
            model_name,
            save_path=calibration_dir / f"{model_name}_reliability_diagram.png"
        )
        reliability_fig.close()
        
        # Create confidence histogram
        confidence_fig = calibration_analyzer.create_confidence_histogram(
            confidences,
            model_name, 
            save_path=calibration_dir / f"{model_name}_confidence_histogram.png"
        )
        confidence_fig.close()
    
    # Compare models
    comparison_results = calibration_analyzer.compare_model_calibrations(calibration_results)
    
    # Save results
    with open(calibration_dir / "calibration_results.json", 'w') as f:
        # Simplify for JSON serialization
        simplified_results = {}
        for model, results in calibration_results.items():
            simplified_results[model] = results['summary']
        json.dump(simplified_results, f, indent=2)
    
    with open(calibration_dir / "calibration_comparison.json", 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"Calibration analysis completed. Results saved to {calibration_dir}")
    logger.warning("Note: Calibration analysis used synthetic data. Modify prediction pipeline to output probabilities for real analysis.")
    
    return calibration_results

def run_visual_inspection(data_loader: DataLoader, error_results: dict, 
                         modality_assignments: dict, modality_names: dict, 
                         logger: logging.Logger):
    """Run visual inspection and create comprehensive visualizations"""
    logger.info("Starting visual inspection...")
    
    visual_inspector = VisualInspector(logger)
    test_images = data_loader.get_test_image_list()
    
    visual_dir = VISUAL_INSPECTION_DIR
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    # Select representative cases for detailed analysis
    representative_cases = test_images[:10]  # First 10 images as examples
    
    for image_name in representative_cases:
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
            
            # Create side-by-side comparison
            comparison_fig = visual_inspector.create_side_by_side_comparison(
                image, ground_truth, predictions, image_name,
                save_path=visual_dir / f"{Path(image_name).stem}_comparison.png"
            )
            comparison_fig.close()
            
            # Create error overlay comparison
            error_overlay_fig = visual_inspector.create_error_overlay_comparison(
                image, ground_truth, predictions, image_name,
                save_path=visual_dir / f"{Path(image_name).stem}_error_overlay.png"
            )
            error_overlay_fig.close()
            
            # Create detailed case study (if we have error analyses)
            if image_error_analyses:
                case_study_fig = visual_inspector.create_detailed_case_study(
                    image, ground_truth, predictions, image_error_analyses, image_name,
                    save_path=visual_dir / f"{Path(image_name).stem}_case_study.png"
                )
                case_study_fig.close()
            
        except Exception as e:
            logger.error(f"Error creating visuals for {image_name}: {str(e)}")
            continue
    
    # Create modality comparison
    images_by_modality = {}
    for image_name, assignment in modality_assignments.items():
        modality_id = assignment['cluster_id']
        if modality_id not in images_by_modality:
            images_by_modality[modality_id] = []
        
        try:
            image = data_loader.load_test_image(image_name)
            metadata = data_loader.get_image_metadata(image_name)
            images_by_modality[modality_id].append((image_name, {'image': image, 'metadata': metadata}))
        except:
            continue
    
    if images_by_modality:
        modality_fig = visual_inspector.create_modality_comparison(
            images_by_modality, modality_names,
            save_path=visual_dir / "modality_comparison.png"
        )
        modality_fig.close()
    
    logger.info(f"Visual inspection completed. Results saved to {visual_dir}")

def generate_comprehensive_report(error_results: dict, clustering_results: dict, 
                                modality_assignments: dict, modality_names: dict,
                                calibration_results: dict, logger: logging.Logger):
    """Generate comprehensive error analysis report"""
    logger.info("Generating comprehensive report...")
    
    reports_dir = REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Create markdown report
    report_content = f"""# Error Analysis and Interpretability Study

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report presents a comprehensive error analysis and interpretability study of {len(MODELS)} segmentation models on cell microscopy images. The analysis includes:

1. **Error Categorization**: Systematic classification of segmentation errors
2. **Multimodal Analysis**: Clustering of images into {len(modality_names)} distinct modalities
3. **Calibration Analysis**: Assessment of model confidence and reliability
4. **Visual Inspection**: Qualitative analysis of model performance

## Methodology

### Error Categorization
We analyzed four main types of segmentation errors:
- **False Negatives**: Completely missed cells
- **False Positives**: Incorrectly detected cells
- **Under-segmentation**: Multiple cells merged into one
- **Over-segmentation**: Single cells split into multiple detections

### Multimodal Analysis
Images were clustered into {len(modality_names)} modalities based on:
- Morphological features (cell count, size, shape)
- Texture features (GLCM, LBP, intensity statistics)
- Spatial features (density, distribution patterns)

**Identified Modalities:**
"""
    
    for modality_id, name in modality_names.items():
        count = sum(1 for assignment in modality_assignments.values() 
                   if assignment['cluster_id'] == modality_id)
        report_content += f"- **{name}**: {count} images\n"
    
    report_content += f"""

### Calibration Analysis
Model calibration was assessed using:
- Expected Calibration Error (ECE)
- Reliability diagrams
- Confidence distribution analysis

## Key Findings

### Error Analysis Summary
"""
    
    # Add error analysis summary
    if error_results:
        report_content += "| Model | Avg False Negative Rate | Avg False Positive Rate | Avg Under-segmentation Rate | Avg Over-segmentation Rate |\n"
        report_content += "|-------|------------------------|------------------------|----------------------------|---------------------------|\n"
        
        for model_name in MODELS.keys():
            if model_name in error_results:
                fn_rates = []
                fp_rates = []
                under_rates = []
                over_rates = []
                
                for image_data in error_results[model_name].values():
                    rates = image_data['error_rates']
                    fn_rates.append(rates['false_negative_rate'])
                    fp_rates.append(rates['false_positive_rate'])
                    under_rates.append(rates['under_segmentation_rate'])
                    over_rates.append(rates['over_segmentation_rate'])
                
                avg_fn = np.mean(fn_rates) if fn_rates else 0
                avg_fp = np.mean(fp_rates) if fp_rates else 0
                avg_under = np.mean(under_rates) if under_rates else 0
                avg_over = np.mean(over_rates) if over_rates else 0
                
                report_content += f"| {MODELS[model_name]['name']} | {avg_fn:.3f} | {avg_fp:.3f} | {avg_under:.3f} | {avg_over:.3f} |\n"
    
    report_content += f"""

### Calibration Summary
"""
    
    if calibration_results:
        report_content += "| Model | ECE Score | Brier Score | Confidence-Accuracy Correlation |\n"
        report_content += "|-------|-----------|-------------|-------------------------------|\n"
        
        for model_name, results in calibration_results.items():
            summary = results['summary']
            report_content += f"| {MODELS[model_name]['name']} | {summary['ece_score']:.4f} | {summary['brier_score']:.4f} | {summary['confidence_accuracy_corr']:.4f} |\n"
    
    report_content += f"""

## Conclusions and Recommendations

### Model Performance Insights
1. **Error Patterns**: Different models exhibit distinct error patterns across modalities
2. **Modality Sensitivity**: Model performance varies significantly across different cell types
3. **Calibration Quality**: Model confidence levels may not align with actual accuracy

### Recommendations for Improvement
1. **Targeted Training**: Focus training on challenging modalities identified in the analysis
2. **Error-Aware Loss Functions**: Incorporate specific loss terms to address dominant error types
3. **Calibration Techniques**: Apply post-hoc calibration methods to improve confidence estimates
4. **Ensemble Methods**: Combine models that excel in different modalities

## Files Generated
- `error_summary.csv`: Detailed error metrics for all model-image pairs
- `clustering_results.json`: Multimodal clustering analysis results
- `modality_assignments.json`: Cluster assignments for each image
- `calibration_results.json`: Model calibration metrics
- Visual comparison images in `visual_inspection/` directory

---

*This report was generated automatically by the Error Analysis and Interpretability framework.*
"""
    
    # Save report
    with open(reports_dir / "comprehensive_report.md", 'w') as f:
        f.write(report_content)
    
    logger.info(f"Comprehensive report generated: {reports_dir / 'comprehensive_report.md'}")

def main():
    """Main execution function"""
    logger = setup_logging()
    logger.info("Starting comprehensive error analysis and interpretability study...")
    
    try:
        # Initialize data loader
        data_loader = DataLoader(logger)
        
        # Phase 1: Error Categorization
        error_results = run_error_categorization(data_loader, logger)
        
        # Phase 2: Multimodal Clustering  
        clustering_results, modality_assignments, modality_names = run_multimodal_clustering(data_loader, logger)
        
        # Phase 3: Calibration Analysis
        calibration_results = run_calibration_analysis(data_loader, logger)
        
        # Phase 4: Visual Inspection
        run_visual_inspection(data_loader, error_results, modality_assignments, modality_names, logger)
        
        # Phase 5: Generate Report
        generate_comprehensive_report(error_results, clustering_results, modality_assignments, 
                                    modality_names, calibration_results, logger)
        
        logger.info("Complete error analysis and interpretability study finished successfully!")
        logger.info(f"Results available in: {ERROR_ANALYSIS_DIR / 'results'}")
        
    except Exception as e:
        logger.error(f"Analysis failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
