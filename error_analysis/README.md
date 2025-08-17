# Error Analysis and Interpretability Framework

A comprehensive framework for analyzing and interpreting segmentation model performance on cell microscopy images. This framework provides systematic error categorization, multimodal clustering analysis, calibration assessment, and visual inspection tools.

## Overview

This framework addresses the need to understand **why** certain models perform better than others by:

1. **Error Categorization**: Systematic classification of segmentation errors (False Negatives, Under-segmentation, False Positives, Boundary Errors)
2. **Multimodal Analysis**: Clustering images into distinct modalities based on cell characteristics
3. **Calibration Analysis**: Assessment of model confidence and reliability through ECE and reliability diagrams
4. **Visual Inspection**: Comprehensive qualitative analysis with side-by-side comparisons

## Project Structure

```
Error Analysis and Interpretability/
├── config/
│   └── analysis_config.py          # Configuration parameters
├── src/
│   ├── data_loader.py              # Data loading utilities
│   ├── error_analyzer.py           # Error categorization system
│   ├── multimodal_analyzer.py      # Multimodal clustering analysis
│   ├── calibration_analyzer.py     # Calibration and confidence analysis
│   └── visual_inspector.py         # Visual inspection tools
├── scripts/
│   └── run_complete_analysis.py    # Main orchestrator script
├── results/
│   ├── error_categorization/       # Error analysis results
│   ├── multimodal_clustering/      # Clustering results
│   ├── calibration_analysis/       # Calibration analysis results
│   ├── visual_inspection/          # Generated visualizations
│   └── reports/                    # Comprehensive reports
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

1. **Clone or navigate to the framework directory**:

   ```bash
   cd "Error Analysis and Interpretability"
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data paths** in `config/analysis_config.py`:
   - Ensure paths point to your test images, ground truth, and model predictions
   - Modify model configurations as needed

## Quick Start

### Run Complete Analysis

```bash
python scripts/run_complete_analysis.py
```

This will execute all analysis phases and generate comprehensive results.

### Individual Components

You can also run individual analysis components:

```python
from src.data_loader import DataLoader
from src.error_analyzer import ErrorAnalyzer

# Initialize
data_loader = DataLoader()
error_analyzer = ErrorAnalyzer()

# Load data
image = data_loader.load_test_image("cell_00001.tiff")
ground_truth = data_loader.load_ground_truth("cell_00001.tiff")
prediction = data_loader.load_prediction("nnunet", "cell_00001.tiff")

# Analyze errors
error_analysis = error_analyzer.analyze_errors(ground_truth, prediction)
error_rates = error_analyzer.compute_error_rates(error_analysis)
```

## Analysis Components

### 1. Error Categorization

Systematically categorizes segmentation errors into:

- **False Negatives**: Completely missed cells
- **False Positives**: Incorrectly detected regions as cells
- **Under-segmentation**: Multiple adjacent cells merged into one
- **Over-segmentation**: Single cells split into multiple detections
- **Boundary Errors**: Misclassification of cell boundaries

**Key Metrics**:

- Error rates per model and image
- Spatial distribution of errors
- Error correlation with cell characteristics

### 2. Multimodal Analysis

Clusters test images into distinct modalities based on:

- **Morphological Features**: Cell count, size, shape descriptors
- **Texture Features**: GLCM properties, Local Binary Patterns, intensity statistics
- **Spatial Features**: Cell density, nearest neighbor distances, clustering patterns

**Outputs**:

- 4 distinct modalities identified automatically
- Feature importance analysis
- Performance comparison across modalities

### 3. Calibration Analysis

Assesses model confidence and reliability:

- **Expected Calibration Error (ECE)**: Measures calibration quality
- **Reliability Diagrams**: Visual assessment of calibration
- **Confidence Distribution**: Analysis of prediction confidence patterns
- **Overconfidence Detection**: Identification of systematic confidence biases

**Note**: Requires probability outputs from models. Current implementation includes placeholder synthetic data.

### 4. Visual Inspection

Creates comprehensive visual comparisons:

- **Side-by-side Comparisons**: Original image, ground truth, and all model predictions
- **Error Overlay Visualizations**: Highlighting different error types with color coding
- **Detailed Case Studies**: In-depth analysis of challenging cases
- **Modality Comparisons**: Visual examples of different cell types
- **Performance Heatmaps**: Model performance across modalities

## Configuration

Key parameters can be modified in `config/analysis_config.py`:

```python
# Error analysis parameters
'error_analysis': {
    'min_cell_area': 10,           # Minimum pixels for valid cell
    'boundary_thickness': 2,       # Pixels for boundary analysis
    'overlap_threshold': 0.5,      # IoU threshold for cell matching
}

# Clustering parameters
'clustering': {
    'n_clusters': 4,               # Number of modalities
    'feature_types': ['morphological', 'texture', 'spatial'],
    'random_state': 42
}

# Calibration parameters
'calibration': {
    'n_bins': 10,                  # Bins for reliability diagrams
    'confidence_thresholds': [0.1, 0.3, 0.5, 0.7, 0.9]
}
```

## Expected Outputs

### 1. Error Analysis Results

- `error_summary.csv`: Error metrics for all model-image pairs
- `detailed_error_analysis.json`: Complete error categorization results

### 2. Multimodal Analysis Results

- `extracted_features.csv`: Feature vectors for all images
- `clustering_results.json`: Clustering analysis and metrics
- `modality_assignments.json`: Cluster assignment for each image

### 3. Calibration Analysis Results

- `calibration_results.json`: ECE scores and calibration metrics
- `[model]_reliability_diagram.png`: Reliability diagrams per model
- `[model]_confidence_histogram.png`: Confidence distributions per model

### 4. Visual Inspection Results

- `[image]_comparison.png`: Side-by-side model comparisons
- `[image]_error_overlay.png`: Error overlay visualizations
- `[image]_case_study.png`: Detailed case study analysis
- `modality_comparison.png`: Examples from each modality

### 5. Comprehensive Report

- `comprehensive_report.md`: Executive summary with key findings and recommendations

## Customization

### Adding New Models

1. Update `MODELS` dictionary in `config/analysis_config.py`
2. Ensure prediction files follow naming convention: `{model_name}/{image_name}_label.tiff`

### Adding New Error Types

1. Extend `ErrorAnalyzer` class with new analysis methods
2. Update error rate computation and visualization components

### Modifying Clustering Features

1. Add new feature extraction methods to `MultimodalAnalyzer`
2. Update feature importance analysis accordingly

## Troubleshooting

### Common Issues

1. **File Not Found Errors**:

   - Verify data paths in `config/analysis_config.py`
   - Check file naming conventions match expected patterns

2. **Memory Issues with Large Datasets**:

   - Process images in batches
   - Reduce image resolution if necessary

3. **Clustering Convergence Issues**:
   - Adjust `n_clusters` parameter
   - Try different random seeds
   - Check for missing or infinite feature values

### Performance Optimization

- Use parallel processing for independent image analyses
- Cache loaded images to avoid repeated disk I/O
- Consider dimensionality reduction for large feature sets

## Contributing

To extend the framework:

1. Follow the modular design pattern
2. Add comprehensive logging
3. Include error handling for edge cases
4. Update configuration files for new parameters
5. Add visualization components for new analyses

## Citation

If you use this framework in your research, please cite:

```
Error Analysis and Interpretability Framework for Cell Segmentation Models
[Your Institution/Publication Details]
```

## License

[Specify your license here]

---

For questions or issues, please contact [your contact information].
