"""
Configuration file for Error Analysis and Interpretability Study
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
ERROR_ANALYSIS_DIR = BASE_DIR / "error_analysis"

# Data paths
TEST_IMAGES_PATH = BASE_DIR / "data" / "test" / "images"
TEST_LABELS_PATH = BASE_DIR / "data" / "test" / "labels"
PREDICTIONS_PATH = BASE_DIR / "test_predictions"
VISUALIZATION_PATH = BASE_DIR / "visualization_results"

# Output paths
RESULTS_DIR = ERROR_ANALYSIS_DIR / "results"
ERROR_CATEGORIZATION_DIR = RESULTS_DIR / "error_categorization"
MULTIMODAL_CLUSTERING_DIR = RESULTS_DIR / "multimodal_clustering"
CALIBRATION_ANALYSIS_DIR = RESULTS_DIR / "calibration_analysis"
VISUAL_INSPECTION_DIR = RESULTS_DIR / "visual_inspection"
REPORTS_DIR = RESULTS_DIR / "reports"

# Model configuration
MODELS = {
    'unet': {
        'name': 'UNET',
        'predictions_dir': 'unet',
        'color': '#1f77b4'
    },
    'nnunet': {
        'name': 'NNUNET', 
        'predictions_dir': 'nnunet',
        'color': '#ff7f0e'
    },
    'sac': {
        'name': 'SAC',
        'predictions_dir': 'sac', 
        'color': '#2ca02c'
    },
    'lstmunet': {
        'name': 'LSTMUNET',
        'predictions_dir': 'lstmunet',
        'color': '#d62728'
    },
    'maunet_resnet50': {
        'name': 'MAUNET_RESNET50',
        'predictions_dir': 'maunet_resnet50',
        'color': '#9467bd'
    },
    'maunet_wide': {
        'name': 'MAUNET_WIDE', 
        'predictions_dir': 'maunet_wide',
        'color': '#8c564b'
    },
    'maunet_ensemble': {
        'name': 'MAUNET_ENSEMBLE',
        'predictions_dir': 'maunet_ensemble',
        'color': '#e377c2'
    }
}

# Analysis parameters
ANALYSIS_CONFIG = {
    # Error categorization
    'error_analysis': {
        'min_cell_area': 10,  # Minimum pixels for a valid cell
        'boundary_thickness': 2,  # Pixels for boundary analysis
        'overlap_threshold': 0.5,  # IoU threshold for cell matching
    },
    
    # Multimodal clustering
    'clustering': {
        'n_clusters': 4,  # Number of modalities to identify
        'feature_types': ['morphological', 'texture', 'spatial'],
        'random_state': 42
    },
    
    # Calibration analysis
    'calibration': {
        'n_bins': 10,  # Number of bins for reliability diagrams
        'confidence_thresholds': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    
    # Visualization
    'visualization': {
        'figure_size': (15, 10),
        'dpi': 300,
        'alpha_overlay': 0.6,
        'colormap': 'viridis'
    }
}

# Image processing parameters
IMAGE_CONFIG = {
    'input_size': 256,
    'num_classes': 3,
    'class_names': ['Background', 'Cell Interior', 'Cell Boundary'],
    'class_colors': [(0, 0, 0), (255, 0, 0), (0, 255, 0)]  # RGB colors for visualization
}

# Statistical analysis parameters
STATS_CONFIG = {
    'alpha': 0.05,  # Significance level
    'bootstrap_iterations': 1000,
    'confidence_level': 0.95
}
