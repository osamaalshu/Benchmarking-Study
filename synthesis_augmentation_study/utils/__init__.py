"""
Utility modules for the synthesis augmentation study.

This package contains:
- data_arms_manager: Dataset arm creation and management
- training_protocol: Unified training protocol for all models
- model_wrappers: Model instantiation and dataset handling
- evaluation_framework: Comprehensive evaluation and statistical analysis
- generate_500_final: Pix2Pix synthetic data generation
- test_pix2pix: Pix2Pix model testing
- train_pix2pix: Pix2Pix model training
- setup_external: External repository setup
- generate_quality_image: Quality assessment utilities
- prepare_pix2pix_dataset: Dataset preparation for Pix2Pix
- _shared: Shared utilities and constants
"""

from .data_arms_manager import DataArmsManager
from .training_protocol import MultiModelTrainer
from .model_wrappers import get_model_creator, CellDataset
from .evaluation_framework import ComprehensiveEvaluator

__all__ = [
    'DataArmsManager',
    'MultiModelTrainer', 
    'get_model_creator',
    'CellDataset',
    'ComprehensiveEvaluator'
]
