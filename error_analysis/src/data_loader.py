"""
Data loading utilities for error analysis
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

from config.analysis_config import *

class DataLoader:
    """Handles loading of test images, ground truth, and model predictions"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.test_images = {}
        self.ground_truth = {}
        self.predictions = {model: {} for model in MODELS.keys()}
        
        # Validate paths
        self._validate_paths()
        
    def _validate_paths(self):
        """Validate that all required data paths exist"""
        required_paths = [TEST_IMAGES_PATH, TEST_LABELS_PATH, PREDICTIONS_PATH]
        
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required path does not exist: {path}")
                
        self.logger.info("All required data paths validated successfully")
        
    def load_test_image(self, image_name: str) -> np.ndarray:
        """Load a test image"""
        if image_name in self.test_images:
            return self.test_images[image_name]
            
        image_path = TEST_IMAGES_PATH / image_name
        if not image_path.exists():
            # Try different extensions
            base_name = Path(image_name).stem
            for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                potential_path = TEST_IMAGES_PATH / f"{base_name}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break
            else:
                raise FileNotFoundError(f"Test image not found: {image_name}")
        
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Resize to standard size if needed
        if image_array.shape[:2] != (IMAGE_CONFIG['input_size'], IMAGE_CONFIG['input_size']):
            image = image.resize((IMAGE_CONFIG['input_size'], IMAGE_CONFIG['input_size']), Image.LANCZOS)
            image_array = np.array(image)
            
        self.test_images[image_name] = image_array
        return image_array
    
    def load_ground_truth(self, image_name: str) -> np.ndarray:
        """Load ground truth label for an image"""
        if image_name in self.ground_truth:
            return self.ground_truth[image_name]
            
        # Convert image name to label name
        base_name = Path(image_name).stem
        label_name = f"{base_name}_label.tiff"
        label_path = TEST_LABELS_PATH / label_name
        
        if not label_path.exists():
            # Try different extensions and naming patterns
            for ext in ['.png', '.tiff', '.tif']:
                for pattern in [f"{base_name}_label{ext}", f"{base_name}{ext}"]:
                    potential_path = TEST_LABELS_PATH / pattern
                    if potential_path.exists():
                        label_path = potential_path
                        break
                if label_path.exists():
                    break
            else:
                raise FileNotFoundError(f"Ground truth not found for: {image_name}")
        
        label = Image.open(label_path).convert('L')
        label_array = np.array(label)
        
        # Convert instance segmentation to semantic segmentation
        # Background = 0, Cells = 1, Boundaries = 2
        if label_array.max() > 2:
            # This is an instance segmentation mask - convert to semantic
            semantic_mask = np.zeros_like(label_array, dtype=np.uint8)
            semantic_mask[label_array > 0] = 1  # All non-background pixels become cells
            label_array = semantic_mask
        
        # Resize if needed (do this AFTER conversion to avoid artifacts)
        if label_array.shape != (IMAGE_CONFIG['input_size'], IMAGE_CONFIG['input_size']):
            label = Image.fromarray(label_array)
            label = label.resize((IMAGE_CONFIG['input_size'], IMAGE_CONFIG['input_size']), Image.NEAREST)
            label_array = np.array(label)
        
        self.ground_truth[image_name] = label_array
        return label_array
    
    def load_prediction(self, model_name: str, image_name: str) -> np.ndarray:
        """Load model prediction for an image"""
        if image_name in self.predictions[model_name]:
            return self.predictions[model_name][image_name]
            
        model_config = MODELS[model_name]
        pred_dir = PREDICTIONS_PATH / model_config['predictions_dir']
        
        # Convert image name to prediction name
        base_name = Path(image_name).stem
        pred_name = f"{base_name}_label.tiff"
        pred_path = pred_dir / pred_name
        
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction not found: {pred_path}")
        
        pred = Image.open(pred_path).convert('L')
        pred_array = np.array(pred)
        
        # Convert instance segmentation to semantic segmentation
        # Background = 0, Cells = 1, Boundaries = 2  
        if pred_array.max() > 2:
            # This is an instance segmentation mask - convert to semantic
            semantic_mask = np.zeros_like(pred_array, dtype=np.uint8)
            semantic_mask[pred_array > 0] = 1  # All non-background pixels become cells
            pred_array = semantic_mask
        
        # Resize if needed (do this AFTER conversion to avoid artifacts)
        if pred_array.shape != (IMAGE_CONFIG['input_size'], IMAGE_CONFIG['input_size']):
            pred = Image.fromarray(pred_array)
            pred = pred.resize((IMAGE_CONFIG['input_size'], IMAGE_CONFIG['input_size']), Image.NEAREST)
            pred_array = np.array(pred)
        
        self.predictions[model_name][image_name] = pred_array
        return pred_array
    
    def get_test_image_list(self) -> List[str]:
        """Get list of all available test images"""
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
            image_files.extend(TEST_IMAGES_PATH.glob(ext))
        
        # Sort and return just the filenames
        image_names = [f.name for f in sorted(image_files)]
        self.logger.info(f"Found {len(image_names)} test images")
        return image_names
    
    def load_performance_metrics(self) -> Dict[str, pd.DataFrame]:
        """Load performance metrics for all models"""
        metrics = {}
        
        for model_name in MODELS.keys():
            csv_file = PREDICTIONS_PATH / f"{MODELS[model_name]['predictions_dir']}-0.5.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                metrics[model_name] = df
                self.logger.info(f"Loaded metrics for {model_name}: {len(df)} samples")
            else:
                self.logger.warning(f"Metrics file not found for {model_name}: {csv_file}")
        
        return metrics
    
    def get_image_metadata(self, image_name: str) -> Dict:
        """Get metadata for an image including ground truth statistics"""
        gt = self.load_ground_truth(image_name)
        
        # Count cells (connected components in foreground)
        from scipy import ndimage
        foreground = (gt > 0).astype(int)
        labeled_array, num_cells = ndimage.label(foreground)
        
        # Calculate cell statistics
        cell_areas = []
        if num_cells > 0:
            for i in range(1, num_cells + 1):
                cell_mask = (labeled_array == i)
                cell_areas.append(np.sum(cell_mask))
        
        metadata = {
            'image_name': image_name,
            'num_cells': num_cells,
            'mean_cell_area': np.mean(cell_areas) if cell_areas else 0,
            'total_cell_pixels': np.sum(gt > 0),
            'background_pixels': np.sum(gt == 0),
            'boundary_pixels': np.sum(gt == 2),
            'image_size': gt.shape
        }
        
        return metadata
