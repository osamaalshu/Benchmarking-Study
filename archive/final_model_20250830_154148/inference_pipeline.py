#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Inference Pipeline for Error-Aware MAUNet

This module implements sophisticated post-processing techniques that leverage
the auxiliary outputs from Error-Aware MAUNet for improved instance segmentation:

1. Seeded Watershed Segmentation: Uses centroid heatmaps as seeds and distance
   transforms to guide watershed-based instance separation
2. Multi-scale Test-Time Augmentation: Improves robustness through ensemble predictions
3. Confidence-based filtering: Removes low-confidence predictions to reduce false positives

The pipeline specifically addresses the systematic error patterns identified in
error analysis, particularly cell merging in dense arrangements and missed
small cells in low-contrast regions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max
from scipy import ndimage
import cv2

from error_aware_maunet import ErrorAwareMAUNet, ErrorAwareMAUNetEnsemble


class SeededWatershedProcessor:
    """
    Advanced post-processing using seeded watershed segmentation
    
    This processor leverages the auxiliary outputs from Error-Aware MAUNet:
    - Centroid heatmaps provide watershed seeds for instance separation
    - Distance transforms guide watershed topology for accurate boundaries
    - Segmentation probabilities provide foreground/background classification
    """
    
    def __init__(
        self,
        min_seed_distance: int = 10,
        min_seed_prominence: float = 0.1,
        watershed_threshold: float = 0.5,
        min_object_size: int = 16,
        max_hole_size: int = 64,
    ):
        """
        Args:
            min_seed_distance: Minimum distance between centroid peaks
            min_seed_prominence: Minimum prominence for peak detection
            watershed_threshold: Threshold for foreground segmentation
            min_object_size: Minimum size for connected components
            max_hole_size: Maximum size of holes to fill
        """
        self.min_seed_distance = min_seed_distance
        self.min_seed_prominence = min_seed_prominence
        self.watershed_threshold = watershed_threshold
        self.min_object_size = min_object_size
        self.max_hole_size = max_hole_size
    
    def extract_seeds_from_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Extract watershed seeds from centroid heatmap using peak detection
        
        Args:
            heatmap: (H, W) centroid heatmap with Gaussian peaks at cell centers
            
        Returns:
            seeds: (H, W) labeled seed mask for watershed
        """
        # Find local maxima in the heatmap
        peak_coords = peak_local_max(
            heatmap,
            min_distance=self.min_seed_distance,
            threshold_abs=self.min_seed_prominence
        )
        
        # Create binary mask from peak coordinates
        peaks = np.zeros_like(heatmap, dtype=bool)
        if len(peak_coords) > 0:
            peaks[peak_coords[:, 0], peak_coords[:, 1]] = True
        
        # Label connected components of peaks
        seeds = measure.label(peaks, connectivity=2)
        
        return seeds
    
    def refine_seeds_with_segmentation(
        self, 
        seeds: np.ndarray, 
        segmentation_prob: np.ndarray
    ) -> np.ndarray:
        """
        Refine watershed seeds using segmentation probabilities
        
        Args:
            seeds: (H, W) initial seed labels
            segmentation_prob: (H, W) foreground probability map
            
        Returns:
            refined_seeds: (H, W) refined seed labels
        """
        # Only keep seeds that are in high-confidence foreground regions
        foreground_mask = segmentation_prob > self.watershed_threshold
        
        refined_seeds = seeds.copy()
        unique_labels = np.unique(seeds)
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
                
            seed_mask = seeds == label
            # Check if seed overlaps with confident foreground
            if not np.any(seed_mask & foreground_mask):
                refined_seeds[seed_mask] = 0  # Remove seed
        
        # Relabel to ensure consecutive labels
        refined_seeds = measure.label(refined_seeds > 0, connectivity=2)
        
        return refined_seeds
    
    def apply_watershed_segmentation(
        self,
        distance_transform: np.ndarray,
        seeds: np.ndarray,
        segmentation_mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply watershed segmentation using distance transform and seeds
        
        Args:
            distance_transform: (H, W) signed distance transform
            seeds: (H, W) labeled seed mask
            segmentation_mask: (H, W) binary foreground mask
            
        Returns:
            watershed_result: (H, W) instance segmentation mask
        """
        # Use negative distance transform as watershed surface
        # (watershed finds minima, so we negate to find maxima/ridges)
        watershed_surface = -distance_transform
        
        # Apply watershed
        watershed_result = segmentation.watershed(
            watershed_surface,
            markers=seeds,
            mask=segmentation_mask,
            compactness=0.1  # Slight bias toward compact regions
        )
        
        return watershed_result
    
    def post_process_instances(self, instances: np.ndarray) -> np.ndarray:
        """
        Apply morphological post-processing to clean up instances
        
        Args:
            instances: (H, W) instance segmentation mask
            
        Returns:
            cleaned_instances: (H, W) cleaned instance mask
        """
        cleaned_instances = instances.copy()
        unique_labels = np.unique(instances)
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
                
            # Extract single instance
            instance_mask = instances == label
            
            # Remove small holes
            filled_mask = morphology.remove_small_holes(
                instance_mask, area_threshold=self.max_hole_size
            )
            
            # Check if instance is too small
            if np.sum(filled_mask) < self.min_object_size:
                cleaned_instances[instance_mask] = 0  # Remove instance
            else:
                # Update instance with filled holes
                cleaned_instances[instance_mask] = 0
                cleaned_instances[filled_mask] = label
        
        # Relabel to ensure consecutive labels
        cleaned_instances = measure.label(cleaned_instances > 0, connectivity=2)
        
        return cleaned_instances
    
    def process(
        self,
        segmentation_logits: np.ndarray,
        distance_transform: np.ndarray,
        centroid_heatmap: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Complete seeded watershed processing pipeline
        
        Args:
            segmentation_logits: (C, H, W) segmentation logits
            distance_transform: (H, W) signed distance transform
            centroid_heatmap: (H, W) centroid heatmap (optional)
            
        Returns:
            instance_mask: (H, W) final instance segmentation
        """
        # Convert logits to probabilities and binary mask
        if segmentation_logits.shape[0] > 1:
            # Multi-class: combine all foreground classes
            seg_probs = np.exp(segmentation_logits) / np.sum(np.exp(segmentation_logits), axis=0, keepdims=True)
            foreground_prob = np.sum(seg_probs[1:], axis=0)  # Sum all non-background classes
        else:
            # Binary segmentation
            foreground_prob = 1 / (1 + np.exp(-segmentation_logits[0]))  # Sigmoid
        
        # Create binary mask
        binary_mask = foreground_prob > self.watershed_threshold
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=self.min_object_size)
        binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=self.max_hole_size)
        
        # If no centroid heatmap provided, use distance transform peaks as seeds
        if centroid_heatmap is None:
            # Use local maxima in distance transform as seeds
            distance_maxima = distance_transform > 0  # Only positive (interior) regions
            peak_coords = peak_local_max(
                distance_transform * distance_maxima,
                min_distance=self.min_seed_distance,
                threshold_abs=0.1
            )
            
            # Create binary mask from peak coordinates
            distance_peaks = np.zeros_like(distance_transform, dtype=bool)
            if len(peak_coords) > 0:
                distance_peaks[peak_coords[:, 0], peak_coords[:, 1]] = True
            seeds = measure.label(distance_peaks, connectivity=2)
        else:
            # Extract seeds from centroid heatmap
            seeds = self.extract_seeds_from_heatmap(centroid_heatmap)
            
        # Refine seeds using segmentation confidence
        refined_seeds = self.refine_seeds_with_segmentation(seeds, foreground_prob)
        
        # Apply watershed segmentation
        if np.max(refined_seeds) > 0:  # Check if we have any seeds
            instance_mask = self.apply_watershed_segmentation(
                distance_transform, refined_seeds, binary_mask
            )
        else:
            # Fallback: use connected components if no seeds found
            instance_mask = measure.label(binary_mask, connectivity=2)
        
        # Post-process instances
        final_instances = self.post_process_instances(instance_mask)
        
        return final_instances


class TestTimeAugmentation:
    """
    Test-time augmentation for improved robustness
    
    Applies multiple transformations during inference and combines predictions
    to reduce variance and improve overall performance.
    """
    
    def __init__(
        self,
        enable_flip: bool = True,
        enable_rotation: bool = True,
        enable_scale: bool = False,
        scale_factors: List[float] = [0.9, 1.0, 1.1]
    ):
        self.enable_flip = enable_flip
        self.enable_rotation = enable_rotation
        self.enable_scale = enable_scale
        self.scale_factors = scale_factors
    
    def apply_augmentations(self, image: torch.Tensor) -> List[Tuple[torch.Tensor, str]]:
        """
        Apply augmentations to input image
        
        Args:
            image: (C, H, W) input image tensor
            
        Returns:
            augmented_images: List of (augmented_image, transform_name) pairs
        """
        augmented = [(image, "original")]
        
        if self.enable_flip:
            # Horizontal flip
            flipped_h = torch.flip(image, dims=[2])
            augmented.append((flipped_h, "flip_h"))
            
            # Vertical flip
            flipped_v = torch.flip(image, dims=[1])
            augmented.append((flipped_v, "flip_v"))
        
        if self.enable_rotation:
            # 90-degree rotations
            for k in [1, 2, 3]:
                rotated = torch.rot90(image, k, dims=[1, 2])
                augmented.append((rotated, f"rot90_{k}"))
        
        if self.enable_scale:
            # Multi-scale testing
            for scale in self.scale_factors:
                if scale != 1.0:
                    h, w = image.shape[1], image.shape[2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled = F.interpolate(
                        image.unsqueeze(0), 
                        size=(new_h, new_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                    augmented.append((scaled, f"scale_{scale}"))
        
        return augmented
    
    def reverse_augmentation(
        self, 
        outputs: Dict[str, torch.Tensor], 
        transform_name: str,
        original_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Reverse augmentation to align predictions with original image
        
        Args:
            outputs: Dictionary of model outputs
            transform_name: Name of the applied transformation
            original_size: (H, W) original image size
            
        Returns:
            reversed_outputs: Outputs aligned with original image
        """
        reversed_outputs = {}
        
        for key, tensor in outputs.items():
            if transform_name == "original":
                reversed_outputs[key] = tensor
            elif transform_name == "flip_h":
                reversed_outputs[key] = torch.flip(tensor, dims=[3])
            elif transform_name == "flip_v":
                reversed_outputs[key] = torch.flip(tensor, dims=[2])
            elif transform_name.startswith("rot90_"):
                k = int(transform_name.split("_")[1])
                reversed_outputs[key] = torch.rot90(tensor, -k, dims=[2, 3])
            elif transform_name.startswith("scale_"):
                reversed_outputs[key] = F.interpolate(
                    tensor, 
                    size=original_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                reversed_outputs[key] = tensor
        
        return reversed_outputs


class ErrorAwareInferencePipeline:
    """
    Complete inference pipeline for Error-Aware MAUNet
    
    Integrates model prediction, test-time augmentation, and seeded watershed
    post-processing for optimal instance segmentation performance.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        use_tta: bool = True,
        use_watershed: bool = True,
        watershed_config: Optional[Dict[str, Any]] = None,
        tta_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            model: Trained Error-Aware MAUNet model
            device: Device for computation
            use_tta: Whether to use test-time augmentation
            use_watershed: Whether to use seeded watershed post-processing
            watershed_config: Configuration for watershed processor
            tta_config: Configuration for test-time augmentation
        """
        self.model = model.to(device)
        self.device = device
        self.use_tta = use_tta
        self.use_watershed = use_watershed
        
        # Initialize processors
        if use_watershed:
            watershed_params = watershed_config or {}
            self.watershed_processor = SeededWatershedProcessor(**watershed_params)
        
        if use_tta:
            tta_params = tta_config or {}
            self.tta_processor = TestTimeAugmentation(**tta_params)
    
    def predict_single(self, image: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Predict on a single image without augmentation
        
        Args:
            image: (C, H, W) input image tensor
            
        Returns:
            predictions: Dictionary containing prediction arrays
        """
        self.model.eval()
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(self.device)
            outputs = self.model(image_batch)
            
            # Convert to numpy
            predictions = {}
            for key, tensor in outputs.items():
                predictions[key] = tensor.squeeze(0).cpu().numpy()
        
        return predictions
    
    def predict_with_tta(self, image: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Predict with test-time augmentation
        
        Args:
            image: (C, H, W) input image tensor
            
        Returns:
            averaged_predictions: Dictionary containing averaged predictions
        """
        original_size = image.shape[1:]  # (H, W)
        
        # Apply augmentations
        augmented_images = self.tta_processor.apply_augmentations(image)
        
        # Collect predictions
        all_predictions = []
        
        for aug_image, transform_name in augmented_images:
            # Predict on augmented image
            aug_predictions = self.predict_single(aug_image)
            
            # Convert to tensors for reversal
            aug_outputs_tensor = {}
            for key, array in aug_predictions.items():
                aug_outputs_tensor[key] = torch.from_numpy(array).unsqueeze(0)
            
            # Reverse augmentation
            reversed_outputs = self.tta_processor.reverse_augmentation(
                aug_outputs_tensor, transform_name, original_size
            )
            
            # Convert back to numpy
            reversed_predictions = {}
            for key, tensor in reversed_outputs.items():
                reversed_predictions[key] = tensor.squeeze(0).numpy()
            
            all_predictions.append(reversed_predictions)
        
        # Average predictions
        averaged_predictions = {}
        for key in all_predictions[0].keys():
            stacked = np.stack([pred[key] for pred in all_predictions], axis=0)
            averaged_predictions[key] = np.mean(stacked, axis=0)
        
        return averaged_predictions
    
    def predict(self, image: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Main prediction method
        
        Args:
            image: (C, H, W) input image tensor
            
        Returns:
            final_predictions: Dictionary containing final predictions and instance mask
        """
        # Get model predictions (with or without TTA)
        if self.use_tta:
            predictions = self.predict_with_tta(image)
        else:
            predictions = self.predict_single(image)
        
        final_predictions = predictions.copy()
        
        # Apply seeded watershed post-processing if enabled
        if self.use_watershed:
            segmentation_logits = predictions['segmentation']
            distance_transform = predictions['distance_transform'].squeeze()
            centroid_heatmap = predictions.get('centroid_heatmap')
            
            if centroid_heatmap is not None:
                centroid_heatmap = centroid_heatmap.squeeze()
            
            # Generate instance mask
            instance_mask = self.watershed_processor.process(
                segmentation_logits,
                distance_transform,
                centroid_heatmap
            )
            
            final_predictions['instance_mask'] = instance_mask
        else:
            # Fallback: simple thresholding and connected components
            seg_probs = np.exp(segmentation_logits) / np.sum(np.exp(segmentation_logits), axis=0, keepdims=True)
            if seg_probs.shape[0] > 1:
                foreground_prob = np.sum(seg_probs[1:], axis=0)
            else:
                foreground_prob = seg_probs[0]
            
            binary_mask = foreground_prob > 0.5
            instance_mask = measure.label(binary_mask, connectivity=2)
            final_predictions['instance_mask'] = instance_mask
        
        return final_predictions


def load_model_for_inference(
    checkpoint_path: str,
    model_config: Dict[str, Any],
    device: torch.device
) -> torch.nn.Module:
    """
    Load trained model from checkpoint for inference
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_config: Model configuration dictionary
        device: Device for computation
        
    Returns:
        loaded_model: Loaded model ready for inference
    """
    from error_aware_maunet import create_error_aware_maunet_model
    
    # Create model
    model = create_error_aware_maunet_model(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model


def create_ensemble_inference_pipeline(
    checkpoint_paths: List[str],
    model_configs: List[Dict[str, Any]],
    device: torch.device,
    **pipeline_kwargs
) -> ErrorAwareInferencePipeline:
    """
    Create inference pipeline with model ensemble
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        model_configs: List of model configuration dictionaries
        device: Device for computation
        **pipeline_kwargs: Additional arguments for inference pipeline
        
    Returns:
        ensemble_pipeline: Inference pipeline with ensemble model
    """
    # Load individual models
    models = []
    for checkpoint_path, model_config in zip(checkpoint_paths, model_configs):
        model = load_model_for_inference(checkpoint_path, model_config, device)
        models.append(model)
    
    # Create ensemble
    ensemble_model = ErrorAwareMAUNetEnsemble(models, average=True)
    
    # Create inference pipeline
    pipeline = ErrorAwareInferencePipeline(ensemble_model, device, **pipeline_kwargs)
    
    return pipeline
