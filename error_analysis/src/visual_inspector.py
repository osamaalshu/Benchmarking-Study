"""
Visual inspection and qualitative analysis system
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from scipy import ndimage
import logging
from skimage import exposure, filters
from skimage.util import img_as_float, img_as_ubyte
from skimage.transform import resize

try:
    from config.analysis_config import ANALYSIS_CONFIG, IMAGE_CONFIG, MODELS
except ImportError:
    from analysis_config import ANALYSIS_CONFIG, IMAGE_CONFIG, MODELS

class VisualInspector:
    """Creates comprehensive visual comparisons and qualitative analysis"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = ANALYSIS_CONFIG['visualization']
        
        # Set up color schemes
        self.class_colors = IMAGE_CONFIG['class_colors']
        self.error_colors = {
            'true_positive': (0, 255, 0),      # Green
            'false_positive': (255, 0, 0),     # Red  
            'false_negative': (0, 0, 255),     # Blue
            'true_negative': (128, 128, 128)   # Gray
        }
    
    def enhance_image_visibility(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image visibility for microscopy images with poor contrast
        Handles different modalities (bright field, fluorescence, phase contrast)
        """
        if len(image.shape) == 3:
            # RGB image - check if it's actually grayscale stored as RGB
            if np.allclose(image[:,:,0], image[:,:,1]) and np.allclose(image[:,:,1], image[:,:,2]):
                # Convert to grayscale
                img_gray = image[:,:,0].astype(np.float32)
            else:
                # True RGB - convert to grayscale using luminance
                img_gray = 0.299 * image[:,:,0].astype(np.float32) + \
                          0.587 * image[:,:,1].astype(np.float32) + \
                          0.114 * image[:,:,2].astype(np.float32)
        else:
            img_gray = image.astype(np.float32)
        
        # Normalize to [0, 1]
        if img_gray.max() > 1.0:
            img_gray = img_gray / 255.0
        
        # Analyze image characteristics
        mean_intensity = np.mean(img_gray)
        std_intensity = np.std(img_gray)
        
        # Apply different enhancement strategies based on image characteristics
        if mean_intensity < 0.3:
            # Dark image - likely fluorescence or dark field
            # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
            img_enhanced = exposure.equalize_adapthist(img_gray, clip_limit=0.03)
        elif mean_intensity > 0.7:
            # Bright image - likely bright field or phase contrast
            # Invert and enhance
            img_enhanced = 1.0 - img_gray
            img_enhanced = exposure.equalize_adapthist(img_enhanced, clip_limit=0.02)
        elif std_intensity < 0.1:
            # Low contrast image
            # Use gamma correction and histogram equalization
            img_enhanced = exposure.adjust_gamma(img_gray, gamma=0.7)
            img_enhanced = exposure.equalize_hist(img_enhanced)
        else:
            # Normal contrast - just enhance slightly
            img_enhanced = exposure.equalize_adapthist(img_gray, clip_limit=0.01)
        
        # Apply slight Gaussian smoothing to reduce noise
        img_enhanced = filters.gaussian(img_enhanced, sigma=0.5)
        
        # Ensure values are in [0, 1]
        img_enhanced = np.clip(img_enhanced, 0, 1)
        
        return img_enhanced
    
    def create_enhanced_display_image(self, image: np.ndarray, title: str = "") -> np.ndarray:
        """
        Create an enhanced version of the image for display
        Returns both original and enhanced versions
        """
        # Create enhanced version
        enhanced = self.enhance_image_visibility(image)
        
        # Create a side-by-side comparison
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image
            if np.allclose(image[:,:,0], image[:,:,1]) and np.allclose(image[:,:,1], image[:,:,2]):
                # Grayscale stored as RGB
                original_gray = image[:,:,0]
            else:
                # True RGB - convert to grayscale for comparison
                original_gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            original_gray = image
        
        # Normalize original for display
        if original_gray.max() > 1.0:
            original_gray = original_gray / 255.0
        
        return enhanced, original_gray

    def create_error_overlay(self, ground_truth, prediction, error_analysis):
        """Create color-coded error overlay showing FP, FN, TP, and boundary errors"""
        h, w = ground_truth.shape
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        
        # Create binary masks for error analysis
        gt_binary = (ground_truth > 0).astype(np.uint8)
        pred_binary = (prediction > 0).astype(np.uint8)
        
        # Create error masks based on pixel-level comparison
        fn_mask = (gt_binary == 1) & (pred_binary == 0)  # Ground truth has cell, prediction doesn't
        fp_mask = (gt_binary == 0) & (pred_binary == 1)  # Ground truth is background, prediction has cell
        tp_mask = (gt_binary == 1) & (pred_binary == 1)  # Both have cell
        
        # Color coding with transparency:
        # False Negatives (missed cells) - RED
        overlay[fn_mask, 0] = 1.0  # Red channel
        
        # False Positives (fake detections) - BLUE  
        overlay[fp_mask, 2] = 1.0  # Blue channel
        
        # True Positives (correct detections) - GREEN
        overlay[tp_mask, 1] = 0.7  # Green channel (slightly dimmer)
        
        # Add boundary errors - find edge pixels where GT and prediction differ
        from scipy import ndimage
        gt_edges = ndimage.binary_dilation(gt_binary) & ~gt_binary
        pred_edges = ndimage.binary_dilation(pred_binary) & ~pred_binary
        
        # Boundary errors where edges don't align
        boundary_error_mask = (gt_edges | pred_edges) & (gt_binary != pred_binary)
        # Boundary errors - YELLOW (Red + Green)
        overlay[boundary_error_mask, 0] = 0.8  # Red
        overlay[boundary_error_mask, 1] = 0.8  # Green
        
        return overlay

    def create_side_by_side_comparison(self, original_image, ground_truth, predictions, error_results, image_name, save_path=None):
        """Create enhanced side-by-side comparison with error overlays"""
        n_models = len(predictions)
        # Show: Original (Enhanced), Original (Raw), GT, and for each model: Prediction + Error Overlay
        n_cols = 5  # Enhanced Original, Raw Original, GT, Model Pred, Error Overlay
        n_rows = max(1, n_models)  # One row per model
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Get enhanced and raw versions of original image
        enhanced_original, raw_original = self.create_enhanced_display_image(original_image)
        
        # Handle image display with proper contrast
        def display_image(ax, img, title, cmap=None):
            if len(img.shape) == 3 and img.shape[2] == 3:
                # RGB image - check if grayscale stored as RGB
                if np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2]):
                    # Enhance contrast for grayscale
                    img_display = img[:,:,0]
                    img_min, img_max = np.percentile(img_display, [2, 98])
                    if img_max > img_min:
                        img_display = np.clip((img_display - img_min) / (img_max - img_min), 0, 1)
                    ax.imshow(img_display, cmap='gray')
                else:
                    ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontweight='bold', fontsize=10)
            ax.axis('off')
        
        # First row: Original (Enhanced), Original (Raw), and GT (same for all models)
        for row in range(n_rows):
            # Column 0: Enhanced Original image
            display_image(axes[row, 0], enhanced_original, 'Original Image (Enhanced)')
            
            # Column 1: Raw Original image
            display_image(axes[row, 1], raw_original, 'Original Image (Raw)')
            
            # Column 2: Ground truth
            # Get gt_cells from the first available model's error results
            first_model = list(error_results.keys())[0]
            gt_cells = error_results[first_model]['summary']['gt_num_cells']
            display_image(axes[row, 2], ground_truth, f'Ground Truth\n{gt_cells} cells', cmap='viridis')
        
        # Model predictions and error overlays
        for row, (model_name, prediction) in enumerate(predictions.items()):
            if row >= n_rows:
                break
                
            error_analysis = error_results[model_name]['error_analysis']
            
            # Column 3: Model prediction
            pred_cells = error_analysis['summary']['pred_num_cells']
            fn_count = error_analysis['false_negatives']['count']
            fp_count = error_analysis['false_positives']['count']
            
            pred_title = f"{model_name.upper()}\n{pred_cells} cells"
            display_image(axes[row, 3], prediction, pred_title, cmap='viridis')
            
            # Column 4: Error overlay
            error_overlay = self.create_error_overlay(ground_truth, prediction, error_analysis)
            
            # Create better background for overlay visibility
            if len(original_image.shape) == 3:
                if np.allclose(original_image[:,:,0], original_image[:,:,1]):
                    # Grayscale stored as RGB - normalize properly
                    base_img = original_image[:,:,0].astype(np.float32)
                    if base_img.max() > 1.0:
                        base_img = base_img / 255.0  # Normalize to [0,1]
                    
                    # Enhance contrast
                    img_min, img_max = np.percentile(base_img, [2, 98])
                    if img_max > img_min:
                        base_img = np.clip((base_img - img_min) / (img_max - img_min), 0, 1)
                    
                    # Check if image is mostly bright (inverted microscopy)
                    mean_intensity = np.mean(base_img)
                    if mean_intensity > 0.7:  # Mostly bright image
                        # Use darker background and invert if needed
                        base_img = 1.0 - base_img  # Invert
                        base_img = np.stack([base_img, base_img, base_img], axis=2) * 0.3
                    else:
                        # Normal dark background
                        base_img = np.stack([base_img, base_img, base_img], axis=2) * 0.3
                else:
                    # True RGB image
                    base_img = original_image.astype(np.float32)
                    if base_img.max() > 1.0:
                        base_img = base_img / 255.0
                    base_img = base_img * 0.3
            else:
                # Grayscale image
                base_img = original_image.astype(np.float32)
                if base_img.max() > 1.0:
                    base_img = base_img / 255.0
                
                # Check brightness and handle inverted microscopy
                mean_intensity = np.mean(base_img)
                if mean_intensity > 0.7:
                    base_img = 1.0 - base_img  # Invert bright images
                
                base_img = np.stack([base_img, base_img, base_img], axis=2) * 0.3
            
            # Blend with high contrast overlay
            # Use additive blending for bright overlays on dark background
            blended = np.clip(base_img + error_overlay * 0.9, 0, 1)
            
            # Ensure overlay is visible by adding a slight dark outline
            overlay_mask = np.sum(error_overlay, axis=2) > 0
            if np.sum(overlay_mask) > 0:
                # Add subtle dark border around overlay regions for visibility
                from scipy import ndimage
                dilated_mask = ndimage.binary_dilation(overlay_mask, iterations=1)
                border_mask = dilated_mask & ~overlay_mask
                blended[border_mask] = blended[border_mask] * 0.7  # Darken border
            
            error_title = f"Errors: FN={fn_count}, FP={fp_count}\nRED=FN, BLUE=FP, GREEN=TP"
            axes[row, 4].imshow(blended)
            axes[row, 4].set_title(error_title, fontweight='bold', fontsize=10)
            axes[row, 4].axis('off')
        
        plt.suptitle(f'Enhanced Model Comparison with Error Analysis - {image_name}', fontsize=16, fontweight='bold')
        
        # Add proper color legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='False Negatives (missed cells)'),
            Patch(facecolor='blue', label='False Positives (fake detections)'),
            Patch(facecolor='green', label='True Positives (correct detections)'),
            Patch(facecolor='yellow', label='Boundary Errors')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            self.logger.info(f"Enhanced comparison saved to {save_path}")
        
        return fig
    
    def create_error_overlay_comparison(self, image: np.ndarray,
                                      ground_truth: np.ndarray,
                                      predictions: Dict[str, np.ndarray],
                                      image_name: str,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Create error overlay visualizations for all models"""
        n_models = len(predictions)
        n_cols = min(3, n_models)
        n_rows = int(np.ceil(n_models / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        for model_name, prediction in predictions.items():
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            
            # Create error overlay
            error_overlay = self._create_error_overlay(image, ground_truth, prediction)
            
            axes[row, col].imshow(error_overlay)
            axes[row, col].set_title(f'{MODELS[model_name]["name"]} - Error Analysis', 
                                   fontsize=12, fontweight='bold')
            axes[row, col].axis('off')
            
            plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        # Add legend
        legend_elements = [
            patches.Patch(color=np.array(self.error_colors['true_positive'])/255, label='True Positive'),
            patches.Patch(color=np.array(self.error_colors['false_positive'])/255, label='False Positive'),
            patches.Patch(color=np.array(self.error_colors['false_negative'])/255, label='False Negative')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.suptitle(f'Error Analysis - {image_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            self.logger.info(f"Error overlay comparison saved to {save_path}")
        
        return fig
    
    def create_detailed_case_study(self, image: np.ndarray,
                                 ground_truth: np.ndarray,
                                 predictions: Dict[str, np.ndarray],
                                 error_analyses: Dict[str, Dict],
                                 image_name: str,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Create detailed case study with error statistics"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.3)
        
        # Original image (top left)
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(image)
        ax_orig.set_title('Original Image', fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        # Ground truth (top middle)
        ax_gt = fig.add_subplot(gs[0, 1])
        gt_colored = self._colorize_segmentation(ground_truth)
        ax_gt.imshow(gt_colored)
        ax_gt.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax_gt.axis('off')
        
        # Error statistics table (top right)
        ax_stats = fig.add_subplot(gs[0, 2:])
        self._create_error_statistics_table(ax_stats, error_analyses)
        
        # Model predictions and error overlays (bottom rows)
        model_names = list(predictions.keys())
        for idx, model_name in enumerate(model_names[:4]):  # Show top 4 models
            row = 1 + idx // 2
            col = (idx % 2) * 3
            
            # Prediction
            ax_pred = fig.add_subplot(gs[row, col])
            pred_colored = self._colorize_segmentation(predictions[model_name])
            ax_pred.imshow(pred_colored)
            ax_pred.set_title(f'{MODELS[model_name]["name"]} Prediction', 
                            fontsize=10, fontweight='bold')
            ax_pred.axis('off')
            
            # Error overlay
            ax_error = fig.add_subplot(gs[row, col+1])
            error_overlay = self._create_error_overlay(image, ground_truth, predictions[model_name])
            ax_error.imshow(error_overlay)
            ax_error.set_title(f'{MODELS[model_name]["name"]} Errors', 
                             fontsize=10, fontweight='bold')
            ax_error.axis('off')
            
            # Error breakdown
            ax_breakdown = fig.add_subplot(gs[row, col+2])
            self._create_error_breakdown_chart(ax_breakdown, error_analyses[model_name], model_name)
        
        plt.suptitle(f'Detailed Case Study - {image_name}', fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            self.logger.info(f"Detailed case study saved to {save_path}")
        
        return fig
    
    def create_modality_comparison(self, images_by_modality: Dict[int, List],
                                 modality_names: Dict[int, str],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Create visual comparison of different modalities"""
        n_modalities = len(images_by_modality)
        n_samples_per_modality = 3  # Show 3 examples per modality
        
        fig, axes = plt.subplots(n_modalities, n_samples_per_modality, 
                               figsize=(n_samples_per_modality * 4, n_modalities * 4))
        
        if n_modalities == 1:
            axes = axes.reshape(1, -1)
        
        for mod_idx, (modality_id, image_list) in enumerate(images_by_modality.items()):
            modality_name = modality_names.get(modality_id, f"Modality {modality_id}")
            
            # Select representative samples
            samples = image_list[:n_samples_per_modality]
            
            for sample_idx, (image_name, image_data) in enumerate(samples):
                axes[mod_idx, sample_idx].imshow(image_data['image'])
                axes[mod_idx, sample_idx].set_title(f'{modality_name}\n{image_name}', 
                                                  fontsize=10, fontweight='bold')
                axes[mod_idx, sample_idx].axis('off')
                
                # Add metadata text
                metadata = image_data.get('metadata', {})
                info_text = f"Cells: {metadata.get('num_cells', 'N/A')}\n"
                info_text += f"Avg Area: {metadata.get('mean_cell_area', 0):.0f}"
                axes[mod_idx, sample_idx].text(0.02, 0.98, info_text,
                                             transform=axes[mod_idx, sample_idx].transAxes,
                                             verticalalignment='top',
                                             bbox=dict(boxstyle="round,pad=0.3", 
                                                     facecolor="white", alpha=0.8),
                                             fontsize=8)
        
        plt.suptitle('Modality Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            self.logger.info(f"Modality comparison saved to {save_path}")
        
        return fig
    
    def create_compact_comparison(self, original_image, ground_truth, predictions, error_results, image_name, save_path=None):
        """Create compact comparison with spacing, clean legend, horizontal headers"""
        n_models = len(predictions)
        n_cols = 4  # Original, Ground Truth, Prediction, Error Analysis
        n_rows = n_models  # One row per model
        
        # Create figure with spacing between subplots and extra space for legend bar
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows + 1))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Add spacing between subplots
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15, wspace=0.05, hspace=0.1)
        
        # Get target size from ground truth
        target_height, target_width = ground_truth.shape[:2]
        
        # Resize original image to match ground truth size
        if original_image.shape[:2] != (target_height, target_width):
            if len(original_image.shape) == 3:
                original_image = resize(original_image, (target_height, target_width, original_image.shape[2]), 
                                      preserve_range=True, anti_aliasing=True)
            else:
                original_image = resize(original_image, (target_height, target_width), 
                                      preserve_range=True, anti_aliasing=True)
        
        # Handle image display with proper contrast
        def display_image(ax, img, cmap=None):
            if len(img.shape) == 3 and img.shape[2] == 3:
                # RGB image - check if grayscale stored as RGB
                if np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2]):
                    # Enhance contrast for grayscale
                    img_display = img[:,:,0]
                    img_min, img_max = np.percentile(img_display, [2, 98])
                    if img_max > img_min:
                        img_display = np.clip((img_display - img_min) / (img_max - img_min), 0, 1)
                    ax.imshow(img_display, cmap='gray')
                else:
                    ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap)
            ax.axis('off')
        
        # Add column headers closer to the images (horizontal)
        column_titles = ['Original Image', 'Ground Truth', 'Prediction', 'Error Analysis']
        for col, title in enumerate(column_titles):
            # Add horizontal text for column headers closer to images
            fig.text(0.125 + col * 0.225, 0.93, title, 
                    fontsize=16, fontweight='bold',
                    horizontalalignment='center', verticalalignment='center')
        
        # Process each model
        for row, (model_name, prediction) in enumerate(predictions.items()):
            error_analysis = error_results[model_name]['error_analysis']
            
            # Column 0: Original Image (same for all rows)
            display_image(axes[row, 0], original_image)
            
            # Column 1: Ground Truth (same for all rows)
            display_image(axes[row, 1], ground_truth, cmap='viridis')
            
            # Column 2: Model prediction
            display_image(axes[row, 2], prediction, cmap='viridis')
            
            # Column 3: Error overlay
            error_overlay = self.create_error_overlay(ground_truth, prediction, error_analysis)
            
            # Create better background for overlay visibility
            if len(original_image.shape) == 3:
                if np.allclose(original_image[:,:,0], original_image[:,:,1]):
                    # Grayscale stored as RGB - normalize properly
                    base_img = original_image[:,:,0].astype(np.float32)
                    if base_img.max() > 1.0:
                        base_img = base_img / 255.0  # Normalize to [0,1]
                    
                    # Enhance contrast
                    img_min, img_max = np.percentile(base_img, [2, 98])
                    if img_max > img_min:
                        base_img = np.clip((base_img - img_min) / (img_max - img_min), 0, 1)
                    
                    # Check if image is mostly bright (inverted microscopy)
                    mean_intensity = np.mean(base_img)
                    if mean_intensity > 0.7:  # Mostly bright image
                        # Use darker background and invert if needed
                        base_img = 1.0 - base_img  # Invert
                        base_img = np.stack([base_img, base_img, base_img], axis=2) * 0.3
                    else:
                        # Normal dark background
                        base_img = np.stack([base_img, base_img, base_img], axis=2) * 0.3
                else:
                    # True RGB image
                    base_img = original_image.astype(np.float32)
                    if base_img.max() > 1.0:
                        base_img = base_img / 255.0
                    base_img = base_img * 0.3
            else:
                # Grayscale image
                base_img = original_image.astype(np.float32)
                if base_img.max() > 1.0:
                    base_img = base_img / 255.0
                
                # Check brightness and handle inverted microscopy
                mean_intensity = np.mean(base_img)
                if mean_intensity > 0.7:
                    base_img = 1.0 - base_img  # Invert bright images
                
                base_img = np.stack([base_img, base_img, base_img], axis=2) * 0.3
            
            # Blend with high contrast overlay
            blended = np.clip(base_img + error_overlay * 0.9, 0, 1)
            
            # Ensure overlay is visible by adding a slight dark outline
            overlay_mask = np.sum(error_overlay, axis=2) > 0
            if np.sum(overlay_mask) > 0:
                # Add subtle dark border around overlay regions for visibility
                from scipy import ndimage
                dilated_mask = ndimage.binary_dilation(overlay_mask, iterations=1)
                border_mask = dilated_mask & ~overlay_mask
                blended[border_mask] = blended[border_mask] * 0.7  # Darken border
            
            axes[row, 3].imshow(blended)
            axes[row, 3].axis('off')
            
            # Add model name on the left side of the row (vertical text) - bigger font
            axes[row, 0].text(-0.15, 0.5, model_name.upper(), 
                            transform=axes[row, 0].transAxes, 
                            fontsize=16, fontweight='bold',
                            rotation=90,
                            verticalalignment='center',
                            horizontalalignment='center')
        
        # Create compact legend with colored circles at the bottom
        legend_y = 0.08
        legend_height = 0.06
        
        # Create legend with circles and short labels
        legend_colors = ['red', 'blue', 'green']
        legend_labels = ['FN', 'FP', 'TP']  # Short labels
        
        for i, (color, label) in enumerate(zip(legend_colors, legend_labels)):
            x_pos = 0.15 + i * 0.15  # Closer spacing
            # Create colored circle
            from matplotlib.patches import Circle
            circle = Circle((x_pos + 0.01, legend_y + legend_height/2), 0.012, 
                          facecolor=color, edgecolor='black', linewidth=1,
                          transform=fig.transFigure)
            fig.add_artist(circle)
            # Add label text (bigger font, closer to circle)
            fig.text(x_pos + 0.03, legend_y + legend_height/2, label, 
                    fontsize=14, fontweight='bold',
                    horizontalalignment='left', verticalalignment='center')
        
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            self.logger.info(f"Compact comparison saved to {save_path}")
        
        return fig
    
    def _colorize_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Convert segmentation to colored visualization"""
        colored = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(self.class_colors):
            mask = (segmentation == class_id)
            colored[mask] = color
        
        return colored
    
    def _create_error_overlay(self, image: np.ndarray, ground_truth: np.ndarray, 
                            prediction: np.ndarray) -> np.ndarray:
        """Create error overlay on original image"""
        # Convert to binary for error analysis
        gt_binary = (ground_truth > 0).astype(int)
        pred_binary = (prediction > 0).astype(int)
        
        # Create error map
        tp = (gt_binary == 1) & (pred_binary == 1)  # True Positive
        fp = (gt_binary == 0) & (pred_binary == 1)  # False Positive  
        fn = (gt_binary == 1) & (pred_binary == 0)  # False Negative
        
        # Start with original image
        overlay = image.copy().astype(float) / 255.0
        
        # Apply error colors with transparency
        alpha = self.config['alpha_overlay']
        
        # True positives (green)
        overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 1, 0]) * alpha
        
        # False positives (red)
        overlay[fp] = overlay[fp] * (1 - alpha) + np.array([1, 0, 0]) * alpha
        
        # False negatives (blue)
        overlay[fn] = overlay[fn] * (1 - alpha) + np.array([0, 0, 1]) * alpha
        
        return (overlay * 255).astype(np.uint8)
    
    def _create_error_statistics_table(self, ax, error_analyses: Dict[str, Dict]):
        """Create table showing error statistics for all models"""
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table
        table_data = []
        headers = ['Model', 'False Neg.', 'False Pos.', 'Under-seg.', 'Over-seg.', 'Boundary F1']
        
        for model_name, analysis in error_analyses.items():
            row = [
                MODELS[model_name]['name'],
                f"{analysis['false_negatives']['count']}",
                f"{analysis['false_positives']['count']}",
                f"{analysis['under_segmentation']['count']}",
                f"{analysis['over_segmentation']['count']}",
                f"{analysis['boundary_errors']['boundary_f1']:.3f}"
            ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Error Statistics Summary', fontsize=12, fontweight='bold', pad=20)
    
    def _create_error_breakdown_chart(self, ax, error_analysis: Dict, model_name: str):
        """Create pie chart showing error breakdown"""
        # Extract error counts
        fn_count = error_analysis['false_negatives']['count']
        fp_count = error_analysis['false_positives']['count']
        under_seg = error_analysis['under_segmentation']['count']
        over_seg = error_analysis['over_segmentation']['count']
        
        # Prepare data for pie chart
        labels = []
        sizes = []
        colors = []
        
        if fn_count > 0:
            labels.append(f'False Neg.\n({fn_count})')
            sizes.append(fn_count)
            colors.append('#FF6B6B')
        
        if fp_count > 0:
            labels.append(f'False Pos.\n({fp_count})')
            sizes.append(fp_count)
            colors.append('#4ECDC4')
        
        if under_seg > 0:
            labels.append(f'Under-seg.\n({under_seg})')
            sizes.append(under_seg)
            colors.append('#45B7D1')
        
        if over_seg > 0:
            labels.append(f'Over-seg.\n({over_seg})')
            sizes.append(over_seg)
            colors.append('#FFA07A')
        
        if not sizes:
            # No errors - show success
            labels = ['Perfect\nSegmentation']
            sizes = [1]
            colors = ['#90EE90']
        
        # Create pie chart
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 8})
        ax.set_title(f'{MODELS[model_name]["name"]}\nError Breakdown', 
                    fontsize=10, fontweight='bold')
    
    def create_performance_heatmap(self, performance_by_modality: Dict,
                                 modality_names: Dict[int, str],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Create heatmap showing model performance across modalities"""
        # Prepare data for heatmap
        models = list(MODELS.keys())
        modalities = list(modality_names.keys())
        
        # Create matrix
        performance_matrix = np.zeros((len(models), len(modalities)))
        
        for i, model in enumerate(models):
            for j, modality in enumerate(modalities):
                if model in performance_by_modality and modality in performance_by_modality[model]:
                    performance_matrix[i, j] = performance_by_modality[model][modality]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(modalities)))
        ax.set_yticks(range(len(models)))
        ax.set_xticklabels([modality_names[mod] for mod in modalities], rotation=45, ha='right')
        ax.set_yticklabels([MODELS[model]['name'] for model in models])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('F1 Score', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(modalities)):
                text = ax.text(j, i, f'{performance_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Model Performance by Modality', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            self.logger.info(f"Performance heatmap saved to {save_path}")
        
        return fig
