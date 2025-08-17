"""
Data Arms Manager for Modality Agnostic Controlled Augmentation Study

Manages different dataset arms:
- R (Real-only): Original training set
- RxS@{10,25,50}: Replace 10%/25%/50% of training images with synthetic
- S (Synthetic-only): Synthetic pairs equal in size to R
- Rmask+SynthTex@25: 25% real masks with synthetic textures
"""

import os
import shutil
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
import logging
from PIL import Image
from sklearn.model_selection import train_test_split

from .cascaded_diffusion_wrapper import CascadedDiffusionWrapper


class DataArmsManager:
    """Manages creation and organization of different dataset arms for the augmentation study"""
    
    def __init__(self, 
                 base_data_dir: str,
                 output_dir: str,
                 cascaded_model: Optional[CascadedDiffusionWrapper] = None,
                 seed: int = 42):
        """
        Initialize the data arms manager
        
        Args:
            base_data_dir: Path to original training data directory
            output_dir: Output directory for all arms
            cascaded_model: Initialized cascaded diffusion model
            seed: Random seed for reproducibility
        """
        self.base_data_dir = Path(base_data_dir)
        self.output_dir = Path(output_dir)
        self.cascaded_model = cascaded_model
        self.seed = seed
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load original dataset info
        self.original_data = self._load_original_dataset()
        
        # Store arm configurations
        self.arm_configs = {
            'R': {'type': 'real_only', 'ratio': 1.0},
            'RxS@10': {'type': 'mixed', 'synthetic_ratio': 0.1},
            'RxS@25': {'type': 'mixed', 'synthetic_ratio': 0.25},
            'RxS@50': {'type': 'mixed', 'synthetic_ratio': 0.5},
            'S': {'type': 'synthetic_only', 'ratio': 1.0},
            'Rmask+SynthTex@25': {'type': 'real_mask_synth_texture', 'ratio': 0.25}
        }
        
    def _load_original_dataset(self) -> Dict:
        """Load information about the original dataset"""
        images_dir = self.base_data_dir / 'images'
        labels_dir = self.base_data_dir / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            raise ValueError(f"Expected images and labels directories in {self.base_data_dir}")
        
        image_files = sorted(list(images_dir.glob('*')))
        label_files = sorted(list(labels_dir.glob('*')))
        
        # Match image and label files
        matched_pairs = []
        for img_file in image_files:
            # Find corresponding label file
            label_file = None
            for lbl_file in label_files:
                # Handle different naming patterns: 
                # - cell_00994.png -> cell_00994_label.png
                # - cell_00994.png -> cell_00994_label.tiff
                # - cell_00994.tif -> cell_00994_label.png
                # - cell_00994.tiff -> cell_00994_label.tiff
                base_name = img_file.stem
                if lbl_file.stem == base_name or lbl_file.stem == f"{base_name}_label":
                    label_file = lbl_file
                    break
            
            if label_file:
                matched_pairs.append((img_file, label_file))
            else:
                self.logger.warning(f"No matching label found for {img_file}")
        
        self.logger.info(f"Found {len(matched_pairs)} matched image-label pairs")
        
        return {
            'pairs': matched_pairs,
            'count': len(matched_pairs),
            'images_dir': images_dir,
            'labels_dir': labels_dir
        }
    
    def create_real_only_arm(self, arm_name: str = 'R') -> str:
        """
        Create Real-only (R) arm - just copy original training data
        
        Args:
            arm_name: Name of the arm
            
        Returns:
            Path to created arm directory
        """
        arm_dir = self.output_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = arm_dir / 'images'
        labels_dir = arm_dir / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # Copy all original data
        for img_path, lbl_path in self.original_data['pairs']:
            shutil.copy2(img_path, images_dir / img_path.name)
            shutil.copy2(lbl_path, labels_dir / lbl_path.name)
        
        # Save arm info
        arm_info = {
            'arm_type': 'real_only',
            'total_samples': len(self.original_data['pairs']),
            'real_samples': len(self.original_data['pairs']),
            'synthetic_samples': 0,
            'seed': self.seed
        }
        
        with open(arm_dir / 'arm_info.json', 'w') as f:
            json.dump(arm_info, f, indent=2)
        
        self.logger.info(f"Created {arm_name} arm with {len(self.original_data['pairs'])} real samples")
        return str(arm_dir)
    
    def create_mixed_arm(self, synthetic_ratio: float, arm_name: str) -> str:
        """
        Create mixed Real+Synthetic (RxS@r) arm
        
        Args:
            synthetic_ratio: Ratio of synthetic samples (0.1 for 10%, etc.)
            arm_name: Name of the arm
            
        Returns:
            Path to created arm directory
        """
        if self.cascaded_model is None:
            raise ValueError("Cascaded diffusion model required for synthetic generation")
        
        arm_dir = self.output_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = arm_dir / 'images'
        labels_dir = arm_dir / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        total_samples = len(self.original_data['pairs'])
        synthetic_count = int(total_samples * synthetic_ratio)
        real_count = total_samples - synthetic_count
        
        # Randomly select which real samples to keep
        real_indices = random.sample(range(total_samples), real_count)
        real_pairs = [self.original_data['pairs'][i] for i in real_indices]
        
        # Copy selected real samples
        for i, (img_path, lbl_path) in enumerate(real_pairs):
            shutil.copy2(img_path, images_dir / f"real_{i:05d}.png")
            shutil.copy2(lbl_path, labels_dir / f"real_{i:05d}.png")
        
        # Generate synthetic samples
        self.logger.info(f"Generating {synthetic_count} synthetic samples for {arm_name}")
        
        synthetic_pairs = self.cascaded_model.generate_paired_synthetic_data(
            num_pairs=synthetic_count,
            image_size=(256, 256)
        )
        
        # Save synthetic samples
        for i, (synth_img, synth_mask) in enumerate(synthetic_pairs):
            # Save image
            if len(synth_img.shape) == 3:
                img_pil = Image.fromarray(synth_img.astype(np.uint8))
            else:
                img_pil = Image.fromarray(synth_img.astype(np.uint8), mode='L')
            img_pil.save(images_dir / f"synth_{i:05d}.png")
            
            # Save mask
            mask_pil = Image.fromarray((synth_mask * 255).astype(np.uint8), mode='L')
            mask_pil.save(labels_dir / f"synth_{i:05d}.png")
        
        # Save arm info
        arm_info = {
            'arm_type': 'mixed',
            'synthetic_ratio': synthetic_ratio,
            'total_samples': total_samples,
            'real_samples': real_count,
            'synthetic_samples': len(synthetic_pairs),
            'real_indices': real_indices,
            'seed': self.seed
        }
        
        with open(arm_dir / 'arm_info.json', 'w') as f:
            json.dump(arm_info, f, indent=2)
        
        self.logger.info(f"Created {arm_name} arm with {real_count} real + {len(synthetic_pairs)} synthetic samples")
        return str(arm_dir)
    
    def create_synthetic_only_arm(self, arm_name: str = 'S') -> str:
        """
        Create Synthetic-only (S) arm
        
        Args:
            arm_name: Name of the arm
            
        Returns:
            Path to created arm directory
        """
        if self.cascaded_model is None:
            raise ValueError("Cascaded diffusion model required for synthetic generation")
        
        arm_dir = self.output_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = arm_dir / 'images'
        labels_dir = arm_dir / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # Generate synthetic samples equal to original dataset size
        total_samples = len(self.original_data['pairs'])
        
        self.logger.info(f"Generating {total_samples} synthetic samples for {arm_name}")
        
        synthetic_pairs = self.cascaded_model.generate_paired_synthetic_data(
            num_pairs=total_samples,
            image_size=(256, 256)
        )
        
        # Save synthetic samples
        for i, (synth_img, synth_mask) in enumerate(synthetic_pairs):
            # Save image
            if len(synth_img.shape) == 3:
                img_pil = Image.fromarray(synth_img.astype(np.uint8))
            else:
                img_pil = Image.fromarray(synth_img.astype(np.uint8), mode='L')
            img_pil.save(images_dir / f"synth_{i:05d}.png")
            
            # Save mask
            mask_pil = Image.fromarray((synth_mask * 255).astype(np.uint8), mode='L')
            mask_pil.save(labels_dir / f"synth_{i:05d}.png")
        
        # Save arm info
        arm_info = {
            'arm_type': 'synthetic_only',
            'total_samples': len(synthetic_pairs),
            'real_samples': 0,
            'synthetic_samples': len(synthetic_pairs),
            'seed': self.seed
        }
        
        with open(arm_dir / 'arm_info.json', 'w') as f:
            json.dump(arm_info, f, indent=2)
        
        self.logger.info(f"Created {arm_name} arm with {len(synthetic_pairs)} synthetic samples")
        return str(arm_dir)
    
    def create_real_mask_synth_texture_arm(self, ratio: float = 0.25, arm_name: str = 'Rmask+SynthTex@25') -> str:
        """
        Create Real mask + Synthetic texture arm
        
        Args:
            ratio: Ratio of samples to use synthetic textures for
            arm_name: Name of the arm
            
        Returns:
            Path to created arm directory
        """
        if self.cascaded_model is None:
            raise ValueError("Cascaded diffusion model required for synthetic texture generation")
        
        arm_dir = self.output_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = arm_dir / 'images'
        labels_dir = arm_dir / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        total_samples = len(self.original_data['pairs'])
        synth_texture_count = int(total_samples * ratio)
        
        # Randomly select which samples get synthetic textures
        synth_texture_indices = set(random.sample(range(total_samples), synth_texture_count))
        
        synth_texture_generated = 0
        
        for i, (img_path, lbl_path) in enumerate(self.original_data['pairs']):
            if i in synth_texture_indices:
                # Generate synthetic texture for this real mask
                # Load real mask
                mask_img = Image.open(lbl_path).convert('L')
                mask_array = np.array(mask_img) > 128  # Binary mask
                
                # Generate synthetic texture
                synth_texture = self.cascaded_model.generate_texture_for_real_mask(
                    mask=mask_array.astype(np.uint8)
                )
                
                if synth_texture is not None:
                    # Save synthetic texture
                    if len(synth_texture.shape) == 3:
                        img_pil = Image.fromarray(synth_texture.astype(np.uint8))
                    else:
                        img_pil = Image.fromarray(synth_texture.astype(np.uint8), mode='L')
                    img_pil.save(images_dir / f"cell_{i:05d}.png")
                    synth_texture_generated += 1
                else:
                    # Fallback to original image if generation failed
                    shutil.copy2(img_path, images_dir / f"cell_{i:05d}.png")
                
                # Copy real mask with consistent naming
                shutil.copy2(lbl_path, labels_dir / f"cell_{i:05d}_label.png")
            else:
                # Use original real image and mask
                shutil.copy2(img_path, images_dir / f"cell_{i:05d}.png")
                shutil.copy2(lbl_path, labels_dir / f"cell_{i:05d}_label.png")
        
        # Save arm info
        arm_info = {
            'arm_type': 'real_mask_synth_texture',
            'synth_texture_ratio': ratio,
            'total_samples': total_samples,
            'real_samples': total_samples - synth_texture_generated,
            'synth_texture_samples': synth_texture_generated,
            'synth_texture_indices': list(synth_texture_indices),
            'seed': self.seed
        }
        
        with open(arm_dir / 'arm_info.json', 'w') as f:
            json.dump(arm_info, f, indent=2)
        
        self.logger.info(f"Created {arm_name} arm with {synth_texture_generated} synthetic textures")
        return str(arm_dir)
    
    def create_all_arms(self) -> Dict[str, str]:
        """
        Create all dataset arms for the augmentation study
        
        Returns:
            Dictionary mapping arm names to their directory paths
        """
        arm_paths = {}
        
        # Create R (Real-only) arm
        arm_paths['R'] = self.create_real_only_arm('R')
        
        # Create mixed arms (RxS@10, RxS@25, RxS@50)
        for ratio, arm_name in [(0.1, 'RxS@10'), (0.25, 'RxS@25'), (0.5, 'RxS@50')]:
            arm_paths[arm_name] = self.create_mixed_arm(ratio, arm_name)
        
        # Create S (Synthetic-only) arm
        arm_paths['S'] = self.create_synthetic_only_arm('S')
        
        # Create Rmask+SynthTex@25 arm
        arm_paths['Rmask+SynthTex@25'] = self.create_real_mask_synth_texture_arm(0.25, 'Rmask+SynthTex@25')
        
        # Save overall study info
        study_info = {
            'arms': arm_paths,
            'original_dataset_size': len(self.original_data['pairs']),
            'seed': self.seed,
            'arm_configs': self.arm_configs
        }
        
        with open(self.output_dir / 'study_info.json', 'w') as f:
            json.dump(study_info, f, indent=2)
        
        self.logger.info(f"Created all {len(arm_paths)} dataset arms")
        return arm_paths
    
    def get_arm_statistics(self) -> Dict:
        """Get statistics for all created arms"""
        stats = {}
        
        for arm_dir in self.output_dir.iterdir():
            if arm_dir.is_dir() and (arm_dir / 'arm_info.json').exists():
                with open(arm_dir / 'arm_info.json', 'r') as f:
                    arm_info = json.load(f)
                stats[arm_dir.name] = arm_info
        
        return stats


def main():
    """Test function for data arms manager"""
    # This would be called with actual paths and models
    print("Data Arms Manager module loaded successfully")
    

if __name__ == "__main__":
    main()
