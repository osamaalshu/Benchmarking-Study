"""
Unified Training Protocol for Modality Agnostic Controlled Augmentation Study

Fixed training protocol across all arms:
- Same models: nnUNet (best baseline) + U-Net (simpler model)
- Same optimizer, LR, augmentation, patch size
- Same number of optimizer steps (adjust epochs if dataset size differs)
- Seeds: {0, 1, 2} per arm
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging
import time
from datetime import datetime
import shutil

# Import model wrappers
from .model_wrappers import get_model_creator, CellDataset


class AugmentationStudyTrainer:
    """Unified trainer for all arms in the augmentation study"""
    
    def __init__(self, 
                 model_name: str = 'nnunet',
                 batch_size: int = 4,
                 initial_lr: float = 6e-4,
                 max_epochs: int = 30,
                 input_size: int = 256,
                 num_classes: int = 3,
                 device: str = 'auto',
                 seed: int = 42):
        """
        Initialize the trainer with fixed hyperparameters
        
        Args:
            model_name: Model to use ('nnunet' or 'unet')
            batch_size: Batch size per GPU
            initial_lr: Initial learning rate
            max_epochs: Maximum training epochs
            input_size: Input image size
            num_classes: Number of classes
            device: Device to use
            seed: Random seed
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.input_size = input_size
        self.num_classes = num_classes
        self.seed = seed
        
        # Device setup
        self.device = self._get_device(device)
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Training configuration
        self.config = {
            'model_name': model_name,
            'batch_size': batch_size,
            'initial_lr': initial_lr,
            'max_epochs': max_epochs,
            'input_size': input_size,
            'num_classes': num_classes,
            'seed': seed,
            'device': str(self.device),
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'loss_function': 'CrossEntropyLoss',
            'early_stopping_patience': 10,
            'val_interval': 2
        }
        
    def _get_device(self, device: str) -> torch.device:
        """Automatically detect best available device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        creator = get_model_creator(self.model_name)
        model = creator(
            input_channels=3,
            num_classes=self.num_classes,
            input_size=self.input_size
        )
        return model.to(self.device)
    
    def _create_data_loaders(self, 
                           train_dir: str, 
                           val_dir: str) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders"""
        
        # Create datasets
        train_dataset = CellDataset(
            data_dir=train_dir,
            input_size=self.input_size,
            num_classes=self.num_classes,
            is_training=True
        )
        
        val_dataset = CellDataset(
            data_dir=val_dir,
            input_size=self.input_size,
            num_classes=self.num_classes,
            is_training=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def train_single_arm(self, 
                        arm_dir: str,
                        val_dir: str,
                        output_dir: str,
                        arm_name: str,
                        seed: int) -> Dict[str, Any]:
        """
        Train model on a single arm
        
        Args:
            arm_dir: Path to arm training data
            val_dir: Path to validation data (fixed across arms)
            output_dir: Output directory for this training run
            arm_name: Name of the arm
            seed: Random seed for this run
            
        Returns:
            Training results dictionary
        """
        # Set seed for this run
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create output directory
        run_output_dir = Path(output_dir) / f"{arm_name}_seed{seed}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        run_config = self.config.copy()
        run_config.update({
            'arm_name': arm_name,
            'arm_dir': str(arm_dir),
            'val_dir': str(val_dir),
            'run_seed': seed,
            'output_dir': str(run_output_dir),
            'start_time': datetime.now().isoformat()
        })
        
        with open(run_output_dir / 'config.json', 'w') as f:
            json.dump(run_config, f, indent=2)
        
        self.logger.info(f"Starting training for {arm_name} with seed {seed}")
        
        # Create model
        model = self._create_model()
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(arm_dir, val_dir)
        
        # Calculate total optimizer steps (fixed across arms)
        # This ensures fair comparison even if dataset sizes differ
        target_steps = 1000  # Fixed number of optimizer steps
        steps_per_epoch = len(train_loader)
        adjusted_epochs = max(1, target_steps // steps_per_epoch)
        
        self.logger.info(f"Training for {adjusted_epochs} epochs ({steps_per_epoch} steps/epoch)")
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.initial_lr,
            weight_decay=1e-4
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=adjusted_epochs
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_dice = 0.0
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_dices = []
        
        start_time = time.time()
        
        for epoch in range(adjusted_epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_batches += 1
                
                # Break if we've reached target steps
                if (epoch * len(train_loader) + batch_idx + 1) >= target_steps:
                    break
            
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase - validate every epoch for short runs
            val_interval = 1 if adjusted_epochs <= 10 else self.config['val_interval']
            if epoch % val_interval == 0:
                model.eval()
                val_loss = 0.0
                val_dice = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for val_images, val_masks in val_loader:
                        val_images = val_images.to(self.device)
                        val_masks = val_masks.to(self.device)
                        
                        val_outputs = model(val_images)
                        v_loss = criterion(val_outputs, val_masks)
                        val_loss += v_loss.item()
                        
                        # Calculate Dice score
                        pred_masks = torch.argmax(val_outputs, dim=1)
                        dice = self._calculate_dice(pred_masks, val_masks)
                        val_dice += dice
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                avg_val_dice = val_dice / val_batches
                
                val_losses.append(avg_val_loss)
                val_dices.append(avg_val_dice)
                
                self.logger.info(
                    f"Epoch {epoch}/{adjusted_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Val Dice: {avg_val_dice:.4f}"
                )
                
                # Save best model
                if avg_val_dice > best_dice:
                    best_dice = avg_val_dice
                    torch.save(model.state_dict(), run_output_dir / 'best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config['early_stopping_patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            scheduler.step()
            
            # Break if we've reached target steps
            if (epoch + 1) * len(train_loader) >= target_steps:
                break
        
        # Save final model
        torch.save(model.state_dict(), run_output_dir / 'final_model.pth')
        
        # Training results
        training_time = time.time() - start_time
        results = {
            'arm_name': arm_name,
            'seed': seed,
            'best_dice': best_dice,
            'final_epoch': epoch,
            'training_time': training_time,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_dices': val_dices,
            'total_steps': min(target_steps, (epoch + 1) * len(train_loader)),
            'config': run_config
        }
        
        # Save results
        with open(run_output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(
            f"Completed training for {arm_name} (seed {seed}): "
            f"Best Dice: {best_dice:.4f}, Time: {training_time:.1f}s"
        )
        
        return results
    
    def _calculate_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice coefficient (macro-averaged for multi-class)"""
        smooth = 1e-6
        
        if self.num_classes == 2:
            # Binary case
            pred_fg = (pred > 0).float()
            target_fg = (target > 0).float()
            
            intersection = (pred_fg * target_fg).sum()
            union = pred_fg.sum() + target_fg.sum()
            
            dice = (2.0 * intersection + smooth) / (union + smooth)
            return dice.item()
        else:
            # Multi-class case: macro-average over classes (excluding background)
            dice_scores = []
            for c in range(1, self.num_classes):  # Skip background class 0
                pred_c = (pred == c).float()
                target_c = (target == c).float()
                
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()
                
                if union > 0:  # Only compute if class is present
                    dice_c = (2.0 * intersection + smooth) / (union + smooth)
                    dice_scores.append(dice_c.item())
            
            return np.mean(dice_scores) if dice_scores else 0.0
    
    def train_all_arms_all_seeds(self, 
                                arms_dir: str,
                                val_dir: str,
                                output_dir: str,
                                seeds: List[int] = [0, 1, 2]) -> Dict[str, List[Dict]]:
        """
        Train all arms with all seeds
        
        Args:
            arms_dir: Directory containing all arms
            val_dir: Validation data directory (fixed)
            output_dir: Output directory for all results
            seeds: List of seeds to use
            
        Returns:
            Dictionary mapping arm names to lists of results for each seed
        """
        arms_dir = Path(arms_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        # Find all arm directories
        arm_dirs = [d for d in arms_dir.iterdir() if d.is_dir()]
        
        self.logger.info(f"Training {len(arm_dirs)} arms with seeds {seeds}")
        
        for arm_dir in arm_dirs:
            arm_name = arm_dir.name
            all_results[arm_name] = []
            
            self.logger.info(f"Training arm: {arm_name}")
            
            for seed in seeds:
                try:
                    results = self.train_single_arm(
                        arm_dir=str(arm_dir),
                        val_dir=val_dir,
                        output_dir=str(output_dir),
                        arm_name=arm_name,
                        seed=seed
                    )
                    all_results[arm_name].append(results)
                except Exception as e:
                    self.logger.error(f"Failed to train {arm_name} with seed {seed}: {e}")
                    continue
        
        # Save overall results
        summary_results = {
            'model_name': self.model_name,
            'config': self.config,
            'arms': list(all_results.keys()),
            'seeds': seeds,
            'results': all_results,
            'completion_time': datetime.now().isoformat()
        }
        
        with open(output_dir / f'{self.model_name}_all_results.json', 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        self.logger.info(f"Completed training all arms with {self.model_name}")
        return all_results


class MultiModelTrainer:
    """Trainer for multiple models across all arms"""
    
    def __init__(self, 
                 models: List[str] = ['nnunet', 'unet'],
                 seeds: List[int] = [0, 1, 2],
                 **trainer_kwargs):
        """
        Initialize multi-model trainer
        
        Args:
            models: List of models to train
            seeds: List of seeds to use
            **trainer_kwargs: Arguments for individual trainers
        """
        self.models = models
        self.seeds = seeds
        self.trainer_kwargs = trainer_kwargs
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_all_models_all_arms(self, 
                                 arms_dir: str,
                                 val_dir: str,
                                 output_dir: str) -> Dict[str, Dict]:
        """
        Train all models on all arms with all seeds
        
        Args:
            arms_dir: Directory containing all arms
            val_dir: Validation data directory
            output_dir: Output directory for all results
            
        Returns:
            Nested dictionary: {model_name: {arm_name: [results]}}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_model_results = {}
        
        for model_name in self.models:
            self.logger.info(f"Training model: {model_name}")
            
            # Create trainer for this model
            trainer = AugmentationStudyTrainer(
                model_name=model_name,
                **self.trainer_kwargs
            )
            
            # Train on all arms
            model_output_dir = output_dir / model_name
            model_results = trainer.train_all_arms_all_seeds(
                arms_dir=arms_dir,
                val_dir=val_dir,
                output_dir=str(model_output_dir),
                seeds=self.seeds
            )
            
            all_model_results[model_name] = model_results
        
        # Save comprehensive results
        comprehensive_results = {
            'models': self.models,
            'seeds': self.seeds,
            'results': all_model_results,
            'completion_time': datetime.now().isoformat()
        }
        
        with open(output_dir / 'comprehensive_results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        self.logger.info("Completed training all models on all arms")
        return all_model_results


def main():
    """Test function for training protocol"""
    print("Training Protocol module loaded successfully")


if __name__ == "__main__":
    main()
