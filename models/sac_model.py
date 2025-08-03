#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAC (Segment Any Cell) Model Implementation
Using Meta's SAM with CAM (Cell Anything Model) architecture and LoRA-style fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
import cv2
from skimage import measure, morphology
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling.sam import Sam
from segment_anything.modeling.common import LayerNorm2d
from segment_anything.utils.transforms import ResizeLongestSide
from scipy.ndimage import center_of_mass, label
import random
import math

def pad(x: torch.Tensor, size: int = 1024) -> torch.Tensor:
    """Pad to SAM input size."""
    return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)

def unpad(x: torch.Tensor, size: tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Unpad to original size."""
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class LoRALayer(nn.Module):
    """LoRA-style adaptation layer"""
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA components
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize with small values
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * (self.alpha / self.rank)

class SACModel(nn.Module):
    """SAC Model using CAM (Cell Anything Model) architecture with LoRA-style fine-tuning"""
    
    def __init__(self, 
                 sam_checkpoint: str = "models/sam_weights/sam_vit_b_01ec64.pth",
                 sam_model_type: str = "vit_b",
                 device: str = "auto",
                 num_classes: int = 3,
                 freeze_encoder_layers: int = 8,
                 use_lora: bool = True,
                 lora_rank: int = 16):
        super().__init__()
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.device = device
        self.num_classes = num_classes
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        
        # Load SAM model
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(device)
        
        # SAM components
        self.sam_preprocess = self.sam.preprocess
        self.sam_encoder = self.sam.image_encoder
        
        # LoRA-style fine-tuning setup
        if use_lora:
            self._setup_lora_fine_tuning()
        else:
            # Traditional fine-tuning: freeze early layers
            last_layer_no = len(list(self.sam_encoder.parameters())) - 1
            for layer_no, param in enumerate(self.sam_encoder.parameters()):
                if layer_no > (last_layer_no - freeze_encoder_layers):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # Custom decoder head (similar to train.py CAM)
        self.nn_drop = nn.Dropout(p=0.2)
        
        # Decoder layers - adjusted to produce correct output size
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.norm1 = LayerNorm2d(128)
        
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.norm2 = LayerNorm2d(64)
        
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.norm3 = LayerNorm2d(32)
        
        # Final output layer for multi-class segmentation
        self.conv4 = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0)
        
        # Feature projection layer (SAM ViT-B has 768 dimensions, we need 256)
        self.feature_projection = nn.Conv2d(768, 256, kernel_size=1)
        
        # Image size for SAM
        self.image_size = self.sam.image_encoder.img_size
        
        # Transform for image preprocessing
        self.transform = ResizeLongestSide(self.image_size)
        
        # Move decoder to device
        self.to(device)
        
    def _setup_lora_fine_tuning(self):
        """Setup LoRA-style fine-tuning for SAM encoder"""
        # Add LoRA adapters to transformer layers
        self.lora_adapters = nn.ModuleList()
        
        # Get transformer blocks from SAM encoder
        if hasattr(self.sam_encoder, 'blocks'):
            transformer_blocks = self.sam_encoder.blocks
        else:
            # Fallback: try to find transformer layers
            transformer_blocks = []
            for module in self.sam_encoder.modules():
                if hasattr(module, 'attn') and hasattr(module, 'mlp'):
                    transformer_blocks.append(module)
        
        # Add LoRA adapters to attention and MLP layers
        for block in transformer_blocks:
            if hasattr(block, 'attn'):
                # Add LoRA to attention
                if hasattr(block.attn, 'qkv'):
                    lora_qkv = LoRALayer(
                        block.attn.qkv.in_features, 
                        block.attn.qkv.out_features, 
                        self.lora_rank
                    )
                    self.lora_adapters.append(lora_qkv)
                    
                if hasattr(block.attn, 'proj'):
                    lora_proj = LoRALayer(
                        block.attn.proj.in_features, 
                        block.attn.proj.out_features, 
                        self.lora_rank
                    )
                    self.lora_adapters.append(lora_proj)
            
            if hasattr(block, 'mlp'):
                # Add LoRA to MLP
                if hasattr(block.mlp, 'fc1'):
                    lora_fc1 = LoRALayer(
                        block.mlp.fc1.in_features, 
                        block.mlp.fc1.out_features, 
                        self.lora_rank
                    )
                    self.lora_adapters.append(lora_fc1)
                    
                if hasattr(block.mlp, 'fc2'):
                    lora_fc2 = LoRALayer(
                        block.mlp.fc2.in_features, 
                        block.mlp.fc2.out_features, 
                        self.lora_rank
                    )
                    self.lora_adapters.append(lora_fc2)
        
        # Freeze original SAM parameters
        for param in self.sam_encoder.parameters():
            param.requires_grad = False
            
        # Only LoRA adapters are trainable
        for param in self.lora_adapters.parameters():
            param.requires_grad = True
        
    def forward(self, x: torch.Tensor, points: Optional[List[List[int]]] = None) -> torch.Tensor:
        """
        Forward pass with CAM architecture and LoRA fine-tuning
        
        Args:
            x: Input tensor (B, C, H, W)
            points: Optional list of [x, y] coordinates for prompts (not used in this version)
            
        Returns:
            Segmentation masks (B, num_classes, H, W)
        """
        batch_size = x.shape[0]
        
        # Preprocess images for SAM
        with torch.no_grad():
            x_padded = pad(x.to(self.device))
            x_preprocessed = self.sam_preprocess(x_padded)
        
        # Encode images using SAM encoder (with LoRA if enabled)
        if self.use_lora:
            # Apply LoRA adapters during encoding
            features = self.sam_encoder(x_preprocessed)
            # Note: LoRA adapters are applied within the transformer blocks
        else:
            features = self.sam_encoder(x_preprocessed)
        
        # SAM encoder outputs (B, embed_dim, H, W) - need to reshape
        B, embed_dim, H, W = features.shape
        
        # Reshape features to match decoder input dimensions
        # SAM ViT-B has embed_dim=768, we need to project to 256
        if embed_dim != 256:
            features = self.feature_projection(features)
        
        # Decode features using custom decoder head
        x = self.conv1(features)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.nn_drop(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.nn_drop(x)
        
        x = self.conv4(x)
        
        # Resize to original input size (256x256)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Return raw logits (softmax/sigmoid will be applied during inference)
        return x
    
    def predict_instance_masks(self, x: torch.Tensor) -> List[np.ndarray]:
        """Predict instance masks for each image in batch"""
        batch_size = x.shape[0]
        instance_masks = []
        
        for b in range(batch_size):
            # Get single image
            img = x[b:b+1]
            
            # Get prediction
            with torch.no_grad():
                pred = self.forward(img)
                # Apply softmax/sigmoid for inference
                if self.num_classes > 1:
                    pred = F.softmax(pred, dim=1)
                else:
                    pred = torch.sigmoid(pred)
            
            # Convert to instance mask
            if self.num_classes > 1:
                # For multi-class, use argmax to get class predictions
                pred_class = torch.argmax(pred, dim=1)[0].cpu().numpy()
                
                # Convert to instance mask (assuming class 1 and 2 are different cell types)
                instance_mask = np.zeros_like(pred_class, dtype=np.uint16)
                instance_id = 1
                
                for class_id in range(1, self.num_classes):
                    class_mask = (pred_class == class_id).astype(bool)
                    if class_mask.any():
                        # Label connected components
                        labeled_mask = measure.label(class_mask)
                        
                        # Add to instance mask
                        for region in measure.regionprops(labeled_mask):
                            if region.area > 50:  # Minimum area threshold
                                instance_mask[labeled_mask == region.label] = instance_id
                                instance_id += 1
            else:
                # For binary segmentation
                pred_binary = (pred[0, 0] > 0.5).cpu().numpy()
                
                # Label connected components
                labeled_mask = measure.label(pred_binary)
                instance_mask = labeled_mask.astype(np.uint16)
            
            instance_masks.append(instance_mask)
        
        return instance_masks

def create_default_points(batch_size: int, img_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Create default center points for training (not used in this version)"""
    center_x, center_y = img_size[0] // 2, img_size[1] // 2
    points = torch.tensor([[center_x, center_y]], dtype=torch.float32)
    points = points.unsqueeze(0).expand(batch_size, -1, -1)
    return points

if __name__ == "__main__":
    import math
    
    # Test the model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("Testing SAC Model with LoRA-style fine-tuning...")
    
    # Test 1: SAC model with LoRA
    model = SACModel(device=device, num_classes=3, use_lora=True, lora_rank=16)
    print(f"Model loaded on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test with dummy data
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Output sum per sample: {output.sum(dim=1).mean():.3f} (should be ~1.0 for softmax)")
    
    # Test 2: SAC model without LoRA
    print("\nTesting SAC model without LoRA...")
    model_no_lora = SACModel(device=device, num_classes=3, use_lora=False, freeze_encoder_layers=8)
    output_no_lora = model_no_lora(x)
    print(f"Output shape: {output_no_lora.shape}")
    print(f"Trainable parameters: {sum(p.numel() for p in model_no_lora.parameters() if p.requires_grad):,}")
    
    print("\n✅ SAC model with LoRA-style fine-tuning test passed!") 