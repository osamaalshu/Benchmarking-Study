#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Official nnU-Net Implementation for Biomedical Image Segmentation
Based on: https://github.com/mic-dkfz/nnunet

This implementation uses MONAI's UNet as the base architecture but applies
nnU-Net configuration principles for automatic adaptation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union, Optional, List
import monai


class nnUNet(nn.Module):
    """
    Official nnU-Net: Self-configuring U-Net for biomedical image segmentation
    
    This implementation uses MONAI's UNet as the base architecture and applies
    nnU-Net configuration principles for automatic adaptation.
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 3,  # background, interior, boundary
                 base_filters: int = 32,
                 depth: int = 5,
                 num_blocks_per_level: int = 2,
                 use_instancenorm: bool = True,
                 dropout_rate: float = 0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.depth = depth
        self.num_blocks_per_level = num_blocks_per_level
        self.use_instancenorm = use_instancenorm
        
        # Calculate filter sizes for each depth level (following nnU-Net pattern)
        self.filters = [base_filters]
        for i in range(depth - 1):
            self.filters.append(self.filters[-1] * 2)
        
        print(f"nnU-Net filters: {self.filters}")
        
        # Create MONAI UNet with nnU-Net configuration
        self.unet = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=self.filters,
            strides=[2] * (depth - 1),
            num_res_units=num_blocks_per_level,
            norm=("INSTANCE" if use_instancenorm else "BATCH"),
            dropout=dropout_rate,
            act=("LEAKYRELU", {"negative_slope": 0.01}),
        )
        
        # Initialize weights using nnU-Net's initialization strategy
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using nnU-Net's initialization strategy"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nnU-Net uses He initialization with ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.unet(x)


class nnUNetConfigurator:
    """
    nnU-Net configuration class that follows official nnU-Net principles
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int],
                 in_channels: int = 3,
                 out_channels: int = 3,
                 gpu_memory_gb: float = 8.0):
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gpu_memory_gb = gpu_memory_gb
        
        # Analyze dataset properties following nnU-Net principles
        self._analyze_dataset()
        
    def _analyze_dataset(self):
        """Analyze dataset properties following official nnU-Net principles"""
        # Calculate image area
        self.image_area = self.image_size[0] * self.image_size[1]
        
        # Determine optimal patch size based on image size (nnU-Net principle)
        if self.image_area <= 256 * 256:
            self.patch_size = (256, 256)
        elif self.image_area <= 512 * 512:
            self.patch_size = (512, 512)
        else:
            self.patch_size = (768, 768)
        
        # Determine network depth based on image size (nnU-Net principle)
        min_dim = min(self.image_size)
        if min_dim <= 128:
            self.depth = 4
        elif min_dim <= 256:
            self.depth = 5
        elif min_dim <= 512:
            self.depth = 5  # nnU-Net typically uses 5 levels for 512x512
        else:
            self.depth = 6
        
        # Determine base filters based on GPU memory (nnU-Net principle)
        if self.gpu_memory_gb >= 16:
            self.base_filters = 64
        elif self.gpu_memory_gb >= 8:
            self.base_filters = 32  # More conservative for better training
        else:
            self.base_filters = 32
        
        # Determine batch size (nnU-Net principle)
        if self.gpu_memory_gb >= 16:
            self.batch_size = 4
        elif self.gpu_memory_gb >= 8:
            self.batch_size = 2
        else:
            self.batch_size = 1
    
    def get_config(self):
        """Get the optimal configuration following nnU-Net principles"""
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'base_filters': self.base_filters,
            'depth': self.depth,
            'num_blocks_per_level': 2,  # nnU-Net standard
            'patch_size': self.patch_size,
            'batch_size': self.batch_size,
            'use_instancenorm': True,  # nnU-Net standard
            'dropout_rate': 0.0  # nnU-Net typically doesn't use dropout during training
        }
    
    def create_model(self):
        """Create and return a configured nnU-Net model"""
        config = self.get_config()
        return nnUNet(**{k: v for k, v in config.items() 
                        if k in ['in_channels', 'out_channels', 'base_filters', 
                                'depth', 'num_blocks_per_level', 'use_instancenorm', 'dropout_rate']})


def create_nnunet_model(image_size: Tuple[int, int], 
                       in_channels: int = 3,
                       out_channels: int = 3,
                       gpu_memory_gb: float = 8.0) -> nnUNet:
    """
    Convenience function to create a configured nnU-Net model
    
    Args:
        image_size: Tuple of (height, width) of input images
        in_channels: Number of input channels
        out_channels: Number of output classes
        gpu_memory_gb: Available GPU memory in GB
    
    Returns:
        Configured nnU-Net model following official architecture
    """
    configurator = nnUNetConfigurator(image_size, in_channels, out_channels, gpu_memory_gb)
    return configurator.create_model()


if __name__ == "__main__":
    # Example usage
    image_size = (512, 512)
    model = create_nnunet_model(image_size, in_channels=3, out_channels=3, gpu_memory_gb=8.0)
    
    # Test forward pass
    x = torch.randn(1, 3, 512, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print configuration
    configurator = nnUNetConfigurator(image_size, 3, 3, 8.0)
    print(f"nnU-Net Configuration: {configurator.get_config()}") 