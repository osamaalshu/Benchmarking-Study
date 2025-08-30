#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error-Aware MAUNet: Enhanced Modality-Aware Anti-Ambiguity U-Net for Multi-Modality Cell Segmentation
with Multi-Objective Loss Function and Auxiliary Task Integration

This implementation addresses systematic error patterns identified through comprehensive error analysis:
1. High false negative rates in crowded cellular regions
2. Cell merging in dense arrangements
3. Low Recognition Quality despite strong boundary Segmentation Quality

Key improvements:
- Multi-objective composite loss function (Focal + Tversky + Boundary)
- Auxiliary tasks: signed distance transform and centroid heatmap prediction
- Enhanced post-processing with seeded watershed segmentation
"""

from typing import Sequence, Tuple, Type, Union, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, optional_import
from torchvision import models

rearrange, _ = optional_import("einops", name="rearrange")


class ErrorAwareMAUNet(nn.Module):
    """
    Error-Aware MAUNet with enhanced detection capabilities and auxiliary tasks
    
    This model addresses the systematic error patterns through:
    1. Multi-path decoder architecture for classification, regression, and auxiliary tasks
    2. Auxiliary centroid heatmap prediction for improved instance detection
    3. Enhanced signed distance transform prediction for boundary localization
    4. Anti-ambiguity embedding regularization
    """
    
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        feature_size: int = 64,
        feature_size2: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        normalize: bool = True,
        spatial_dims: int = 2,
        backbone: str = "resnet50",  # "resnet50" or "wide_resnet50"
        enable_auxiliary_tasks: bool = True,
        centroid_sigma: float = 2.0,  # Gaussian sigma for centroid heatmaps
    ) -> None:
        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.normalize = normalize
        self.backbone_type = backbone
        self.enable_auxiliary_tasks = enable_auxiliary_tasks
        self.centroid_sigma = centroid_sigma
        
        # Initialize backbone with compatibility for different torchvision versions
        if backbone == "resnet50":
            try:
                from torchvision.models import ResNet50_Weights
                self.res = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            except ImportError:
                self.res = models.resnet50(pretrained=True)
        elif backbone == "wide_resnet50":
            try:
                from torchvision.models import Wide_ResNet50_2_Weights
                self.res = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            except ImportError:
                self.res = models.wide_resnet50_2(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        del self.res.fc  # Remove fully connected layer
        
        # Shared encoder blocks
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=64,  # ResNet x0 channels
            out_channels=feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=256,  # ResNet x1 channels
            out_channels=2 * feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=512,  # ResNet x2 channels
            out_channels=4 * feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=1024,  # ResNet x3 channels
            out_channels=8 * feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2048,  # ResNet x4 channels
            out_channels=16 * feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # Classification decoder path (main segmentation)
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size2,
            out_channels=8 * feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size2,
            out_channels=4 * feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size2,
            out_channels=2 * feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        # Distance transform decoder path (enhanced boundary localization)
        self.decoder5_dist = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size2,
            out_channels=8 * feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4_dist = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size2,
            out_channels=4 * feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3_dist = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size2,
            out_channels=2 * feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2_dist = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1_dist = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        # Centroid heatmap decoder path (auxiliary task for instance detection)
        if self.enable_auxiliary_tasks:
            self.decoder5_centroid = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=16 * feature_size2,
                out_channels=8 * feature_size2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

            self.decoder4_centroid = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=8 * feature_size2,
                out_channels=4 * feature_size2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

            self.decoder3_centroid = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=4 * feature_size2,
                out_channels=2 * feature_size2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

            self.decoder2_centroid = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=2 * feature_size2,
                out_channels=feature_size2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

            self.decoder1_centroid = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size2,
                out_channels=feature_size2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

        # Embedding head for anti-ambiguity regularization
        self.emb_dim = 64
        self.emb_head = nn.Sequential(
            nn.Conv2d(feature_size2, self.emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.emb_dim),
            nn.ReLU(inplace=True),
        )

        # Learnable proxies for embedding regularization
        self.num_proxy_classes = out_channels if out_channels in (2, 3) else 2
        self.proxies = nn.Parameter(torch.randn(self.num_proxy_classes, self.emb_dim))
        nn.init.kaiming_uniform_(self.proxies, a=math.sqrt(5))

        # Output layers
        self.out_seg = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2,
            out_channels=out_channels
        )
        
        self.out_dist = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2,
            out_channels=1  # Single channel for signed distance transform
        )

        if self.enable_auxiliary_tasks:
            self.out_centroid = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size2,
                out_channels=1  # Single channel for centroid heatmap
            )

    def forward(self, x_in):
        # Shared backbone feature extraction
        x = self.res.conv1(x_in)
        x = self.res.bn1(x)
        x0 = self.res.relu(x)  # x0 is before maxpool (128x128)
        x = self.res.maxpool(x0)
        x1 = self.res.layer1(x)
        x2 = self.res.layer2(x1)
        x3 = self.res.layer3(x2)
        x4 = self.res.layer4(x3)
        
        hidden_states_out = [x0, x1, x2, x3, x4]
        
        # Shared encoders
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        enc4 = self.encoder5(hidden_states_out[3])
        enc5 = self.encoder10(hidden_states_out[4])
        
        # Main classification decoder
        dec4_seg = self.decoder5(enc5, enc4)
        dec3_seg = self.decoder4(dec4_seg, enc3)
        dec2_seg = self.decoder3(dec3_seg, enc2)
        dec1_seg = self.decoder2(dec2_seg, enc1)
        dec0_seg = self.decoder1(dec1_seg, enc0)
        
        # Generate embedding for anti-ambiguity regularization
        emb = self.emb_head(dec0_seg)
        
        # Main segmentation output
        logits_seg = self.out_seg(dec0_seg)
        
        # Distance transform decoder
        dec4_dist = self.decoder5_dist(enc5, enc4)
        dec3_dist = self.decoder4_dist(dec4_dist, enc3)
        dec2_dist = self.decoder3_dist(dec3_dist, enc2)
        dec1_dist = self.decoder2_dist(dec2_dist, enc1)
        dec0_dist = self.decoder1_dist(dec1_dist, enc0)
        
        # Distance transform output (signed distance transform)
        logits_dist = self.out_dist(dec0_dist)
        
        outputs = {
            'segmentation': logits_seg,
            'distance_transform': logits_dist,
            'embedding': emb
        }
        
        # Auxiliary centroid heatmap decoder (if enabled)
        if self.enable_auxiliary_tasks:
            dec4_centroid = self.decoder5_centroid(enc5, enc4)
            dec3_centroid = self.decoder4_centroid(dec4_centroid, enc3)
            dec2_centroid = self.decoder3_centroid(dec3_centroid, enc2)
            dec1_centroid = self.decoder2_centroid(dec2_centroid, enc1)
            dec0_centroid = self.decoder1_centroid(dec1_centroid, enc0)
            
            # Centroid heatmap output
            logits_centroid = self.out_centroid(dec0_centroid)
            outputs['centroid_heatmap'] = logits_centroid
        
        return outputs

    def generate_centroid_heatmaps(self, instance_masks, sigma=None):
        """
        Generate ground truth centroid heatmaps from instance masks
        
        Args:
            instance_masks: (B, 1, H, W) tensor with instance labels
            sigma: Gaussian sigma for heatmap generation
            
        Returns:
            centroid_heatmaps: (B, 1, H, W) tensor with Gaussian heatmaps at centroids
        """
        if sigma is None:
            sigma = self.centroid_sigma
            
        batch_size, _, height, width = instance_masks.shape
        device = instance_masks.device
        
        centroid_heatmaps = torch.zeros_like(instance_masks, dtype=torch.float32)
        
        for b in range(batch_size):
            mask = instance_masks[b, 0].cpu().numpy()
            unique_labels = np.unique(mask)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background
            
            heatmap = np.zeros((height, width), dtype=np.float32)
            
            for label in unique_labels:
                # Find centroid of each instance
                y_coords, x_coords = np.where(mask == label)
                if len(y_coords) > 0:
                    centroid_y = int(np.mean(y_coords))
                    centroid_x = int(np.mean(x_coords))
                    
                    # Generate Gaussian heatmap around centroid
                    y_grid, x_grid = np.ogrid[:height, :width]
                    gaussian = np.exp(-((x_grid - centroid_x)**2 + (y_grid - centroid_y)**2) / (2 * sigma**2))
                    heatmap = np.maximum(heatmap, gaussian)  # Take maximum to handle overlaps
            
            centroid_heatmaps[b, 0] = torch.from_numpy(heatmap).to(device)
        
        return centroid_heatmaps


def create_error_aware_maunet_model(
    num_classes: int = 3,
    input_size: int = 256,
    in_channels: int = 3,
    backbone: str = "resnet50",
    enable_auxiliary_tasks: bool = True,
    centroid_sigma: float = 2.0
):
    """
    Factory function to create Error-Aware MAUNet model
    
    Args:
        num_classes: Number of output classes
        input_size: Input image size (assumed square)
        in_channels: Number of input channels
        backbone: Backbone architecture ("resnet50" or "wide_resnet50")
        enable_auxiliary_tasks: Whether to enable auxiliary centroid prediction
        centroid_sigma: Gaussian sigma for centroid heatmap generation
    
    Returns:
        Error-Aware MAUNet model instance
    """
    model = ErrorAwareMAUNet(
        img_size=(input_size, input_size),
        in_channels=in_channels,
        out_channels=num_classes,
        spatial_dims=2,
        backbone=backbone,
        enable_auxiliary_tasks=enable_auxiliary_tasks,
        centroid_sigma=centroid_sigma
    )
    return model


class ErrorAwareMAUNetEnsemble(nn.Module):
    """
    Ensemble wrapper for Error-Aware MAUNet models
    
    Combines predictions from multiple backbone variants for improved robustness
    """
    
    def __init__(
        self,
        models_list: List[ErrorAwareMAUNet],
        average: bool = True,
    ) -> None:
        super().__init__()
        if not models_list:
            raise ValueError("models_list must contain at least one ErrorAwareMAUNet instance")
        self.models = nn.ModuleList(models_list)
        self.average = average

    def forward(self, x: torch.Tensor):
        seg_logits_list = []
        dist_logits_list = []
        centroid_logits_list = []
        
        for model in self.models:
            outputs = model(x)
            seg_logits_list.append(outputs['segmentation'])
            dist_logits_list.append(outputs['distance_transform'])
            if 'centroid_heatmap' in outputs:
                centroid_logits_list.append(outputs['centroid_heatmap'])

        # Stack and reduce
        seg_stack = torch.stack(seg_logits_list, dim=0)
        dist_stack = torch.stack(dist_logits_list, dim=0)
        
        if self.average:
            seg_logits = seg_stack.mean(dim=0)
            dist_logits = dist_stack.mean(dim=0)
        else:
            seg_logits = seg_stack.sum(dim=0)
            dist_logits = dist_stack.sum(dim=0)

        ensemble_outputs = {
            'segmentation': seg_logits,
            'distance_transform': dist_logits
        }
        
        if centroid_logits_list:
            centroid_stack = torch.stack(centroid_logits_list, dim=0)
            if self.average:
                ensemble_outputs['centroid_heatmap'] = centroid_stack.mean(dim=0)
            else:
                ensemble_outputs['centroid_heatmap'] = centroid_stack.sum(dim=0)

        return ensemble_outputs


def create_error_aware_maunet_ensemble_model(
    num_classes: int = 3,
    input_size: int = 256,
    in_channels: int = 3,
    backbones: Optional[List[str]] = None,
    average: bool = True,
    enable_auxiliary_tasks: bool = True,
    centroid_sigma: float = 2.0
):
    """
    Factory for Error-Aware MAUNet ensemble of multiple backbones
    
    Args:
        num_classes: number of segmentation classes
        input_size: input square size
        in_channels: number of input channels
        backbones: list of backbones to include, defaults to ["resnet50", "wide_resnet50"]
        average: whether to average or sum logits across members
        enable_auxiliary_tasks: whether to enable auxiliary centroid prediction
        centroid_sigma: Gaussian sigma for centroid heatmap generation
    """
    if backbones is None:
        backbones = ["resnet50", "wide_resnet50"]

    members = []
    for backbone in backbones:
        members.append(
            ErrorAwareMAUNet(
                img_size=(input_size, input_size),
                in_channels=in_channels,
                out_channels=num_classes,
                spatial_dims=2,
                backbone=backbone,
                enable_auxiliary_tasks=enable_auxiliary_tasks,
                centroid_sigma=centroid_sigma
            )
        )

    return ErrorAwareMAUNetEnsemble(models_list=members, average=average)


# Import numpy for centroid heatmap generation
import numpy as np
