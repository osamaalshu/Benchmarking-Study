#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAUNet: Modality-Aware Anti-Ambiguity U-Net for Multi-Modality Cell Segmentation
Adapted from neurips22-cellseg_saltfish repository
"""

from typing import Sequence, Tuple, Type, Union, List, Optional
import torch
import torch.nn as nn

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep, optional_import
from torchvision import models

rearrange, _ = optional_import("einops", name="rearrange")


class MAUNet(nn.Module):
    """
    MAUNet based on ResNet backbone with dual decoder paths
    for classification and regression (distance transform)
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
        backbone: str = "resnet50"  # "resnet50" or "wide_resnet50"
    ) -> None:
        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.normalize = normalize
        self.backbone_type = backbone
        
        # Initialize backbone with compatibility for different torchvision versions
        if backbone == "resnet50":
            try:
                # New torchvision API (>= 0.13)
                from torchvision.models import ResNet50_Weights
                self.res = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            except ImportError:
                # Fallback for older torchvision versions
                self.res = models.resnet50(pretrained=True)
        elif backbone == "wide_resnet50":
            try:
                # New torchvision API (>= 0.13)
                from torchvision.models import Wide_ResNet50_2_Weights
                self.res = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            except ImportError:
                # Fallback for older torchvision versions
                self.res = models.wide_resnet50_2(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        del self.res.fc  # Remove fully connected layer
        
        # Encoder blocks
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
            in_channels=2048,  # ResNet x4 channels (same for both backbones)
            out_channels=16 * feature_size2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # Classification decoder path
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

        # Regression decoder path (for distance transform)
        self.decoder5_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size2,
            out_channels=8 * feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size2,
            out_channels=4 * feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size2,
            out_channels=2 * feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1_2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2,
            out_channels=feature_size2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        # Output layers
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2,
            out_channels=out_channels
        )
        
        self.out_2 = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size2,
            out_channels=1  # Single channel for distance transform
        )

    def forward(self, x_in):
        x = self.res.conv1(x_in)
        x = self.res.bn1(x)
        x0 = self.res.relu(x)  # x0 is before maxpool (128x128)
        x = self.res.maxpool(x0)
        x1 = self.res.layer1(x)
        x2 = self.res.layer2(x1)
        x3 = self.res.layer3(x2)
        x4 = self.res.layer4(x3)
        
        hidden_states_out = [x0, x1, x2, x3, x4]
        
        # Encoders
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        enc4 = self.encoder5(hidden_states_out[3])
        enc5 = self.encoder10(hidden_states_out[4])
        
        # Classification decoder
        dec4 = self.decoder5(enc5, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
            
        dec0 = self.decoder1(dec1, enc0)
        logits = self.out(dec0)
        
        # Regression decoder (distance transform)
        dec4_2 = self.decoder5_2(enc5, enc4)
        dec3_2 = self.decoder4_2(dec4_2, enc3)
        dec2_2 = self.decoder3_2(dec3_2, enc2)
        dec1_2 = self.decoder2_2(dec2_2, enc1)
            
        dec0_2 = self.decoder1_2(dec1_2, enc0)
        logits_2 = self.out_2(dec0_2)
        
        return logits, logits_2


def create_maunet_model(
    num_classes: int = 3,
    input_size: int = 256,
    in_channels: int = 3,
    backbone: str = "resnet50"
):
    """
    Factory function to create MAUNet model
    
    Args:
        num_classes: Number of output classes
        input_size: Input image size (assumed square)
        in_channels: Number of input channels
        backbone: Backbone architecture ("resnet50" or "wide_resnet50")
    
    Returns:
        MAUNet model instance
    """
    model = MAUNet(
        img_size=(input_size, input_size),
        in_channels=in_channels,
        out_channels=num_classes,
        spatial_dims=2,
        backbone=backbone
    )
    return model


# Custom loss for distance transform
class WeightedL1Loss(nn.Module):
    """Weighted L1 loss for distance transform regression"""
    
    def __init__(self):
        super(WeightedL1Loss, self).__init__()
        
    def forward(self, inputs, weight):
        weight2 = (weight == 0)
        weight1 = (weight > 0)
        num1 = torch.sum(weight2)
        num2 = torch.sum(weight1)
        loss = (torch.sum(torch.abs(inputs * weight2)) / (num1 + 1e-3) + 
                torch.sum(torch.abs(inputs * weight1 - weight)) / (num2 + 1e-3))
        return loss 


class MAUNetEnsemble(nn.Module):
    """Ensemble wrapper that averages logits from multiple MAUNet backbones.

    This mirrors the "complete model" reported by the saltfish team, which
    ensembles ResNet50 and Wide-ResNet50 variants.
    """

    def __init__(
        self,
        models_list: List[MAUNet],
        average: bool = True,
    ) -> None:
        super().__init__()
        if not models_list:
            raise ValueError("models_list must contain at least one MAUNet instance")
        self.models = nn.ModuleList(models_list)
        self.average = average

    def forward(self, x: torch.Tensor):
        class_logits_list: List[torch.Tensor] = []
        reg_logits_list: List[torch.Tensor] = []

        for model in self.models:
            logits_cls, logits_reg = model(x)
            class_logits_list.append(logits_cls)
            reg_logits_list.append(logits_reg)

        # Stack along new dim and reduce
        class_stack = torch.stack(class_logits_list, dim=0)
        reg_stack = torch.stack(reg_logits_list, dim=0)

        if self.average:
            class_logits = class_stack.mean(dim=0)
            reg_logits = reg_stack.mean(dim=0)
        else:
            # Sum as an alternative; caller can scale later
            class_logits = class_stack.sum(dim=0)
            reg_logits = reg_stack.sum(dim=0)

        return class_logits, reg_logits


def create_maunet_ensemble_model(
    num_classes: int = 3,
    input_size: int = 256,
    in_channels: int = 3,
    backbones: Optional[List[str]] = None,
    average: bool = True,
):
    """Factory for MAUNet ensemble of multiple backbones.

    Args:
        num_classes: number of segmentation classes
        input_size: input square size
        in_channels: number of input channels
        backbones: list of backbones to include, defaults to ["resnet50", "wide_resnet50"]
        average: whether to average or sum logits across members
    """
    if backbones is None:
        backbones = ["resnet50", "wide_resnet50"]

    members: List[MAUNet] = []
    for backbone in backbones:
        members.append(
            MAUNet(
                img_size=(input_size, input_size),
                in_channels=in_channels,
                out_channels=num_classes,
                spatial_dims=2,
                backbone=backbone,
            )
        )

    return MAUNetEnsemble(models_list=members, average=average)