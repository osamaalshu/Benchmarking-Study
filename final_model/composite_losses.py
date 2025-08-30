#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composite Loss Functions for Error-Aware MAUNet

This module implements the multi-objective loss function designed to address
systematic error patterns identified in cell segmentation:

L = λ_det * L_focal + λ_seg * L_tversky + λ_bnd * L_boundary(SDT) + λ_aux * L_centroid

Key components:
1. Focal Loss: Addresses class imbalance by down-weighting easy examples
2. Tversky Loss: Asymmetric weighting to reduce false negatives
3. Boundary Loss: Precise boundary localization using signed distance transforms
4. Centroid Loss: Auxiliary task for improved instance detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union
from scipy.ndimage import distance_transform_edt


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Focuses learning on hard examples by down-weighting easily classified samples.
    Particularly effective for dense cellular regions where background dominates.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) raw logits
            targets: (B, C, H, W) one-hot encoded targets
        """
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets.argmax(dim=1), reduction='none')
        
        # Get the probability of the true class
        pt = torch.gather(probs, 1, targets.argmax(dim=1).unsqueeze(1)).squeeze(1)
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply focal weight and alpha
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss with asymmetric weighting for false positive/negative control
    
    Designed to reduce false negatives (missed cells) by setting β > α.
    This directly addresses the systematic under-detection in crowded regions.
    
    Reference: Salehi et al. "Tversky loss function for image segmentation 
               using 3D fully convolutional deep networks" (MICCAI 2017)
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives (β > α reduces FN)
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) raw logits
            targets: (B, C, H, W) one-hot encoded targets
        """
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Flatten tensors
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)  # (B, C, H*W)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)  # (B, C, H*W)
        
        # Compute Tversky coefficient for each class
        tversky_loss = 0
        num_classes = probs.size(1)
        
        for c in range(num_classes):
            if c == 0:  # Skip background class or weight differently
                continue
                
            pred_c = probs_flat[:, c, :]
            target_c = targets_flat[:, c, :]
            
            # True positives, false positives, false negatives
            tp = torch.sum(pred_c * target_c, dim=1)
            fp = torch.sum(pred_c * (1 - target_c), dim=1)
            fn = torch.sum((1 - pred_c) * target_c, dim=1)
            
            # Tversky coefficient
            tversky_coeff = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            
            # Tversky loss (1 - Tversky coefficient)
            tversky_loss += (1 - tversky_coeff).mean()
        
        return tversky_loss / max(1, num_classes - 1)  # Average over foreground classes


class BoundaryLoss(nn.Module):
    """
    Boundary Loss using Signed Distance Transform for precise boundary localization
    
    Encourages accurate boundary placement by leveraging distance transform information.
    Particularly effective for separating merged cells in dense arrangements.
    
    Reference: Kervadec et al. "Boundary loss for highly unbalanced segmentation" (MICCAI 2019)
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(BoundaryLoss, self).__init__()
        self.reduction = reduction
        
    def compute_sdt(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance transform for segmentation mask
        
        Args:
            segmentation: (B, H, W) binary segmentation mask
            
        Returns:
            sdt: (B, H, W) signed distance transform
        """
        batch_size = segmentation.shape[0]
        device = segmentation.device
        
        sdt_batch = torch.zeros_like(segmentation, dtype=torch.float32)
        
        for b in range(batch_size):
            mask = segmentation[b].cpu().numpy().astype(np.uint8)
            
            # Compute distance transform for foreground and background
            posmask = mask.astype(bool)
            negmask = ~posmask
            
            if posmask.any():
                pos_edt = distance_transform_edt(posmask)
            else:
                pos_edt = np.zeros_like(mask, dtype=np.float32)
                
            if negmask.any():
                neg_edt = distance_transform_edt(negmask)
            else:
                neg_edt = np.zeros_like(mask, dtype=np.float32)
            
            # Signed distance transform: positive inside, negative outside
            sdt = pos_edt - neg_edt
            sdt_batch[b] = torch.from_numpy(sdt.astype(np.float32)).to(device)
        
        return sdt_batch
    
    def forward(self, pred_sdt: torch.Tensor, target_seg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_sdt: (B, 1, H, W) predicted signed distance transform
            target_seg: (B, C, H, W) one-hot segmentation targets
        """
        # Convert one-hot to binary mask (foreground vs background)
        if target_seg.size(1) > 1:
            target_binary = target_seg[:, 1:].sum(dim=1)  # Sum all foreground classes
        else:
            target_binary = target_seg.squeeze(1)
        
        # Compute ground truth signed distance transform
        target_sdt = self.compute_sdt(target_binary)
        
        # Boundary loss: L1 distance between predicted and target SDT
        pred_sdt_squeezed = pred_sdt.squeeze(1)
        boundary_loss = F.l1_loss(pred_sdt_squeezed, target_sdt, reduction='none')
        
        # Weight by distance to focus on boundary regions
        weight = torch.exp(-torch.abs(target_sdt) / 5.0)  # Higher weight near boundaries
        weighted_loss = boundary_loss * weight
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class CentroidLoss(nn.Module):
    """
    Centroid Heatmap Loss for auxiliary instance detection task
    
    Supervises the auxiliary centroid prediction head to improve instance detection
    in low-contrast cellular regions where traditional approaches fail.
    """
    
    def __init__(self, sigma: float = 2.0, reduction: str = 'mean'):
        super(CentroidLoss, self).__init__()
        self.sigma = sigma
        self.reduction = reduction
        
    def generate_centroid_targets(self, instance_masks: torch.Tensor) -> torch.Tensor:
        """
        Generate ground truth centroid heatmaps from instance masks
        
        Args:
            instance_masks: (B, H, W) instance segmentation masks
            
        Returns:
            centroid_heatmaps: (B, 1, H, W) Gaussian heatmaps at centroids
        """
        batch_size, height, width = instance_masks.shape
        device = instance_masks.device
        
        centroid_heatmaps = torch.zeros((batch_size, 1, height, width), 
                                      dtype=torch.float32, device=device)
        
        for b in range(batch_size):
            mask = instance_masks[b].cpu().numpy()
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
                    gaussian = np.exp(-((x_grid - centroid_x)**2 + (y_grid - centroid_y)**2) / (2 * self.sigma**2))
                    heatmap = np.maximum(heatmap, gaussian)  # Take maximum for overlaps
            
            centroid_heatmaps[b, 0] = torch.from_numpy(heatmap.astype(np.float32)).to(device)
        
        return centroid_heatmaps
    
    def forward(self, pred_centroids: torch.Tensor, instance_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_centroids: (B, 1, H, W) predicted centroid heatmaps
            instance_masks: (B, H, W) instance segmentation masks
        """
        # Generate ground truth centroid heatmaps
        target_centroids = self.generate_centroid_targets(instance_masks)
        
        # MSE loss between predicted and target heatmaps
        centroid_loss = F.mse_loss(pred_centroids, target_centroids, reduction=self.reduction)
        
        return centroid_loss


class CompositeLoss(nn.Module):
    """
    Multi-Objective Composite Loss Function for Error-Aware MAUNet
    
    Combines multiple loss components to address systematic error patterns:
    L = λ_det * L_focal + λ_seg * L_tversky + λ_bnd * L_boundary + λ_aux * L_centroid
    
    This composite approach directly targets:
    1. Class imbalance (Focal Loss)
    2. False negative reduction (Tversky Loss)
    3. Boundary precision (Boundary Loss)
    4. Instance detection (Centroid Loss)
    """
    
    def __init__(
        self,
        lambda_focal: float = 1.0,
        lambda_tversky: float = 1.0,
        lambda_boundary: float = 0.5,
        lambda_centroid: float = 0.3,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        centroid_sigma: float = 2.0,
    ):
        super(CompositeLoss, self).__init__()
        
        # Loss weights
        self.lambda_focal = lambda_focal
        self.lambda_tversky = lambda_tversky
        self.lambda_boundary = lambda_boundary
        self.lambda_centroid = lambda_centroid
        
        # Individual loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.boundary_loss = BoundaryLoss()
        self.centroid_loss = CentroidLoss(sigma=centroid_sigma)
        
    def forward(
        self, 
        outputs: dict, 
        targets_seg: torch.Tensor,
        targets_instance: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute composite loss from model outputs and targets
        
        Args:
            outputs: Dictionary containing model outputs:
                - 'segmentation': (B, C, H, W) segmentation logits
                - 'distance_transform': (B, 1, H, W) predicted SDT
                - 'centroid_heatmap': (B, 1, H, W) predicted centroids (optional)
            targets_seg: (B, C, H, W) one-hot segmentation targets
            targets_instance: (B, H, W) instance masks for centroid supervision
            
        Returns:
            loss_dict: Dictionary containing individual and total losses
        """
        loss_dict = {}
        total_loss = 0
        
        # Focal loss (detection-focused)
        if self.lambda_focal > 0:
            focal_loss_val = self.focal_loss(outputs['segmentation'], targets_seg)
            loss_dict['focal_loss'] = focal_loss_val
            total_loss += self.lambda_focal * focal_loss_val
        
        # Tversky loss (false negative reduction)
        if self.lambda_tversky > 0:
            tversky_loss_val = self.tversky_loss(outputs['segmentation'], targets_seg)
            loss_dict['tversky_loss'] = tversky_loss_val
            total_loss += self.lambda_tversky * tversky_loss_val
        
        # Boundary loss (precise localization)
        if self.lambda_boundary > 0:
            boundary_loss_val = self.boundary_loss(outputs['distance_transform'], targets_seg)
            loss_dict['boundary_loss'] = boundary_loss_val
            total_loss += self.lambda_boundary * boundary_loss_val
        
        # Centroid loss (auxiliary task)
        if self.lambda_centroid > 0 and 'centroid_heatmap' in outputs and targets_instance is not None:
            centroid_loss_val = self.centroid_loss(outputs['centroid_heatmap'], targets_instance)
            loss_dict['centroid_loss'] = centroid_loss_val
            total_loss += self.lambda_centroid * centroid_loss_val
        
        loss_dict['total_loss'] = total_loss
        return loss_dict


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 loss for distance transform regression (legacy compatibility)
    """
    
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
