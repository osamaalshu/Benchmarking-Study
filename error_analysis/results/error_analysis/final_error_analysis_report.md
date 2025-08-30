# Comprehensive Error Analysis Report

## Dataset Overview
- Total images analyzed: 100
- Models evaluated: unet, nnunet, sac, lstmunet, maunet_resnet50, maunet_wide, maunet_ensemble
- Minimum instance size: 10 pixels
- IoU threshold: 0.5

## Key Findings

### Overall Performance
- **Best F1-Score**: maunet_wide (0.529)
- **Best Panoptic Quality**: maunet_ensemble (0.437)
- **Average GT cells per image**: 454.4

### Error Analysis Summary

| Model | F1-Score | PQ | Precision | Recall | Avg FN | Avg FP | Avg Splits | Avg Merges |
|-------|----------|----|-----------|---------|---------|---------|-----------|-----------|
| lstmunet | 0.282 | 0.199 | 0.263 | 0.305 | 315.8 | 389.0 | 1.7 | 41.8 |
| maunet_ensemble | 0.499 | 0.437 | 0.632 | 0.413 | 266.9 | 109.0 | 3.4 | 28.3 |
| maunet_resnet50 | 0.507 | 0.399 | 0.598 | 0.440 | 254.7 | 134.6 | 0.9 | 36.9 |
| maunet_wide | 0.529 | 0.413 | 0.605 | 0.470 | 241.0 | 139.1 | 1.3 | 35.2 |
| nnunet | 0.357 | 0.278 | 0.367 | 0.348 | 296.4 | 272.7 | 3.6 | 47.3 |
| sac | 0.003 | 0.002 | 0.005 | 0.002 | 453.5 | 179.0 | 0.1 | 8.2 |
| unet | 0.315 | 0.239 | 0.335 | 0.297 | 319.6 | 268.0 | 3.1 | 53.1 |

### Detailed Metrics Explanation
- **False Negatives (FN)**: Ground truth cells that were missed by the model
- **False Positives (FP)**: Artifacts wrongly segmented as cells
- **Splits**: Ground truth cells that were over-segmented into multiple predictions
- **Merges**: Multiple ground truth cells that were under-segmented into one prediction
- **PQ (Panoptic Quality)**: Overall segmentation quality combining recognition and segmentation
- **RQ (Recognition Quality)**: How well the model detects instances
- **SQ (Segmentation Quality)**: How accurately detected instances are segmented

### Model Rankings
Based on F1-Score:
1. maunet_wide: 0.529
2. maunet_resnet50: 0.507
3. maunet_ensemble: 0.499
4. nnunet: 0.357
5. unet: 0.315
6. lstmunet: 0.282
7. sac: 0.003

Based on Panoptic Quality:
1. maunet_ensemble: 0.437
2. maunet_wide: 0.413
3. maunet_resnet50: 0.399
4. nnunet: 0.278
5. unet: 0.239
6. lstmunet: 0.199
7. sac: 0.002
