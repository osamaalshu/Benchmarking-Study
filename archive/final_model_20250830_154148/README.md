# Error-Aware MAUNet: Final Model Implementation

This directory contains the complete implementation of the Error-Aware MAUNet model, designed to address systematic error patterns identified through comprehensive error analysis of cell segmentation models.

## 🎯 Key Improvements

### 1. Multi-Objective Loss Function

- **Focal Loss**: Addresses class imbalance by focusing on hard examples
- **Tversky Loss**: Asymmetric weighting (β > α) to reduce false negatives
- **Boundary Loss**: Uses signed distance transforms for precise boundary localization
- **Centroid Loss**: Auxiliary task for improved instance detection

### 2. Enhanced Architecture

- **Auxiliary Task Integration**: Dedicated heads for distance transform and centroid heatmap prediction
- **Anti-Ambiguity Regularization**: Embedding space regularization with learnable proxies
- **Dual Backbone Support**: ResNet50 and Wide-ResNet50 variants for ensemble

### 3. Advanced Post-Processing

- **Seeded Watershed Segmentation**: Uses centroid heatmaps as seeds and distance transforms for topology
- **Test-Time Augmentation**: Multiple transformations for improved robustness
- **Confidence-Based Filtering**: Removes low-confidence predictions

## 📁 File Structure

```
final_model/
├── error_aware_maunet.py          # Enhanced MAUNet model architecture
├── composite_losses.py            # Multi-objective loss functions
├── inference_pipeline.py          # Advanced inference with post-processing
├── train_error_aware_maunet.py    # Single model training script
├── train_dual_backbone.py         # Dual backbone training orchestrator
├── predict_error_aware_maunet.py  # Prediction and evaluation script
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Training Single Model

Train a single Error-Aware MAUNet model:

```bash
python train_error_aware_maunet.py \
    --data_path ./data/train-preprocessed/ \
    --work_dir ./final_model/work_dir \
    --backbone resnet50 \
    --enable_auxiliary_tasks \
    --lambda_focal 1.0 \
    --lambda_tversky 1.0 \
    --lambda_boundary 0.5 \
    --lambda_centroid 0.3 \
    --tversky_beta 0.7 \
    --max_epochs 2000
```

### 2. Training Dual Backbone Models

Train both ResNet50 and Wide-ResNet50 variants:

```bash
python train_dual_backbone.py \
    --data_path ./data/train-preprocessed/ \
    --work_dir ./final_model/work_dir \
    --train_backbones resnet50 wide_resnet50 \
    --max_epochs 2000
```

### 3. Creating Ensemble

After training both models, create the ensemble:

```bash
cd ./final_model/work_dir
python create_ensemble.py
```

### 4. Running Predictions

Single model prediction:

```bash
python predict_error_aware_maunet.py \
    --input_dir ./data/test/images \
    --output_dir ./predictions/error_aware_maunet_resnet50 \
    --model_path ./final_model/work_dir/error_aware_maunet_resnet50_3class/best_Dice_model.pth \
    --backbone resnet50 \
    --use_tta \
    --use_watershed
```

Ensemble prediction:

```bash
python predict_error_aware_maunet.py \
    --input_dir ./data/test/images \
    --output_dir ./predictions/error_aware_maunet_ensemble \
    --ensemble_path ./final_model/work_dir/error_aware_maunet_ensemble.pth \
    --use_tta \
    --use_watershed \
    --gt_dir ./data/test/labels
```

## 🔧 Model Configuration

### Loss Function Weights

- `lambda_focal`: Weight for focal loss (default: 1.0)
- `lambda_tversky`: Weight for Tversky loss (default: 1.0)
- `lambda_boundary`: Weight for boundary loss (default: 0.5)
- `lambda_centroid`: Weight for centroid loss (default: 0.3)
- `lambda_proxy`: Weight for proxy regularization (default: 0.1)

### Tversky Loss Parameters

- `tversky_alpha`: False positive weight (default: 0.3)
- `tversky_beta`: False negative weight (default: 0.7)
  - Setting β > α specifically targets false negative reduction

### Focal Loss Parameters

- `focal_alpha`: Weighting factor (default: 1.0)
- `focal_gamma`: Focusing parameter (default: 2.0)

### Watershed Post-Processing

- `min_seed_distance`: Minimum distance between centroid peaks (default: 10)
- `min_seed_prominence`: Minimum prominence for peak detection (default: 0.1)
- `watershed_threshold`: Threshold for foreground segmentation (default: 0.5)
- `min_object_size`: Minimum size for connected components (default: 16)

## 📊 Expected Performance Improvements

Based on error analysis findings, the Error-Aware MAUNet is designed to address:

1. **High False Negative Rates**: Tversky loss with β > α specifically reduces missed cells
2. **Cell Merging in Dense Regions**: Seeded watershed with centroid guidance improves separation
3. **Boundary Localization**: Distance transform supervision enhances boundary precision
4. **Instance Detection**: Auxiliary centroid task improves detection in low-contrast regions

## 🔬 Technical Details

### Architecture Enhancements

- **Triple Decoder Design**: Separate paths for segmentation, distance transform, and centroid prediction
- **Shared Encoder**: ResNet backbone with UNETR blocks for feature extraction
- **Embedding Regularization**: Anti-ambiguity loss using learnable proxies

### Loss Function Composition

The composite loss function is:

```
L = λ_focal * L_focal + λ_tversky * L_tversky + λ_boundary * L_boundary + λ_centroid * L_centroid + λ_proxy * L_proxy
```

### Post-Processing Pipeline

1. **Centroid Detection**: Peak detection in heatmap with minimum distance constraints
2. **Seed Refinement**: Filter seeds using segmentation confidence
3. **Watershed Application**: Use distance transform as topology guide
4. **Morphological Cleanup**: Remove small objects and fill holes

## 🧪 Experimental Validation

The model design is based on comprehensive error analysis that identified:

- **Recognition Quality**: Low due to missed cells (high false negatives)
- **Segmentation Quality**: Good boundary delineation once cells are detected
- **Dense Region Performance**: Systematic failures in crowded cellular arrangements

The Error-Aware MAUNet directly addresses these patterns through:

- **Detection-Focused Losses**: Focal and Tversky losses prioritize detection
- **Auxiliary Tasks**: Centroid prediction improves instance separation
- **Advanced Post-Processing**: Seeded watershed leverages auxiliary outputs

## 📈 Training Tips

1. **Loss Weight Tuning**: Start with provided defaults, adjust based on validation performance
2. **Learning Rate**: Use 6e-4 with exponential decay (γ=0.95, step=10)
3. **Batch Size**: 8 works well for 256x256 images on modern GPUs
4. **Early Stopping**: Use patience of 10 epochs with Dice metric
5. **Data Augmentation**: Comprehensive augmentation pipeline included

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or disable auxiliary tasks temporarily
2. **Slow Training**: Ensure proper GPU utilization, consider mixed precision
3. **Poor Convergence**: Check loss weights, ensure proper data preprocessing
4. **Watershed Failures**: Adjust seed detection parameters for your data characteristics

### Performance Optimization

- **Mixed Precision**: Add `--amp` flag for faster training (if implemented)
- **Multi-GPU**: Modify scripts for DataParallel/DistributedDataParallel
- **Preprocessing**: Ensure consistent data preprocessing between training and inference

## 📚 References

1. Lin, T. Y., et al. "Focal loss for dense object detection." ICCV 2017.
2. Salehi, S. S. M., et al. "Tversky loss function for image segmentation using 3D fully convolutional deep networks." MICCAI 2017.
3. Kervadec, H., et al. "Boundary loss for highly unbalanced segmentation." MICCAI 2019.
4. Original MAUNet: "Modality-Aware Anti-Ambiguity U-Net for Multi-Modality Cell Segmentation"

## 🤝 Contributing

This implementation is designed for research purposes. Key areas for enhancement:

- **Multi-Scale Training**: Implement multi-resolution training
- **Advanced Augmentation**: Add more sophisticated augmentation strategies
- **Uncertainty Estimation**: Add prediction uncertainty quantification
- **Real-Time Inference**: Optimize for deployment scenarios
