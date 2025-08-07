# NeurIPS Cell Segmentation Benchmarking Study

This project benchmarks 5 different state-of-the-art segmentation models on cell microscopy images for the NeurIPS Cell Segmentation Challenge. The study includes comprehensive evaluation metrics and performance comparisons across multiple architectures.

## 📁 Project Structure

```
├── data/
│   ├── train/                 # Original training data (1000 images)
│   ├── train-preprocessed/    # Preprocessed training data (900 images)
│   ├── val/                   # Validation data (100 images)
│   └── test/                  # Test data (101 images)
|
├── utils/
│   ├── pre_process.py         # Data preprocessing script
│   ├── split_data.py          # Train/validation split script
│   ├── model_training_3class.py # Main training script
│   ├── predict.py             # Inference script
│   └── compute_metric.py      # Evaluation metrics script
|
├── models/
│   ├── sac_model.py          # SAC (Segment Any Cell) model using Meta's SAM
│   ├── nnunet.py             # nnU-Net self-configuring model
│   ├── lstmunet.py           # LSTM-enhanced U-Net model
│   └── maunet.py             # MAUNet (Modality-Aware Anti-Ambiguity U-Net) with dual decoders
|
├── test_predictions/         # Model predictions and evaluation metrics
│   ├── benchmarking_report.* # Comprehensive benchmarking reports
│   └── *_metrics-*.csv       # Per-model performance metrics
|
├── visualization_results/    # Analysis and comparison visualizations
│   ├── models_summary.csv    # Model performance summary
│   └── training_summary.csv  # Training statistics
|
└── notebooks/
    ├── Data_Analysis.ipynb   # Dataset exploration
    └── Visualizations.ipynb  # Results analysis and plotting
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify MPS (Apple Silicon GPU) is available
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### 2. Data Preprocessing

```bash
# Preprocess training data (converts to 3-class labels)
python utils/pre_process.py -i ./data/train -o ./data/train-preprocessed

# Split data into train/validation sets (10% validation)
python utils/split_data.py --data_path ./data/train-preprocessed --val_frac 0.1
```

### 3. Model Training

```bash
# Train UNet model (30 epochs, batch size 8, MPS acceleration)
python utils/model_training_3class.py --model_name unet --batch_size 8 --max_epochs 30

# Train SAC model (Segment Any Cell using Meta's SAM)
python utils/model_training_3class.py --model_name sac --batch_size 2 --max_epochs 30

# Train nnU-Net model (self-configuring U-Net)
python utils/model_training_3class.py --model_name nnunet --batch_size 4 --max_epochs 30

# Train LSTM-UNet model
python utils/model_training_3class.py --model_name lstmunet --batch_size 6 --max_epochs 30

# Train MAUNet model (Modality-Aware Anti-Ambiguity U-Net with dual decoders)
python utils/model_training_3class.py --model_name maunet --batch_size 4 --max_epochs 30
```

### 4. Monitor Training

```bash
# Start TensorBoard (access at http://localhost:6006)
tensorboard --logdir=baseline/work_dir/unet_3class --port=6006 --host=0.0.0.0

# Check training progress
ls -la baseline/work_dir/unet_3class/
python -c "import numpy as np; data = np.load('baseline/work_dir/unet_3class/train_log.npz'); print('Epochs:', len(data['epoch_loss'])); print('Losses:', data['epoch_loss'])"
```

### 5. Model Inference

```bash
# Run inference on test data
python utils/predict.py -i ./data/test/images -o ./results --model_path ./baseline/work_dir/unet_3class
```

### 6. Evaluate Results

```bash
# Compute evaluation metrics for individual models
python utils/compute_metric.py -g ./data/test/labels -s ./results --gt_suffix .tiff --seg_suffix _label.png

# Run comprehensive evaluation across all models
python run_evaluation.py

# Generate detailed benchmarking report
python generate_results_report.py

# Visualize results and comparisons
python visualize_results.py
```

## Training Options

| Parameter        | Default | Description                                               |
| ---------------- | ------- | --------------------------------------------------------- |
| `--model_name`   | `unet`  | Model type: `unet`, `sac`, `nnunet`, `lstmunet`, `maunet` |
| `--batch_size`   | `8`     | Batch size per GPU                                        |
| `--max_epochs`   | `2000`  | Maximum training epochs                                   |
| `--initial_lr`   | `6e-4`  | Learning rate                                             |
| `--val_interval` | `2`     | Validation interval                                       |
| `--input_size`   | `256`   | Input image size                                          |
| `--num_class`    | `3`     | Number of classes (background, interior, boundary)        |

## Dataset Information

- **Training**: 900 images (after validation split)
- **Validation**: 100 images (10% split)
- **Test**: 101 images
- **Classes**: 3-class segmentation (background, interior, boundary)
- **Image Format**: Preprocessed to PNG with normalized intensities

## Technical Details

### Hardware Acceleration

- **MPS**: Apple Silicon GPU acceleration (automatically detected)
- **CUDA**: NVIDIA GPU support (if available)
- **CPU**: Fallback option

### Model Architectures

1. **UNet**: Standard U-Net with residual connections and skip connections
2. **SAC**: Segment Any Cell model using Meta's SAM with CAM architecture - uses SAM's image encoder as feature extractor with custom decoder head
3. **nnU-Net**: Self-configuring U-Net that automatically adapts to dataset characteristics
4. **LSTM-UNet**: LSTM-enhanced U-Net with temporal modeling capabilities for improved feature learning
5. **MAUNet**: Modality-Aware Anti-Ambiguity U-Net with dual decoders for classification and distance transform regression

### Data Preprocessing

- **Normalization**: Percentile-based intensity normalization (1-99%)
- **Labels**: Instance labels converted to 3-class (background/interior/boundary)
- **Augmentation**: Random crops, flips, rotations, noise, contrast adjustments

## Benchmarking Results

The project includes comprehensive benchmarking results for all 5 models:

### Performance Metrics (Available in `test_predictions/`)

- **Dice Score**: Overlap-based similarity metric
- **IoU (Jaccard Index)**: Intersection over Union
- **F1 Score**: Harmonic mean of precision and recall
- **Precision & Recall**: Per-class performance metrics

### Model Comparison

- **UNet**: Fast training, good baseline performance with proven architecture
- **SAC**: Leverages pre-trained SAM features for robust segmentation with foundation model capabilities
- **nnU-Net**: Self-optimizing architecture with automatic hyperparameter tuning and adaptive preprocessing
- **LSTM-UNet**: Enhanced temporal modeling for improved boundary detection and feature learning
- **MAUNet**: Dual-decoder architecture with ResNet backbone, combining classification and distance transform regression for enhanced boundary detection

### Training Performance

- **Training Time**: Varies by model complexity (15-45 minutes for 30 epochs on MPS)
- **Memory Usage**: 2-8GB depending on model and batch size
- **Convergence**: Most models converge within 20-50 epochs

## Troubleshooting

### Common Issues

1. **MPS not available**: Ensure PyTorch 2.0+ is installed
2. **Memory issues**: Reduce batch size (try 2 or 4)
3. **TensorBoard errors**: Install compatible protobuf version: `pip install protobuf==3.20.3`

### Performance Tips

- Use MPS acceleration for faster training on Mac
- Adjust batch size based on available memory
- Monitor GPU usage with Activity Monitor

## Notes

- Training automatically saves best model based on validation Dice score
- Early stopping implemented with configurable epoch tolerance (default: 10)
- All training logs saved to TensorBoard format
- Model checkpoints saved as `best_Dice_model.pth` and `final_model.pth`
- Comprehensive evaluation metrics computed at multiple IoU thresholds (0.5, 0.7, 0.9)
- Results include detailed performance analysis and model comparisons

### Current Model Status

- **Active Models**: UNet, SAC, nnU-Net, LSTM-UNet, MAUNet (5 models)
- **MAUNet Features**: Dual-decoder architecture with classification and distance transform regression

## Contributing

This project is part of the NeurIPS Cell Segmentation Challenge. For questions or issues, please refer to the challenge documentation.
