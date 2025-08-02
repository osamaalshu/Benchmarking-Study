# NeurIPS Cell Segmentation Benchmarking Study

This project benchmarks different segmentation models on cell microscopy images for the NeurIPS Cell Segmentation Challenge.

## 📁 Project Structure

```
├── data/
│   ├── train/                 # Original training data (1000 images)
│   ├── train-preprocessed/    # Preprocessed training data (900 images)
│   ├── val/                   # Validation data (100 images)
│   └── test/                  # Test data (101 images)
├── utils/
│   ├── pre_process.py         # Data preprocessing script
│   ├── split_data.py          # Train/validation split script
│   ├── model_training_3class.py # Main training script
│   ├── predict.py             # Inference script
│   └── compute_metric.py      # Evaluation metrics script
├── models/
│   └── unetr2d.py            # UNETR model architecture
├── baseline/
│   └── work_dir/             # Training outputs and checkpoints
└── notebooks/
    ├── Data_Analysis.ipynb   # Dataset exploration
    └── Benchmarking.ipynb    # Results analysis
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

# Train UNETR model
python utils/model_training_3class.py --model_name unetr --batch_size 4 --max_epochs 30

# Train SwinUNETR model
python utils/model_training_3class.py --model_name swinunetr --batch_size 2 --max_epochs 30
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
# Compute evaluation metrics
python utils/compute_metric.py -g ./data/test/labels -s ./results --gt_suffix .tiff --seg_suffix _label.png
```

## ⚙️ Training Options

| Parameter        | Default | Description                                        |
| ---------------- | ------- | -------------------------------------------------- |
| `--model_name`   | `unet`  | Model type: `unet`, `unetr`, `swinunetr`           |
| `--batch_size`   | `8`     | Batch size per GPU                                 |
| `--max_epochs`   | `2000`  | Maximum training epochs                            |
| `--initial_lr`   | `6e-4`  | Learning rate                                      |
| `--val_interval` | `2`     | Validation interval                                |
| `--input_size`   | `256`   | Input image size                                   |
| `--num_class`    | `3`     | Number of classes (background, interior, boundary) |

## 📊 Dataset Information

- **Training**: 900 images (after validation split)
- **Validation**: 100 images (10% split)
- **Test**: 101 images
- **Classes**: 3-class segmentation (background, interior, boundary)
- **Image Format**: Preprocessed to PNG with normalized intensities

## 🔧 Technical Details

### Hardware Acceleration

- **MPS**: Apple Silicon GPU acceleration (automatically detected)
- **CUDA**: NVIDIA GPU support (if available)
- **CPU**: Fallback option

### Model Architectures

1. **UNet**: Standard U-Net with residual connections
2. **UNETR**: Vision Transformer-based U-Net
3. **SwinUNETR**: Swin Transformer-based U-Net

### Data Preprocessing

- **Normalization**: Percentile-based intensity normalization (1-99%)
- **Labels**: Instance labels converted to 3-class (background/interior/boundary)
- **Augmentation**: Random crops, flips, rotations, noise, contrast adjustments

## 📈 Expected Results

Based on previous training runs:

- **Loss Reduction**: ~35% (1.368 → 0.891 over 9 epochs)
- **Training Time**: ~15-20 minutes for 30 epochs with MPS
- **Model Size**: ~19.6MB for trained checkpoints

## 🐛 Troubleshooting

### Common Issues

1. **MPS not available**: Ensure PyTorch 2.0+ is installed
2. **Memory issues**: Reduce batch size (try 2 or 4)
3. **TensorBoard errors**: Install compatible protobuf version: `pip install protobuf==3.20.3`

### Performance Tips

- Use MPS acceleration for faster training on Mac
- Adjust batch size based on available memory
- Monitor GPU usage with Activity Monitor

## 📝 Notes

- Training automatically saves best model based on validation Dice score
- Early stopping implemented with 100 epoch tolerance
- All training logs saved to TensorBoard format
- Model checkpoints saved as `best_Dice_model.pth` and `final_model.pth`

## 🤝 Contributing

This project is part of the NeurIPS Cell Segmentation Challenge. For questions or issues, please refer to the challenge documentation.
