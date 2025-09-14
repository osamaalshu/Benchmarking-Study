# Cell Segmentation Benchmarking Study

A comprehensive benchmarking study comparing state-of-the-art deep learning models for microscopy cell segmentation, including synthetic data augmentation and error analysis.

## Project Overview

This project evaluates 5 different segmentation models on cell microscopy images:

- **UNet**: Standard U-Net baseline
- **SAC**: Segment Any Cell using Meta's SAM foundation model
- **nnU-Net**: Self-configuring U-Net with automatic optimization
- **LSTM-UNet**: LSTM-enhanced U-Net for improved feature learning
- **MAUNet**: Modality-Aware Anti-Ambiguity U-Net with dual decoders

Through comprehensive benchmarking, error analysis, and synthetic data augmentation studies, this research identifies key limitations in existing approaches and leads to the development of an **Error-Aware and Data-Augmented MAUNet**. This enhanced architecture incorporates:

- **Composite Loss Functions**: Advanced loss combining focal, tversky, boundary, distance transform, and centroid losses
- **Error-Aware Design**: Architecture improvements based on systematic error analysis
- **Synthetic Data Integration**: Seamless augmentation with Pix2Pix-generated synthetic images
- **Dual Backbone Support**: Both ResNet50 and Wide ResNet50 configurations for optimal performance

## Directory Structure

```
├── data/                     # Dataset (train/val/test splits)
├── models/                   # Model implementations
├── utils/                    # Training, inference, and evaluation scripts
├── notebooks/                # Training and analysis notebooks
├── error_analysis/           # Error analysis framework
├── synthesis_augmentation_study/  # Synthetic data augmentation research
├── test_predictions/         # Benchmarking results (CSV files)
└── requirements.txt          # Project dependencies
```

### Core Components

#### `models/`

Model architecture implementations:

- `maunet_error_aware.py` - Error-aware MAUNet with composite loss
- `maunet.py` - Standard MAUNet implementation
- `sac_model.py` - Segment Any Cell using Meta's SAM
- `nnunet.py` - Self-configuring nnU-Net
- `lstmunet.py` - LSTM-enhanced U-Net

#### `utils/`

Core training and evaluation utilities:

- `model_training_3class.py` - Main training script (supports `--synthetic` flag)
- `predict.py` - Model inference script
- `compute_metric.py` - Evaluation metrics calculation
- `pre_process.py` - Data preprocessing pipeline

#### `notebooks/`

Ready-to-run Colab notebooks for training and analysis:

- `Data_Analysis.ipynb` - Dataset exploration
- `SAC.ipynb` - Segment Any Cell training
- `nnUnet_Benchmarking.ipynb` - nnU-Net training
- `MAUNET_Benchmarking_Training.ipynb` - Standard MAUNet training
- `Final Model ErrorAwareMAUNET *.ipynb` - Error-Aware MAUNet variants
- `Synthetic_*.ipynb` - Models with synthetic data augmentation

#### `error_analysis/`

Comprehensive error analysis framework:

- `scripts/` - Analysis pipeline scripts
- `src/` - Core analysis modules
- `config/` - Analysis configuration

#### `synthesis_augmentation_study/`

Synthetic data augmentation research:

- `external/` - Pix2Pix synthesis tools
- `utils/` - Augmentation utilities
- `fixed_dataset_arms/` - Dataset configurations

## 🚀 Quick Start

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Data Preprocessing

```bash
# Preprocess training data (converts to 3-class labels)
python utils/pre_process.py -i ./data/train -o ./data/train-preprocessed

# Split into train/validation (90/10 split)
python utils/split_data.py --data_path ./data/train-preprocessed --val_frac 0.1
```

### 3. Train Models

```bash
# Standard training
python utils/model_training_3class.py --model_name maunet_error_aware --backbone resnet50

# With synthetic data augmentation (adds synthetic images)
python utils/model_training_3class.py --model_name maunet_error_aware --backbone resnet50 --synthetic
```

### 4. Evaluate Models

```bash
# Run inference
python utils/predict.py -i ./data/test/images -o ./results --model_path ./baseline/work_dir/model_3class

# Compute metrics
python utils/compute_metric.py -g ./data/test/labels -s ./results
```

## Key Features

### Model Architectures

- **MAUNet Error-Aware**: Advanced dual-decoder architecture with composite loss
- **Synthetic Data Integration**: Automatic integration of synthetic images for data augmentation
- **Multiple Backbones**: ResNet50 and Wide ResNet50 support

### Training Options

| Parameter      | Description           | Example                      |
| -------------- | --------------------- | ---------------------------- |
| `--model_name` | Model type            | `maunet_error_aware`         |
| `--backbone`   | Backbone architecture | `resnet50`, `wide_resnet50`  |
| `--synthetic`  | Add synthetic data    | Flag to add synthetic images |
| `--batch_size` | Batch size            | `6`                          |
| `--max_epochs` | Max training epochs   | `200`                        |

### Hardware Support

- **Apple Silicon (MPS)**: Optimized for M1/M2/M3 Macs
- **NVIDIA CUDA**: GPU acceleration support
- **CPU**: Fallback option

## Dataset

- **Training**: 900 images (+ synthetic with `--synthetic` flag)
- **Validation**: 100 images
- **Test**: 101 images
- **Classes**: 3-class segmentation (background, interior, boundary)

## Key Innovations

- **Error-Aware Architecture**: Advanced MAUNet with composite loss functions
- **Synthetic Data Integration**: Seamless augmentation with Pix2Pix generated data
- **Comprehensive Evaluation**: Multi-threshold metrics and statistical analysis

## Usage Notes

- Models automatically save best checkpoints based on validation Dice score
- TensorBoard logging for training monitoring
- Early stopping with configurable tolerance
- Comprehensive error analysis and visualization tools

---

_Note: Large files (images, model weights, results) are excluded from the repository. Use the provided scripts to generate these locally._
