# NeurIPS Cell Segmentation Benchmarking Study

This project benchmarks 5 different state-of-the-art segmentation models on cell microscopy images for the NeurIPS Cell Segmentation Challenge. The study includes comprehensive evaluation metrics, performance comparisons across multiple architectures, error analysis, and synthetic data augmentation studies.

## 📁 Project Structure

```
├── data/                      # Dataset storage
│   ├── train/                 # Original training data (1000 images)
│   ├── train-preprocessed/    # Preprocessed training data (900 images)
│   ├── val/                   # Validation data (100 images)
│   └── test/                  # Test data (101 images)
│
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── sac_model.py          # SAC (Segment Any Cell) model using Meta's SAM
│   ├── nnunet.py             # nnU-Net self-configuring model
│   ├── lstmunet.py           # LSTM-enhanced U-Net model
│   ├── maunet.py             # MAUNet (Modality-Aware Anti-Ambiguity U-Net) with dual decoders
│   ├── unet.py               # Standard U-Net implementation
│   └── sam_weights/          # SAM model weights directory
│
├── utils/                     # Utility scripts and tools
│   ├── pre_process.py         # Data preprocessing script
│   ├── split_data.py          # Train/validation split script
│   ├── model_training_3class.py # Main training script
│   ├── predict.py             # Inference script
│   ├── compute_metric.py      # Evaluation metrics script
│   ├── evaluate_maunet.py     # MAUNet-specific evaluation
│   ├── generate_results_report.py # Results report generation
│   └── [additional utility scripts]
│
├── baseline/                  # Training outputs and checkpoints
│   └── work_dir/             # Model training directories
│
├── test_predictions/          # Model predictions and evaluation results
│   ├── benchmarking_report.* # Comprehensive benchmarking reports (MD, PDF, TXT)
│   ├── *_metrics-*.csv       # Per-model performance metrics
│   ├── lstmunet/             # LSTM-UNet predictions
│   ├── maunet_ensemble/      # MAUNet ensemble predictions
│   ├── maunet_resnet50/      # MAUNet with ResNet50 backbone
│   ├── maunet_wide/          # MAUNet wide variant
│   ├── nnunet/               # nnU-Net predictions
│   ├── sac/                  # SAC model predictions
│   └── unet/                 # UNet predictions
│
├── visualization_results/     # Analysis and comparison visualizations
│   ├── models_summary.csv    # Model performance summary
│   ├── training_summary.csv  # Training statistics
│   ├── lstmunet/             # LSTM-UNet visualizations
│   ├── maunet_ensemble/      # MAUNet ensemble visualizations
│   ├── maunet_resnet50/      # MAUNet ResNet50 visualizations
│   ├── maunet_wide/          # MAUNet wide visualizations
│   ├── nnunet/               # nnU-Net visualizations
│   ├── sac/                  # SAC model visualizations
│   └── unet/                 # UNet visualizations
│
├── error_analysis/           # Comprehensive error analysis framework
│   ├── config/
│   │   └── analysis_config.py # Error analysis configuration
│   ├── src/
│   │   ├── __init__.py
│   │   ├── calibration_analyzer.py # Model calibration analysis
│   │   ├── data_loader.py    # Data loading utilities
│   │   └── [additional analysis modules]
│   ├── scripts/
│   │   ├── run_complete_analysis.py # Complete analysis pipeline
│   │   ├── run_complete_test_set_analysis.py # Test set analysis
│   │   └── run_error_analysis_only.py # Error analysis only
│   ├── results/
│   │   ├── calibration_analysis/ # Calibration results
│   │   ├── error_categorization/ # Error categorization results
│   │   ├── reports/          # Analysis reports (MD, PDF)
│   │   └── visual_inspection/ # Visual inspection results
│   ├── logs/                 # Analysis logs
│   ├── requirements.txt      # Error analysis dependencies
│   ├── README.md            # Error analysis documentation
│   └── GETTING_STARTED.md   # Quick start guide
│
├── synthesis_augmentation_study/ # Synthetic data augmentation research
│   ├── __init__.py
│   ├── datasets/
│   │   └── neurips_masks2imgs/ # Dataset for synthesis
│   ├── external/             # External synthesis tools (CycleGAN, pix2pix)
│   ├── final_augmentation_results/ # Comprehensive augmentation study results
│   │   ├── evaluation_results/ # Evaluation outcomes
│   │   ├── training_results/  # Training results by model and configuration
│   │   ├── study_config.json # Study configuration
│   │   └── study_state.json  # Study state tracking
│   ├── fixed_dataset_arms/   # Dataset arms for different augmentation strategies
│   │   ├── R/               # Real data only
│   │   ├── S/               # Synthetic data only
│   │   ├── R+S@10/          # Real + 10% synthetic
│   │   ├── R+S@25/          # Real + 25% synthetic
│   │   └── R+S@50/          # Real + 50% synthetic
│   ├── synthetic_data_500/   # Generated synthetic dataset
│   ├── utils/               # Augmentation study utilities
│   └── experiment_runner.py # Main experiment execution script
│
├── thesis_visuals/          # Thesis visualization materials
│   ├── config/              # Visualization configuration
│   ├── data/                # Data for visualizations
│   ├── notebooks/           # Jupyter notebooks for analysis
│   │   └── tiny_cnn_analysis.ipynb
│   ├── tiny_cnn_experiment/ # Tiny CNN experiment results
│   ├── unet_experiment/     # UNet experiment results
│   └── utils/               # Visualization utilities
│
├── archive/                 # Archived files and previous versions
│   ├── old_files/           # Previous script versions
│   └── unused_directories/  # Unused synthesis and augmentation code
│
├── notebooks/               # Jupyter notebooks for analysis
│   └── Data_Analysis.ipynb  # Dataset exploration and analysis
│
├── requirements.txt         # Project dependencies
├── visualize_results.py     # Main visualization script
└── README.md               # This file
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
python utils/generate_results_report.py

# Visualize results and comparisons
python visualize_results.py
```

### 7. Error Analysis

```bash
# Run complete error analysis pipeline
cd error_analysis
python scripts/run_complete_analysis.py

# Run analysis on complete test set
python scripts/run_complete_test_set_analysis.py

# Generate error analysis reports
python scripts/run_error_analysis_only.py
```

### 8. Synthetic Data Augmentation Study

```bash
# Run augmentation study experiments
cd synthesis_augmentation_study
python experiment_runner.py

# View augmentation study results
ls final_augmentation_results/
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

## Error Analysis Framework

The project includes a comprehensive error analysis framework (`error_analysis/`) that provides:

- **Model Calibration Analysis**: Evaluates prediction confidence calibration
- **Error Categorization**: Classifies different types of segmentation errors
- **Visual Inspection Tools**: Automated and manual error inspection
- **Performance Reports**: Detailed analysis reports in multiple formats
- **Challenging Image Identification**: Identifies images that are difficult for all models

### Error Analysis Features

- **Complete Test Set Analysis**: Comprehensive evaluation of all test images
- **Calibration Metrics**: Reliability diagrams and calibration statistics
- **Error Classification**: Systematic categorization of segmentation failures
- **Visual Reports**: Automated generation of error visualization reports

## Synthetic Data Augmentation Study

The `synthesis_augmentation_study/` directory contains a comprehensive study on synthetic data augmentation:

### Study Design

- **Dataset Arms**: Multiple augmentation strategies (R, S, R+S@10, R+S@25, R+S@50)
- **Multiple Seeds**: Experiments run with different random seeds for robustness
- **Comprehensive Evaluation**: Detailed performance analysis across all configurations

### Augmentation Strategies

- **R**: Real data only (baseline)
- **S**: Synthetic data only
- **R+S@10**: Real data + 10% synthetic augmentation
- **R+S@25**: Real data + 25% synthetic augmentation
- **R+S@50**: Real data + 50% synthetic augmentation

### Results

- **Training Results**: Detailed training metrics for each configuration
- **Evaluation Results**: Comprehensive performance evaluation
- **Statistical Analysis**: Robust statistical comparisons across strategies

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
- Error analysis provides insights into model failure modes and calibration
- Synthetic data augmentation study explores data augmentation strategies

### Current Model Status

- **Active Models**: UNet, SAC, nnU-Net, LSTM-UNet, MAUNet (5 models)
- **MAUNet Features**: Dual-decoder architecture with classification and distance transform regression
- **Error Analysis**: Comprehensive framework for model evaluation and debugging
- **Augmentation Study**: Systematic investigation of synthetic data augmentation strategies


## Recent Updates

- **Error Analysis Framework**: Added comprehensive error analysis capabilities
- **Synthetic Data Augmentation Study**: Implemented systematic augmentation research
- **Enhanced Visualization**: Improved result visualization and reporting
- **Thesis Materials**: Added thesis visualization and analysis tools
- **Archived Code**: Organized previous versions and unused code in archive directory
