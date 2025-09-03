# Training MAUNet Error-Aware Models with Synthetic Data

## Updated Colab Script

Here's your updated Colab script that includes training both backbones with synthetic data:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/osamaalshu/Benchmarking-Study.git

%cd Benchmarking-Study/

!pip install -r requirements.txt

!ln -s /content/drive/MyDrive/data /content/Benchmarking-Study/data

!ls

!ls -l /content/Benchmarking-Study/data

!ls /content/Benchmarking-Study/data/train/images | head

# === Install Segment Anything and download SAM weights ===

# Clone Segment Anything repository (SAM)
!git clone https://github.com/facebookresearch/segment-anything.git

# Install the package in editable mode
%cd segment-anything
!pip install -e .

# Go back to your benchmarking repo
%cd /content/Benchmarking-Study

# Create the folder for SAM weights
!mkdir -p models/sam_weights

# Download the ViT-B SAM checkpoint (used by default in SAC)
!wget -P models/sam_weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Confirm the file exists
!ls -lh models/sam_weights

!git pull

# === Training MAUNet Error-Aware ResNet50 with Synthetic Data ===
!nohup python utils/model_training_3class.py \
  --model_name maunet_error_aware \
  --backbone resnet50 \
  --num_class 3 \
  --input_size 768 \
  --batch_size 6 \
  --max_epochs 200 \
  --epoch_tolerance 25 \
  --initial_lr 6e-4 \
  --lr_scheduler \
  --lr_step_size 10 \
  --lr_gamma 0.95 \
  --data_path ./data/train-preprocessed \
  --work_dir ./baseline/work_dir \
  --num_workers 8 \
  --synthetic \
  --synthetic_data_path ./data/synthetic_data_500 \
  > train_log_maunet_error_aware_resnet50_synthetic.txt 2>&1 &

# Monitor the training log
!tail -f train_log_maunet_error_aware_resnet50_synthetic.txt

# Wait for first training to complete or manually stop when needed
# Then train the Wide ResNet50 backbone

# === Training MAUNet Error-Aware Wide ResNet50 with Synthetic Data ===
!nohup python utils/model_training_3class.py \
  --model_name maunet_error_aware \
  --backbone wide_resnet50 \
  --num_class 3 \
  --input_size 768 \
  --batch_size 6 \
  --max_epochs 200 \
  --epoch_tolerance 25 \
  --initial_lr 6e-4 \
  --lr_scheduler \
  --lr_step_size 10 \
  --lr_gamma 0.95 \
  --data_path ./data/train-preprocessed \
  --work_dir ./baseline/work_dir \
  --num_workers 8 \
  --synthetic \
  --synthetic_data_path ./data/synthetic_data_500 \
  > train_log_maunet_error_aware_wideresnet50_synthetic.txt 2>&1 &

# Monitor the training log
!tail -f train_log_maunet_error_aware_wideresnet50_synthetic.txt
```

## Key Changes Made:

1. **Added `--synthetic` flag**: This enables the integration of 10% synthetic data (50 images) into the training set
2. **Added `--synthetic_data_path`**: Points to the synthetic data directory
3. **Updated model naming**: Models will be saved with `_synthetic` suffix to distinguish them
4. **Reproducible selection**: Uses the same seed to ensure consistent synthetic data selection

## Expected Output Directories:

The trained models will be saved in:

- `./baseline/work_dir/maunet_error_aware_3class_synthetic/` (ResNet50 backbone)
- `./baseline/work_dir/maunet_error_aware_3class_synthetic/` (Wide ResNet50 backbone)

## Training Data Composition:

- **Real training data**: 1000 images
- **Synthetic data added**: 50 images (10% of 500 available)
- **Total training data**: 1050 images per backbone
- **Validation data**: Unchanged (only real data)

## Model Architecture:

Both models will use the MAUNet Error-Aware architecture with:

- Composite loss (detection + segmentation + boundary + distance transform + centroid)
- Advanced loss functions (Focal + Tversky + Custom Boundary)
- Input size: 768x768
- 3 classes (background, interior, boundary)

The synthetic data integration happens automatically when the `--synthetic` flag is used, selecting 50 random synthetic images from the 500 available ones using a fixed seed for reproducibility.
