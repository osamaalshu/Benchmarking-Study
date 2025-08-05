# MAUNet (Modality-Aware Anti-Ambiguity U-Net) Usage Guide

This guide explains how to use the MAUNet model from the neurips22-cellseg_saltfish repository in your benchmarking study.

## Overview

MAUNet is a state-of-the-art cell segmentation model that won the NeurIPS 2022 Cell Segmentation Challenge. It features:

- Dual decoder paths for classification and distance transform regression
- ResNet backbone (ResNet50 or Wide-ResNet50)
- Anti-ambiguity design for handling multi-modality images

## Prerequisites

1. Ensure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

2. Make sure you have the following additional packages:

```bash
pip install einops
```

## Data Preprocessing

MAUNet requires special preprocessing including distance transform generation:

### 1. Basic Data Preprocessing

```bash
python utils/maunet_data_preprocess.py -i <path_to_original_data> -o <path_to_processed_data>
```

This will:

- Process images to the required format
- Handle multi-channel images
- Generate labels in the correct format

### 2. Distance Transform Preprocessing

```bash
python utils/maunet_dist_transform_preprocess.py \
    --data_path <path_to_processed_data> \
    --data_ori_path <path_to_original_data> \
    --save_path <path_to_distance_transform>
```

This creates distance transform weights used for the regression branch.

## Training MAUNet

### Basic Training

Train MAUNet with ResNet50 backbone:

```bash
python utils/train_maunet.py \
    --data_path <path_to_processed_data> \
    --weight_path <path_to_distance_transform> \
    --backbone resnet50 \
    --batch_size 8 \
    --max_epochs 500 \
    --input_size 256
```

Train MAUNet with Wide-ResNet50 backbone (better performance, more memory):

```bash
python utils/train_maunet.py \
    --data_path <path_to_processed_data> \
    --weight_path <path_to_distance_transform> \
    --backbone wide_resnet50 \
    --batch_size 4 \
    --max_epochs 500 \
    --input_size 256
```

### Fine-tuning from Pretrained Model

```bash
python utils/train_maunet.py \
    --data_path <path_to_processed_data> \
    --weight_path <path_to_distance_transform> \
    --backbone resnet50 \
    --model_path <path_to_pretrained_model> \
    --batch_size 8 \
    --max_epochs 200 \
    --lr 5e-5
```

### Training Parameters

- `--data_path`: Path to preprocessed data (must contain images/ and labels/ folders)
- `--weight_path`: Path to distance transform weights
- `--backbone`: Choose between "resnet50" or "wide_resnet50"
- `--batch_size`: Batch size (reduce if out of memory)
- `--max_epochs`: Number of training epochs
- `--input_size`: Input image size (default: 256)
- `--lr`: Learning rate (default: 1e-4)
- `--val_interval`: Validation frequency (default: every 5 epochs)

## Prediction/Inference

Run predictions using the trained MAUNet model:

```bash
python utils/predict.py \
    --model_name maunet \
    --model_path ./baseline/work_dir/maunet_3class/maunet_resnet50 \
    --input_path <path_to_test_images> \
    --output_path <path_to_predictions> \
    --num_class 3 \
    --input_size 256 \
    --show_overlay
```

## Model Evaluation

To include MAUNet in your benchmarking evaluation:

```bash
python run_evaluation.py
```

Make sure to update the `run_evaluation.py` to include MAUNet models:

- Add MAUNet ResNet50: `"maunet": "./baseline/work_dir/maunet_3class/maunet_resnet50"`
- Add MAUNet Wide-ResNet50: `"maunet_wide": "./baseline/work_dir/maunet_3class/maunet_wide_resnet50"`

## Tips and Best Practices

1. **Memory Management**:

   - Wide-ResNet50 requires more GPU memory (~12GB for batch size 4)
   - ResNet50 works well with 8GB GPUs (batch size 8)

2. **Data Format**:

   - Images should be RGB (3 channels)
   - Labels should be single-channel with values: 0 (background), 1 (cell), 2 (boundary)

3. **Training Strategy**:

   - Start with ResNet50 for faster experimentation
   - Use Wide-ResNet50 for best performance
   - Monitor both segmentation loss and distance transform loss

4. **Hyperparameters**:
   - The model uses combined loss: CrossEntropy + Dice + Weighted L1
   - Learning rate: 1e-4 works well for training from scratch
   - Use 5e-5 for fine-tuning

## Expected Performance

Based on the NeurIPS 2022 Cell Segmentation Challenge:

- ResNet50 backbone: F1 score ~0.816
- Wide-ResNet50 backbone: F1 score ~0.821
- Combined model: F1 score ~0.825

## Troubleshooting

1. **CUDA out of memory**: Reduce batch_size or use smaller input_size
2. **Poor performance**: Ensure distance transform is computed correctly
3. **Training instability**: Reduce learning rate or increase batch size
4. **Import errors**: Make sure to install einops: `pip install einops`

## Citation

If you use MAUNet in your research, please cite:

```
@article{maunet2022,
  title={MAUNet: Modality-Aware Anti-Ambiguity U-Net for Multi-Modality Cell Segmentation},
  author={Team saltfish},
  journal={NeurIPS 2022 Cell Segmentation Challenge},
  year={2022}
}
```

## Original Repository

For more details, visit: https://github.com/Woof6/neurips22-cellseg_saltfish
