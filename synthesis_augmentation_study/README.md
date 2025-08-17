# Modality Agnostic Controlled Augmentation Study

## Overview

This study evaluates the effectiveness of synthetic data augmentation for microscopy image segmentation using a controlled experimental design. The research demonstrates that **adding 10% synthetic data improves segmentation performance by 44.5%**.

## Key Results

**Optimal Strategy: R+S@10 (Real + 10% Synthetic)**

- **Dice Score**: 0.137 → **0.198** (+44.5% improvement)
- **IoU**: 0.112 → **0.178** (+58.2% improvement)
- **Precision**: 0.257 → **0.382** (+48.4% improvement)
- **All improvements statistically significant** (p < 0.01, Bonferroni-corrected)

## Experimental Design

### Dataset Arms

- **R**: Real-only baseline (900 images)
- **R+S@10**: Real + 10% synthetic (900 + 90 images)
- **R+S@25**: Real + 25% synthetic (900 + 225 images)
- **R+S@50**: Real + 50% synthetic (900 + 450 images)
- **S**: Synthetic-only (450 images)

### Methodology

- **Model**: nnU-Net (best baseline)
- **Training**: 5 epochs per run, 3 seeds (0,1,2)
- **Evaluation**: Real test set, paired statistical analysis
- **Metrics**: Dice, IoU, Precision, Recall, Boundary F1, HD95

## Project Structure

```
synthesis_augmentation_study/
├── README.md                           # This file
├── run_study.py                        # Main entry point
├── experiment_runner.py                # Study orchestration
├── requirements.txt                    # Dependencies
├── utils/                              # Utility modules
│   ├── data_arms_manager.py           # Dataset arm creation
│   ├── training_protocol.py           # Model training
│   ├── evaluation_framework.py        # Evaluation & statistics
│   ├── model_wrappers.py              # Model abstractions
│   └── [other utilities...]
├── external/                          # Github Repo https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
├── fixed_dataset_arms/                # Pre-created dataset arms
├── synthetic_data_500/                # Generated synthetic data
└── final_augmentation_results/        # Complete study results
```

## Quick Start

```bash
# Setup environment
conda create -n synthesis_aug python=3.10
conda activate synthesis_aug
pip install -r synthesis_augmentation_study/requirements.txt

# Run the complete study
cd synthesis_augmentation_study
python run_study.py
```

## Synthetic Data Generation

### Pix2Pix Training Pipeline

The synthetic data used in this study was generated using a Pix2Pix conditional GAN trained on real image-mask pairs.

```bash
# Setup pix2pix framework
python utils/setup_external.py --clone

# Prepare training dataset
python utils/prepare_pix2pix_dataset.py \
  --train_path ../data/train-preprocessed \
  --val_path ../data/val \
  --dataset_name high_quality_pix2pix \
  --tile_size 512 \
  --stride 256

# Train pix2pix model
python utils/train_pix2pix.py \
  --dataroot ./datasets/high_quality_pix2pix \
  --name high_quality_synthesis \
  --direction AtoB \
  --batch_size 8 \
  --netG unet_256 \
  --ngf 64 --ndf 64 \
  --n_epochs 40 --n_epochs_decay 40 \
  --lr 0.0002 --beta1 0.5 \
  --gan_mode lsgan \
  --lambda_L1 100 \
  --load_size 512 --crop_size 512 \
  --no_flip

# Generate synthetic data
python utils/generate_500_final.py \
  --model_name high_quality_synthesis \
  --output_dir synthetic_data_500 \
  --num_samples 500
```

### Training Configuration

- **Architecture**: U-Net generator (unet_256) with basic discriminator
- **Resolution**: 512×512 images and masks
- **Training**: 40 epochs + 40 decay epochs
- **Loss**: L1 + LSGAN with λ=100
- **Optimizer**: Adam (lr=0.0002, β₁=0.5)
- **Data**: Real image-mask pairs from NeurIPS dataset

## Key Findings

1. **Synthetic augmentation is effective**: Small amounts of high-quality synthetic data provide significant performance gains
2. **Optimal ratio exists**: 10% synthetic addition yields maximum benefit with diminishing returns at higher ratios
3. **Real data remains essential**: Pure synthetic training performs significantly worse than mixed approaches
4. **Statistical robustness**: Results validated across multiple seeds with appropriate corrections

## Technical Details

- **Synthetic Generation**: Pix2Pix conditional GAN trained on real image-mask pairs
- **Resolution**: 512×512 synthetic images integrated with higher-resolution real data
- **Augmentation Strategy**: Additive (adding synthetic to real) rather than replacement
- **Statistical Analysis**: Paired t-tests with Bonferroni correction, effect size reporting

## Results Location

Complete study results, statistical analysis, and visualizations are available in `final_augmentation_results/`.
