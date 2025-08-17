# Synthetic Data for Microscopy Segmentation

## Overview

This directory contains synthetic microscopy data generated using pix2pix and related tools for the augmentation study.

## 📁 Directory Structure

```
synthesis/
├── README.md                          # This file
├── synthetic_data_500/                # Main synthetic dataset
│   ├── synthetic_images_500/          # 500 synthetic cell images (512×512 RGB)
│   └── synthetic_labels_grayscale/    # Corresponding segmentation masks (512×512 L)
├── external/                          # External pix2pix implementation
└── checkpoints/                       # Model checkpoints (if any)
```

## 🎯 Synthetic Data Quality

The synthetic data in `synthetic_data_500/` was generated using a pix2pix model trained on cell microscopy images. Key characteristics:

- **Format**: PNG images, 512×512 resolution
- **Image Type**: RGB (3-channel) synthetic cell images
- **Label Type**: Grayscale (1-channel) segmentation masks
- **Classes**: 3-class segmentation (background, cell boundary, cell interior)
- **Quality**: High-quality synthetic pairs suitable for augmentation

## 📊 Usage in Augmentation Study

This synthetic data is used in the main augmentation study with the following strategy:

- **R+S@10**: Add 90 synthetic images (10% of 900 real images)
- **R+S@25**: Add 225 synthetic images (25% of 900 real images)  
- **R+S@50**: Add 450 synthetic images (50% of 900 real images)
- **S**: Use 450 synthetic images only (for comparison)

The synthetic data is automatically resized and integrated with real data during the study.

## 🔬 Data Generation Process

The synthetic data was generated using:

1. **pix2pix Model**: Conditional GAN for paired image-to-image translation
2. **Training Data**: Real microscopy images from the training set
3. **Style Control**: Image-based conditioning for realistic texture generation
4. **Quality Control**: Manual verification of synthetic pair quality

## 📈 Performance Impact

Key findings from the augmentation study:

- **R+S@10 (10% synthetic)**: **+44.5% Dice score improvement** (best performance)
- **R+S@25 (25% synthetic)**: +15.1% improvement (diminishing returns)
- **R+S@50 (50% synthetic)**: +31.7% improvement (good but not optimal)
- **S (100% synthetic)**: -83% performance decrease (synthetic alone fails)

**Conclusion**: Small amounts (10%) of high-quality synthetic data provide maximum augmentation benefit.

## 🛠 Generating New Synthetic Data

To generate additional synthetic data:

### Using Pix2Pix

1. **Setup Environment**:
   ```bash
   cd synthesis/external/
   pip install -r requirements.txt
   ```

2. **Prepare Training Data**:
   ```bash
   python prepare_pix2pix_dataset.py --input-dir ../../data/train-preprocessed --output-dir pix2pix_data
   ```

3. **Train Model**:
   ```bash
   python train_pix2pix.py --dataroot pix2pix_data --name cell_pix2pix --epochs 200
   ```

4. **Generate Synthetic Data**:
   ```bash
   python test_pix2pix.py --dataroot test_data --name cell_pix2pix --num_samples 500
   ```

### Using Alternative Methods

For other generative approaches (diffusion models, StyleGAN, etc.), see the main augmentation study framework which includes:

- `cascaded_diffusion_wrapper.py`: Diffusion model interface
- Stable Diffusion integration for texture generation
- ControlNet support for mask-conditioned generation

## 📊 Data Statistics

### Synthetic Images (synthetic_images_500/)
- **Count**: 500 images
- **Resolution**: 512×512 pixels
- **Format**: RGB PNG
- **File naming**: `synthetic_XXXXX.png` (where XXXXX is zero-padded index)
- **Size range**: ~200-800 KB per image

### Synthetic Labels (synthetic_labels_grayscale/)
- **Count**: 500 masks
- **Resolution**: 512×512 pixels  
- **Format**: Grayscale PNG
- **File naming**: `synthetic_XXXXX_label.png`
- **Classes**: 0 (background), 128 (cell boundary), 255 (cell interior)
- **Size range**: ~20-100 KB per mask

## 🔍 Quality Assessment

The synthetic data quality was validated through:

1. **Visual Inspection**: Manual review of image-mask pairs
2. **Statistical Comparison**: Distribution matching with real data
3. **Augmentation Performance**: Empirical validation in segmentation tasks
4. **Ablation Studies**: Comparison with different synthetic ratios

## 📚 References

- **Pix2Pix**: Isola et al. "Image-to-Image Translation with Conditional Adversarial Networks" (2017)
- **Augmentation Study**: See main study results in `../final_augmentation_results/`
- **Cell Segmentation**: Original NeurIPS Cell Segmentation Challenge dataset

## 🤝 Contributing

To add new synthetic data:

1. Follow the naming convention (`synthetic_XXXXX.png` / `synthetic_XXXXX_label.png`)
2. Ensure 512×512 resolution
3. Validate image-mask pair alignment
4. Update count in this README
5. Test with augmentation study framework

## 📄 License

Synthetic data generated for research purposes. Original pix2pix implementation follows its respective license terms.

---

**Last Updated**: August 2025  
**Synthetic Data Count**: 500 image-mask pairs