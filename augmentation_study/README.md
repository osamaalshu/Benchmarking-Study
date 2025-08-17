# Modality Agnostic Controlled Augmentation Study

This directory contains a comprehensive framework for evaluating synthetic data augmentation using **Cascaded Diffusion Models** for cell microscopy image segmentation, based on the paper by [Yilmaz et al. (2024)](https://github.com/ruveydayilmaz0/cascaded_diffusion).

## 🎯 Study Overview

The study systematically evaluates the effectiveness of synthetic data augmentation across multiple dataset configurations while keeping the NeurIPS official splits exactly as they are (val/test untouched).

### Dataset Arms

- **R (Real-only)**: Original training set (baseline)
- **RxS@10**: Replace 10% of training images with synthetic
- **RxS@25**: Replace 25% of training images with synthetic
- **RxS@50**: Replace 50% of training images with synthetic
- **S (Synthetic-only)**: Synthetic pairs equal in size to R
- **Rmask+SynthTex@25**: 25% real masks with synthetic textures

### Models Evaluated

- **nnU-Net**: Best baseline model with self-configuring architecture
- **U-Net**: Simpler baseline model for comparison

### Training Protocol (Fixed Across Arms)

- **Seeds**: {0, 1, 2} per arm for statistical robustness
- **Optimizer**: AdamW with same LR, weight decay
- **Steps**: Fixed number of optimizer steps (adjust epochs if dataset size differs)
- **Augmentation**: Consistent across all arms
- **Patch size**: 256x256 pixels

### Evaluation Framework

- **Test set**: Always real and untouched
- **Metrics**: Dice, IoU, Precision, Recall, F1, Boundary F1, HD95
- **Statistics**: Paired t-tests (R vs each arm) with effect sizes
- **Multiple comparisons**: Bonferroni correction applied

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install base requirements (if not already installed)
pip install -r ../requirements.txt

# Install cascaded diffusion specific requirements
pip install -r requirements_cascaded.txt
```

### 2. Run Complete Study

```bash
# From the project root directory
python run_augmentation_study.py --train-data ./data/train-preprocessed --val-data ./data/val --test-data ./data/test
```

### 3. Custom Configuration

```bash
# Run with specific models and seeds
python run_augmentation_study.py --models nnunet --seeds 0 1 2

# Run with custom output directory
python run_augmentation_study.py --output-dir ./my_study_results

# Use pre-trained cascaded diffusion models (if available)
python run_augmentation_study.py --mask-model ./path/to/mask_model --texture-model ./path/to/texture_model
```

## 📁 Module Structure

```
augmentation_study/
├── __init__.py
├── README.md
├── requirements_cascaded.txt
├── cascaded_diffusion_wrapper.py    # Wrapper for cascaded diffusion model
├── data_arms_manager.py             # Creates and manages dataset arms
├── training_protocol.py             # Unified training across all arms
├── evaluation_framework.py          # Comprehensive evaluation and statistics
└── experiment_runner.py             # Main orchestrator
```

## 🔬 Detailed Usage

### Configuration Options

Create a configuration file to customize the study:

```json
{
  "train_data_dir": "./data/train-preprocessed",
  "val_data_dir": "./data/val",
  "test_data_dir": "./data/test",
  "output_dir": "./augmentation_study_results",
  "models": ["nnunet", "unet"],
  "seeds": [0, 1, 2],
  "batch_size": 4,
  "initial_lr": 6e-4,
  "max_epochs": 30,
  "input_size": 256,
  "num_classes": 3,
  "device": "auto",
  "mask_model_path": null,
  "texture_model_path": null
}
```

### Resume from Specific Phase

If the study is interrupted, you can resume from any phase:

```bash
python run_augmentation_study.py --resume-from dataset_arms_created
python run_augmentation_study.py --resume-from training_completed
python run_augmentation_study.py --resume-from evaluation_completed
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
python run_augmentation_study.py --debug
```

## 📊 Output Structure

The study generates comprehensive results:

```
augmentation_study_results/
├── STUDY_SUMMARY.md                 # High-level overview
├── study_state.json                 # Complete study metadata
├── experiment.log                   # Detailed execution log
├── dataset_arms/                    # Generated dataset arms
│   ├── R/                          # Real-only arm
│   ├── RxS@10/                     # 10% synthetic replacement
│   ├── RxS@25/                     # 25% synthetic replacement
│   ├── RxS@50/                     # 50% synthetic replacement
│   ├── S/                          # Synthetic-only arm
│   ├── Rmask+SynthTex@25/          # Real masks + synthetic textures
│   └── study_info.json             # Arms metadata
├── training_results/                # Training outputs
│   ├── nnunet/                     # nnU-Net results
│   │   ├── R_seed0/                # Individual training runs
│   │   ├── R_seed1/
│   │   └── ...
│   ├── unet/                       # U-Net results
│   └── comprehensive_results.json  # Training summary
└── evaluation_results/              # Evaluation outputs
    ├── comprehensive_evaluation.json
    ├── statistical_summary.csv     # Statistical comparison table
    ├── evaluation_report.md        # Detailed analysis report
    └── ...
```

## 🔍 Key Features

### Cascaded Diffusion Integration

- **Automatic Setup**: Downloads and configures Stable Diffusion models
- **Mask Generation**: Uses DDPM for synthetic mask creation
- **Texture Generation**: Conditioned texture synthesis with Stable Diffusion
- **Fallback Support**: Works with base models if custom models unavailable

### Statistical Rigor

- **Paired Testing**: Per-image comparisons between baseline and treatments
- **Effect Sizes**: Cohen's d for practical significance
- **Multiple Comparisons**: Bonferroni correction for family-wise error control
- **Confidence Intervals**: 95% CIs for difference estimates

### Reproducibility

- **Fixed Seeds**: Consistent randomization across all experiments
- **Version Control**: All configurations saved with results
- **Resumable**: Can restart from any phase if interrupted
- **Containerizable**: Ready for Docker deployment

## 📈 Expected Results

The study will generate statistical comparisons showing:

1. **Performance Impact**: How synthetic data affects segmentation metrics
2. **Optimal Ratios**: Which synthetic/real ratios work best
3. **Model Sensitivity**: How different models respond to synthetic data
4. **Statistical Significance**: Rigorous testing of improvements

## 🛠 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use smaller models
2. **CUDA Issues**: Set device to 'cpu' or 'mps' for Mac
3. **Missing Models**: Study works with base Stable Diffusion if custom models unavailable
4. **Data Format**: Ensure images and labels are in separate directories

### Performance Tips

- Use GPU acceleration when available
- Monitor disk space (synthetic generation can be large)
- Consider running on HPC for full study
- Use resume functionality for long experiments

## 📚 References

1. **Cascaded Diffusion**: Yilmaz, R. et al. (2024). "Cascaded Diffusion Models for 2D and 3D Microscopy Image Synthesis to Enhance Cell Segmentation." arXiv:2411.11515.

2. **Statistical Methods**: Bonferroni correction, paired t-tests, Cohen's d effect sizes

3. **Evaluation Metrics**: Standard segmentation metrics with boundary-aware measures

## 🤝 Contributing

This framework is designed to be:

- **Extensible**: Easy to add new models or metrics
- **Configurable**: JSON-based configuration system
- **Modular**: Each component can be used independently
- **Well-documented**: Comprehensive logging and reporting

For questions or improvements, please refer to the main project documentation.
