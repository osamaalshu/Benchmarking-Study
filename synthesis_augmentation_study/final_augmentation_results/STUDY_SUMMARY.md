# Modality Agnostic Controlled Augmentation Study
## Using Cascaded Diffusion for Microscopy Image Synthesis

**Study Completion Date:** 2025-08-17T21:03:45.381242
**Total Duration:** 1.37 hours
**Models Evaluated:** nnunet
**Seeds Used:** [0, 1, 2]

## Study Overview

This study evaluates the effectiveness of synthetic data augmentation using
cascaded diffusion models for cell microscopy image segmentation.

### Dataset Arms Evaluated
- **R (Real-only):** Original training set as baseline
- **RxS@10:** 10% synthetic replacement
- **RxS@25:** 25% synthetic replacement
- **RxS@50:** 50% synthetic replacement
- **S (Synthetic-only):** 100% synthetic data
- **Rmask+SynthTex@25:** Real masks with 25% synthetic textures

## Key Findings

### Statistical Significance
Results are based on paired t-tests with Bonferroni correction for multiple comparisons.

### Performance Summary
Detailed results are available in:
- Statistical summary: `final_augmentation_results/evaluation_results/statistical_summary.csv`
- Comprehensive evaluation: `final_augmentation_results/evaluation_results/evaluation_report.md`

## Files Generated

### Dataset Arms
- Location: `fixed_dataset_arms/`
- Contains all generated dataset arms with metadata

### Training Results
- Location: `final_augmentation_results/training_results/`
- Contains trained models and training logs for all arms and seeds

### Evaluation Results
- Location: `final_augmentation_results/evaluation_results/`
- Contains comprehensive evaluation metrics and statistical analysis

## Reproducibility

This study used fixed seeds ([0, 1, 2]) and standardized
hyperparameters across all arms to ensure fair comparison.

Configuration details are saved in `study_state.json`.

## Citation

If you use this work, please cite:
- The cascaded diffusion paper: Yilmaz et al. (2024)
- This benchmarking study: [Your study details]