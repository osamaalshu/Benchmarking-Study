# Comprehensive Model Comparison Figure

This directory contains the comprehensive model comparison visualization for the thesis.

## Files

- `comprehensive_model_comparison.png` - High-resolution PNG version of the figure
- `comprehensive_model_comparison.pdf` - Vector PDF version for publication
- `comprehensive_model_comparison_figure.tex` - LaTeX code for including the figure

## Figure Description

The comprehensive model comparison figure consists of three subplots:

### (a) Performance Radar Chart

Shows the performance of all models across five key metrics:

- F1 Score
- Dice Score
- Precision
- Recall
- Boundary F1 (approximated using F1 score)

### (b) Efficiency Plot

Displays F1 score versus training time to show model efficiency:

- X-axis: Training time in hours
- Y-axis: F1 score
- Each point represents a model
- Higher F1 with lower training time = more efficient

### (c) Robustness Assessment

Shows model robustness based on performance variance:

- X-axis: Robustness score (1 - normalized variance)
- Y-axis: Mean F1 score
- Lower variance across different thresholds = more robust

## Models Included

1. **MAUNET_ENSEMBLE** - Best overall performance (F1: 0.6015)
2. **MAUNET_RESNET50** - Strong performance with ResNet50 backbone
3. **MAUNET_WIDE** - Wide-ResNet50 variant
4. **NNUNET** - Medical imaging standard
5. **UNET** - Baseline U-Net implementation
6. **LSTMUNET** - U-Net with LSTM layers
7. **SAC** - Segment Anything + Custom head

## Usage in LaTeX

Copy the contents of `comprehensive_model_comparison_figure.tex` into your LaTeX document:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/comprehensive_model_comparison.png}
\caption{Multi-dimensional model comparison. (a) Performance radar chart showing F1, Dice, Precision, Recall, and Boundary F1. (b) Efficiency plot of F1 score versus training time. (c) Robustness assessment showing performance variance across image modalities.}
\label{fig:model_comparison}
\end{figure}
```

## Key Findings

- **Best F1 Score**: MAUNET_ENSEMBLE (0.6015)
- **Best Dice Score**: MAUNET_WIDE (0.7303)
- **Most Efficient**: UNET (0.1336 F1/hour)
- **Most Robust**: SAC (0.6184)

## Generation

The figure was generated using the script `create_comprehensive_model_comparison.py` which:

1. Loads performance data from CSV files
2. Creates radar chart using polar coordinates
3. Calculates efficiency metrics
4. Analyzes robustness across different IoU thresholds
5. Combines all visualizations into a single figure

## Requirements

- Python 3.7+
- matplotlib
- pandas
- numpy
- seaborn
