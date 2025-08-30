# LaTeX Tables and Figures for Error Analysis

This folder contains all tables and figures from the error analysis converted to LaTeX format for easy inclusion in your thesis.

## Files Overview

### Tables (.tex files)

- `model_performance_table.tex` - Main performance comparison table with F1, PQ, Precision, Recall
- `error_analysis_table.tex` - Error type breakdown (FN, FP, Splits, Merges)
- `dataset_overview_table.tex` - Dataset statistics and model descriptions
- `panoptic_quality_table.tex` - Detailed PQ, RQ, SQ breakdown

### Figures (.tex files)

- `comprehensive_visualization_figure.tex` - 4×7 grid error analysis examples
- `best_worst_analysis_figure.tex` - Challenging vs easy image analysis
- `success_failure_analysis_figure.tex` - Model-specific success/failure cases
- `performance_comparison_chart.tex` - Bar chart comparing key metrics (TikZ version)
- `error_distribution_chart.tex` - Stacked bar chart of error types (TikZ version)
- `png_chart_figures.tex` - **HIGH-QUALITY PNG CHARTS** (recommended)
- `density_analysis_figures.tex` - **DENSITY & IMAGE-LEVEL ANALYSIS** (new)

### PNG Figures (High Quality - Recommended)

Located in `png_figures/` subdirectory:

#### Performance Comparison Charts:

- `performance_comparison.png` - Performance metrics bar chart (Green=Excellent, Blue=Good, Orange=Poor, Red=Failure)
- `error_distribution.png` - Stacked error type distribution with performance-based colors
- `precision_recall_analysis.png` - Precision vs recall scatter plot with F1 contours
- `panoptic_quality_breakdown.png` - PQ component analysis (RQ vs SQ breakdown)

#### Density & Image-Level Analysis Charts:

- `density_performance_analysis.png` - Cell density impact on performance (4-panel analysis)
- `image_level_success_rates.png` - Success rates and F1-score distributions across images
- `challenging_vs_easy_analysis.png` - Performance on difficult vs easy images (4-panel analysis)
- `density_correlation_analysis.png` - Correlation between cell density and performance metrics

### Support Files

- `error_analysis_packages.tex` - All required LaTeX packages and formatting
- `main_error_analysis_document.tex` - Complete standalone document template
- `README.md` - This file

## How to Use

### Option 1: Include Individual Tables/Figures in Your Thesis

```latex
% In your thesis preamble, include the packages:
\input{path/to/error_analysis_packages.tex}

% In your document, include specific tables:
\input{path/to/model_performance_table.tex}
\input{path/to/error_analysis_table.tex}

% Include PNG charts (RECOMMENDED for best quality):
\input{path/to/png_chart_figures.tex}

% Include visualization figures:
\input{path/to/comprehensive_visualization_figure.tex}
```

### Option 2: Use the Complete Document Template

```bash
# Compile the complete document:
pdflatex main_error_analysis_document.tex
```

### Option 3: Copy-Paste Individual Elements

Each .tex file contains complete, self-contained LaTeX code that can be copied directly into your thesis.

## Required Packages

The following packages are required (included in `error_analysis_packages.tex`):

- `booktabs` - Professional table formatting
- `array` - Enhanced table columns
- `graphicx` - Include images
- `pgfplots` - Charts and graphs
- `tikz` - Drawing graphics
- `caption` - Enhanced captions
- `threeparttable` - Table notes

## Image Paths

Update the image paths in the figure files to match your document structure:

```latex
% Current paths assume:
\includegraphics[width=\textwidth]{results/error_analysis/comprehensive_visualizations/...}

% Adjust to your structure, e.g.:
\includegraphics[width=\textwidth]{figures/error_analysis/...}
```

## Customization

### Colors

Predefined colors in `error_analysis_packages.tex`:

- `maunetcolor` - Blue for MAUNet variants
- `traditionalcolor` - Orange for traditional models
- `errorcolor` - Red for errors
- `successcolor` - Green for success

### Formatting Commands

- `\bestvalue{0.529}` - Bold formatting for best values
- `\modelname{MAUNet-Wide}` - Bold formatting for model names

## Tables Summary

1. **Model Performance Table**: Key metrics comparison (F1, PQ, Precision, Recall)
2. **Error Analysis Table**: Detailed error breakdown per model
3. **Dataset Overview Table**: Dataset statistics and methodology
4. **Panoptic Quality Table**: PQ component analysis (RQ, SQ)

## Figures Summary

1. **Comprehensive Visualization**: 4×7 grid showing original, GT, predictions, errors
2. **Best/Worst Analysis**: Dataset difficulty analysis
3. **Success/Failure Analysis**: Model-specific performance examples
4. **Performance Charts**: Bar charts comparing metrics and error distributions

## Notes

- All tables use professional `booktabs` formatting
- Figures include detailed captions with analysis
- Charts are created with `pgfplots` for high-quality output
- All content is based on the comprehensive error analysis of 100 test images
- Best values are highlighted in bold throughout

## Compilation Tips

1. Ensure all image files are accessible from your LaTeX document
2. Use `pdflatex` for best results with graphics
3. May need multiple compilations for proper cross-references
4. Adjust figure sizes with `[width=0.8\textwidth]` as needed
