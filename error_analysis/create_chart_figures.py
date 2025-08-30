#!/usr/bin/env python3
"""
Create high-quality PNG charts for thesis figures
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')

# Define consistent color scheme: green (excellent), blue (good), orange (poor), red (failure)
PERFORMANCE_COLORS = {
    'excellent': '#2ca02c',  # Green
    'good': '#1f77b4',       # Blue  
    'poor': '#ff7f0e',       # Orange
    'failure': '#d62728'     # Red
}

# Model-specific color mapping (unique color for each model)
MODEL_COLORS = {
    'MAUNet-Ensemble': '#2ca02c',    # Green - Best precision/PQ
    'MAUNet-Wide': '#1f77b4',        # Blue - Best F1
    'MAUNet-ResNet50': '#ff7f0e',    # Orange - Good performance
    'nnU-Net': '#d62728',            # Red - Moderate performance
    'U-Net': '#9467bd',              # Purple - Poor performance
    'LSTM-UNet': '#8c564b',          # Brown - Poor performance  
    'SAC': '#e377c2'                 # Pink - Failure
}

# Create output directory
output_dir = Path("latex_tables_figures/png_figures")
output_dir.mkdir(exist_ok=True)

# Model performance data
models = ['SAC', 'LSTM-UNet', 'U-Net', 'nnU-Net', 'MAUNet-ResNet50', 'MAUNet-Wide', 'MAUNet-Ensemble']
f1_scores = [0.003, 0.282, 0.315, 0.357, 0.507, 0.529, 0.499]
pq_scores = [0.002, 0.199, 0.239, 0.278, 0.399, 0.413, 0.437]
precision = [0.005, 0.263, 0.335, 0.367, 0.598, 0.605, 0.632]
recall = [0.002, 0.305, 0.297, 0.348, 0.440, 0.470, 0.413]

# Error data
false_negatives = [453.5, 315.8, 319.6, 296.4, 254.7, 241.0, 241.0]
false_positives = [179.0, 389.0, 268.0, 272.7, 134.6, 139.1, 109.0]
splits = [0.1, 1.7, 3.1, 3.6, 0.9, 1.3, 1.3]
merges = [8.2, 41.8, 53.1, 47.3, 36.9, 35.2, 28.3]

def create_performance_comparison():
    """Create performance comparison bar chart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.25
    
    # Create bars with consistent color scheme
    model_colors = [MODEL_COLORS.get(model, '#808080') for model in models]
    bars1 = ax.bar(x - width, f1_scores, width, label='F1-Score', color=model_colors, alpha=0.8)
    bars2 = ax.bar(x, pq_scores, width, label='Panoptic Quality', color=model_colors, alpha=0.6)
    bars3 = ax.bar(x + width, precision, width, label='Precision', color=model_colors, alpha=0.4)
    
    # Customize chart
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison Across Key Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:  # Only label significant values
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created performance_comparison.png")

def create_error_distribution():
    """Create stacked error distribution chart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Reorder models by total error count (ascending)
    model_data = list(zip(models, false_negatives, false_positives, splits, merges))
    model_data.sort(key=lambda x: x[1] + x[2] + x[3] + x[4])  # Sort by total errors
    
    sorted_models = [x[0] for x in model_data]
    sorted_fn = [x[1] for x in model_data]
    sorted_fp = [x[2] for x in model_data]
    sorted_splits = [x[3] for x in model_data]
    sorted_merges = [x[4] for x in model_data]
    
    x = np.arange(len(sorted_models))
    
    # Create stacked bars
    p1 = ax.bar(x, sorted_fn, label='False Negatives', color='#d62728', alpha=0.8)
    p2 = ax.bar(x, sorted_fp, bottom=sorted_fn, label='False Positives', color='#ff7f0e', alpha=0.8)
    p3 = ax.bar(x, sorted_splits, bottom=np.array(sorted_fn) + np.array(sorted_fp), 
                label='Splits', color='#2ca02c', alpha=0.8)
    p4 = ax.bar(x, sorted_merges, 
                bottom=np.array(sorted_fn) + np.array(sorted_fp) + np.array(sorted_splits),
                label='Merges', color='#9467bd', alpha=0.8)
    
    # Customize chart
    ax.set_xlabel('Model (Ordered by Total Error Count)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Error Count per Image', fontsize=14, fontweight='bold')
    ax.set_title('Error Type Distribution Across Models', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_models, rotation=45, ha='right')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created error_distribution.png")

def create_precision_recall_chart():
    """Create precision vs recall scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', '#e377c2']
    
    for i, (model, prec, rec) in enumerate(zip(models, precision, recall)):
        ax.scatter(rec, prec, s=150, c=colors[i], alpha=0.8, label=model, edgecolors='black', linewidth=1)
        
        # Add model labels
        ax.annotate(model, (rec, prec), xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, ha='left')
    
    # Customize chart
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision vs Recall Trade-off Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.7)
    
    # Add diagonal lines for F1-score contours
    f1_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    for f1 in f1_levels:
        recall_range = np.linspace(0.01, 0.7, 100)
        precision_range = (f1 * recall_range) / (2 * recall_range - f1)
        valid_idx = (precision_range > 0) & (precision_range <= 0.7)
        ax.plot(recall_range[valid_idx], precision_range[valid_idx], 
                '--', alpha=0.5, color='gray', linewidth=1)
        
        # Label F1 contours
        if f1 <= 0.4:
            ax.text(0.6, f1*0.6/(2*0.6-f1), f'F1={f1}', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created precision_recall_analysis.png")

def create_panoptic_quality_breakdown():
    """Create PQ breakdown chart showing RQ and SQ components"""
    # PQ = RQ * SQ, so we can derive RQ and SQ
    rq_scores = f1_scores  # RQ is essentially F1-score
    sq_scores = [pq/rq if rq > 0 else 0 for pq, rq in zip(pq_scores, rq_scores)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: PQ components
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, rq_scores, width, label='Recognition Quality (RQ)', 
                    color='#1f77b4', alpha=0.8)
    bars2 = ax1.bar(x + width/2, sq_scores, width, label='Segmentation Quality (SQ)', 
                    color='#ff7f0e', alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
    ax1.set_title('Panoptic Quality Components', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: PQ vs components correlation
    ax2.scatter(rq_scores, sq_scores, s=150, c=range(len(models)), 
               cmap='viridis', alpha=0.8, edgecolors='black', linewidth=1)
    
    for i, model in enumerate(models):
        ax2.annotate(model, (rq_scores[i], sq_scores[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Recognition Quality (RQ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Segmentation Quality (SQ)', fontsize=12, fontweight='bold')
    ax2.set_title('RQ vs SQ Trade-off', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'panoptic_quality_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created panoptic_quality_breakdown.png")

def create_model_ranking_chart():
    """Create comprehensive model ranking visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create ranking data
    metrics = ['F1-Score', 'Panoptic Quality', 'Precision', 'Recall']
    values = [f1_scores, pq_scores, precision, recall]
    
    # Create heatmap-style visualization
    data_matrix = np.array(values).T  # Transpose for models x metrics
    
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.7)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.3f}', 
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Customize chart
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_yticklabels(models, fontsize=12)
    ax.set_title('Model Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Performance Score', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_ranking_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created model_ranking_heatmap.png")

def main():
    """Generate all chart figures"""
    print("Creating high-quality PNG charts...")
    
    create_performance_comparison()
    create_error_distribution()
    create_precision_recall_chart()
    create_panoptic_quality_breakdown()
    create_model_ranking_chart()
    
    print(f"\n🎉 All charts created successfully in: {output_dir}")
    print("\nGenerated files:")
    for png_file in output_dir.glob("*.png"):
        print(f"  - {png_file.name}")

if __name__ == "__main__":
    main()
