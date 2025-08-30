#!/usr/bin/env python3
"""
Create density-based performance analysis charts
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
    'maunet_ensemble': '#2ca02c',    # Green - Best precision/PQ
    'maunet_wide': '#1f77b4',        # Blue - Best F1
    'maunet_resnet50': '#ff7f0e',    # Orange - Good performance
    'nnunet': '#d62728',             # Red - Moderate performance
    'unet': '#9467bd',               # Purple - Poor performance
    'lstmunet': '#8c564b',           # Brown - Poor performance  
    'sac': '#e377c2'                 # Pink - Failure
}

# Create output directory
output_dir = Path("latex_tables_figures/png_figures")
output_dir.mkdir(exist_ok=True)

def load_error_data():
    """Load the error analysis data"""
    try:
        error_data = pd.read_csv("results/error_analysis/error_summary.csv")
        return error_data
    except FileNotFoundError:
        print("Error: Could not find error_summary.csv file")
        return None

def create_density_performance_analysis(df):
    """Create analysis of performance vs cell density"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cell Density Impact on Model Performance', fontsize=16, fontweight='bold')
    
    models = df['model'].unique()
    colors = [MODEL_COLORS.get(model, '#808080') for model in models]
    
    # Create density bins
    df['density_bin'] = pd.cut(df['gt_count'], bins=[0, 200, 400, 600, 800, 2000], 
                              labels=['Very Low\n(0-200)', 'Low\n(200-400)', 'Medium\n(400-600)', 
                                     'High\n(600-800)', 'Very High\n(800+)'])
    
    # 1. F1-Score vs Density
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        density_performance = model_data.groupby('density_bin')['f1_score'].agg(['mean', 'std']).reset_index()
        
        ax1.errorbar(range(len(density_performance)), density_performance['mean'], 
                    yerr=density_performance['std'], marker='o', label=model, 
                    color=colors[i], capsize=5, capthick=2)
    
    ax1.set_xlabel('Cell Density Category', fontweight='bold')
    ax1.set_ylabel('Average F1-Score', fontweight='bold')
    ax1.set_title('F1-Score vs Cell Density', fontweight='bold')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['Very Low\n(0-200)', 'Low\n(200-400)', 'Medium\n(400-600)', 
                        'High\n(600-800)', 'Very High\n(800+)'])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Success Rate by Density (F1 > 0.5)
    success_rates = []
    for model in models:
        model_rates = []
        for density_cat in df['density_bin'].cat.categories:
            subset = df[(df['model'] == model) & (df['density_bin'] == density_cat)]
            if len(subset) > 0:
                success_rate = (subset['f1_score'] > 0.5).mean() * 100
                model_rates.append(success_rate)
            else:
                model_rates.append(0)
        success_rates.append(model_rates)
    
    x = np.arange(5)
    width = 0.1
    for i, (model, rates) in enumerate(zip(models, success_rates)):
        ax2.bar(x + i*width, rates, width, label=model, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Cell Density Category', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontweight='bold')
    ax2.set_title('Success Rate (F1 > 0.5) by Density', fontweight='bold')
    ax2.set_xticks(x + width * (len(models)-1) / 2)
    ax2.set_xticklabels(['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error Rate vs Density
    df['total_errors'] = df['false_negatives'] + df['false_positives'] + df['splits'] + df['merges']
    df['error_rate'] = df['total_errors'] / df['gt_count']  # Errors per GT cell
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        density_errors = model_data.groupby('density_bin')['error_rate'].mean().reset_index()
        
        ax3.plot(range(len(density_errors)), density_errors['error_rate'], 
                marker='s', label=model, color=colors[i], linewidth=2, markersize=8)
    
    ax3.set_xlabel('Cell Density Category', fontweight='bold')
    ax3.set_ylabel('Error Rate (Errors per GT Cell)', fontweight='bold')
    ax3.set_title('Error Rate vs Cell Density', fontweight='bold')
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution of GT cell counts
    ax4.hist([df[df['model'] == model]['gt_count'] for model in models[:3]], 
             bins=30, alpha=0.7, label=models[:3], density=True)
    ax4.set_xlabel('Ground Truth Cell Count', fontweight='bold')
    ax4.set_ylabel('Density', fontweight='bold')
    ax4.set_title('Distribution of Cell Densities in Dataset', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'density_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created density_performance_analysis.png")

def create_image_level_success_failure_rates(df):
    """Create chart showing success/failure rates at image level"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Image-Level Performance Analysis', fontsize=16, fontweight='bold')
    
    models = df['model'].unique()
    colors = [MODEL_COLORS.get(model, '#808080') for model in models]
    
    # Define success thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    # 1. Success rates at different F1 thresholds
    success_data = []
    for model in models:
        model_data = df[df['model'] == model]
        model_success = []
        for threshold in thresholds:
            success_rate = (model_data['f1_score'] >= threshold).mean() * 100
            model_success.append(success_rate)
        success_data.append(model_success)
    
    x = np.arange(len(thresholds))
    for i, (model, success_rates) in enumerate(zip(models, success_data)):
        ax1.plot(x, success_rates, marker='o', linewidth=3, markersize=8, 
                label=model, color=colors[i])
    
    ax1.set_xlabel('F1-Score Threshold', fontweight='bold')
    ax1.set_ylabel('Percentage of Images (%)', fontweight='bold')
    ax1.set_title('Success Rate Across Different Performance Thresholds', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'≥{t}' for t in thresholds])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add percentage labels
    for i, (model, success_rates) in enumerate(zip(models, success_data)):
        for j, rate in enumerate(success_rates):
            if rate > 5:  # Only label significant rates
                ax1.annotate(f'{rate:.1f}%', (j, rate), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
    
    # 2. Performance distribution histogram
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        ax2.hist(model_data['f1_score'], bins=20, alpha=0.6, label=model, 
                color=colors[i], density=True)
    
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Success Threshold (0.5)')
    ax2.set_xlabel('F1-Score', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('Distribution of F1-Scores Across All Images', fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'image_level_success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created image_level_success_rates.png")

def create_challenging_vs_easy_images_analysis(df):
    """Analyze performance on challenging vs easy images"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Challenging vs Easy Images Analysis', fontsize=16, fontweight='bold')
    
    models = df['model'].unique()
    colors = [MODEL_COLORS.get(model, '#808080') for model in models]
    
    # Calculate average F1 per image across all models
    image_avg_f1 = df.groupby('image')['f1_score'].mean().reset_index()
    image_avg_f1['difficulty'] = pd.cut(image_avg_f1['f1_score'], 
                                       bins=[0, 0.2, 0.4, 0.6, 1.0],
                                       labels=['Very Hard', 'Hard', 'Medium', 'Easy'])
    
    # Merge back with original data
    df_with_difficulty = df.merge(image_avg_f1[['image', 'difficulty']], on='image')
    
    # 1. Performance by difficulty category
    difficulty_performance = df_with_difficulty.groupby(['model', 'difficulty'])['f1_score'].mean().unstack()
    
    difficulty_performance.plot(kind='bar', ax=ax1, color=[
        PERFORMANCE_COLORS['failure'],    # Very Hard - Red
        PERFORMANCE_COLORS['poor'],       # Hard - Orange  
        PERFORMANCE_COLORS['good'],       # Medium - Blue
        PERFORMANCE_COLORS['excellent']   # Easy - Green
    ])
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Average F1-Score', fontweight='bold')
    ax1.set_title('Performance by Image Difficulty Category', fontweight='bold')
    ax1.legend(title='Image Difficulty', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Percentage of images in each difficulty category
    difficulty_counts = image_avg_f1['difficulty'].value_counts()
    ax2.pie(difficulty_counts.values, labels=difficulty_counts.index, autopct='%1.1f%%',
            colors=[
                PERFORMANCE_COLORS['failure'],    # Very Hard - Red
                PERFORMANCE_COLORS['poor'],       # Hard - Orange
                PERFORMANCE_COLORS['good'],       # Medium - Blue  
                PERFORMANCE_COLORS['excellent']   # Easy - Green
            ])
    ax2.set_title('Distribution of Image Difficulty', fontweight='bold')
    
    # 3. Model robustness (std deviation across difficulties)
    model_robustness = df_with_difficulty.groupby('model')['f1_score'].std().sort_values()
    robustness_colors = [MODEL_COLORS.get(model, '#808080') for model in model_robustness.index]
    ax3.barh(range(len(model_robustness)), model_robustness.values, 
             color=robustness_colors)
    ax3.set_yticks(range(len(model_robustness)))
    ax3.set_yticklabels(model_robustness.index)
    ax3.set_xlabel('F1-Score Standard Deviation', fontweight='bold')
    ax3.set_title('Model Robustness (Lower = More Consistent)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Success rate on hard vs easy images
    hard_images = image_avg_f1[image_avg_f1['difficulty'].isin(['Very Hard', 'Hard'])]['image'].tolist()
    easy_images = image_avg_f1[image_avg_f1['difficulty'].isin(['Easy', 'Medium'])]['image'].tolist()
    
    success_comparison = []
    for model in models:
        model_data = df[df['model'] == model]
        hard_success = (model_data[model_data['image'].isin(hard_images)]['f1_score'] > 0.3).mean() * 100
        easy_success = (model_data[model_data['image'].isin(easy_images)]['f1_score'] > 0.5).mean() * 100
        success_comparison.append([hard_success, easy_success])
    
    success_comparison = np.array(success_comparison)
    x = np.arange(len(models))
    width = 0.35
    
    ax4.bar(x - width/2, success_comparison[:, 0], width, label='Hard Images (F1>0.3)', 
            color=PERFORMANCE_COLORS['poor'], alpha=0.8)
    ax4.bar(x + width/2, success_comparison[:, 1], width, label='Easy Images (F1>0.5)', 
            color=PERFORMANCE_COLORS['excellent'], alpha=0.8)
    
    ax4.set_xlabel('Model', fontweight='bold')
    ax4.set_ylabel('Success Rate (%)', fontweight='bold')
    ax4.set_title('Success Rate: Hard vs Easy Images', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'challenging_vs_easy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created challenging_vs_easy_analysis.png")

def create_density_correlation_analysis(df):
    """Create correlation analysis between density and various metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cell Density Correlation Analysis', fontsize=16, fontweight='bold')
    
    models = df['model'].unique()[:4]  # Focus on top 4 models for clarity
    colors = [MODEL_COLORS.get(model, '#808080') for model in models]
    
    # 1. Scatter: GT Count vs F1-Score
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        ax1.scatter(model_data['gt_count'], model_data['f1_score'], 
                   alpha=0.6, label=model, color=colors[i], s=50)
    
    ax1.set_xlabel('Ground Truth Cell Count', fontweight='bold')
    ax1.set_ylabel('F1-Score', fontweight='bold')
    ax1.set_title('Cell Density vs Performance', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add trend lines
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        z = np.polyfit(model_data['gt_count'], model_data['f1_score'], 1)
        p = np.poly1d(z)
        ax1.plot(model_data['gt_count'], p(model_data['gt_count']), 
                color=colors[i], linestyle='--', alpha=0.8)
    
    # 2. Scatter: GT Count vs Precision
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        ax2.scatter(model_data['gt_count'], model_data['precision'], 
                   alpha=0.6, label=model, color=colors[i], s=50)
    
    ax2.set_xlabel('Ground Truth Cell Count', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Cell Density vs Precision', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter: GT Count vs Recall
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        ax3.scatter(model_data['gt_count'], model_data['recall'], 
                   alpha=0.6, label=model, color=colors[i], s=50)
    
    ax3.set_xlabel('Ground Truth Cell Count', fontweight='bold')
    ax3.set_ylabel('Recall', fontweight='bold')
    ax3.set_title('Cell Density vs Recall', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation heatmap
    correlation_data = []
    for model in models:
        model_data = df[df['model'] == model]
        corr = model_data[['gt_count', 'f1_score', 'precision', 'recall', 'PQ']].corr()['gt_count'][1:]
        correlation_data.append(corr.values)
    
    correlation_matrix = np.array(correlation_data)
    im = ax4.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(['F1-Score', 'Precision', 'Recall', 'PQ'])
    ax4.set_yticks(range(len(models)))
    ax4.set_yticklabels(models)
    ax4.set_title('Correlation with Cell Density', fontweight='bold')
    
    # Add correlation values
    for i in range(len(models)):
        for j in range(4):
            text = ax4.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax4, label='Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'density_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created density_correlation_analysis.png")

def main():
    """Generate all density and image-level analysis charts"""
    print("Loading error analysis data...")
    
    df = load_error_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded data for {len(df)} evaluations across {df['model'].nunique()} models")
    print(f"Cell density range: {df['gt_count'].min()}-{df['gt_count'].max()} cells per image")
    
    print("\nCreating density and image-level analysis charts...")
    
    create_density_performance_analysis(df)
    create_image_level_success_failure_rates(df)
    create_challenging_vs_easy_images_analysis(df)
    create_density_correlation_analysis(df)
    
    print(f"\n🎉 All analysis charts created successfully in: {output_dir}")
    print("\nGenerated files:")
    for png_file in sorted(output_dir.glob("*analysis*.png")):
        print(f"  - {png_file.name}")

if __name__ == "__main__":
    main()
