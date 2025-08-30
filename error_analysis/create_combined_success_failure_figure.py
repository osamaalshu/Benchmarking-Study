#!/usr/bin/env python3
"""
Create combined success and failure analysis figure for all models
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from PIL import Image
import tifffile as tiff

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')

# Define consistent color scheme
PERFORMANCE_COLORS = {
    'excellent': '#2ca02c',  # Green
    'good': '#1f77b4',       # Blue  
    'poor': '#ff7f0e',       # Orange
    'failure': '#d62728'     # Red
}

# Model-specific color mapping
MODEL_COLORS = {
    'maunet_ensemble': PERFORMANCE_COLORS['excellent'],
    'maunet_wide': PERFORMANCE_COLORS['excellent'],
    'maunet_resnet50': PERFORMANCE_COLORS['good'],
    'nnunet': PERFORMANCE_COLORS['good'],
    'unet': PERFORMANCE_COLORS['poor'],
    'lstmunet': PERFORMANCE_COLORS['poor'],
    'sac': PERFORMANCE_COLORS['failure']
}

# Create output directory
output_dir = Path("latex_tables_figures/png_figures")
output_dir.mkdir(exist_ok=True)

# Base paths
BASE_DIR = Path(__file__).parent
TEST_IMAGES_PATH = BASE_DIR.parent / "data" / "test" / "images"

def load_error_data():
    """Load the error analysis data"""
    try:
        error_data = pd.read_csv("results/error_analysis/error_summary.csv")
        return error_data
    except FileNotFoundError:
        print("Error: Could not find error_summary.csv file")
        return None

def enhance_image_for_display(image):
    """Enhance image contrast and brightness for better visualization"""
    if image.dtype != np.float32:
        enhanced = image.astype(np.float32)
    else:
        enhanced = image.copy()
    
    # Normalize to 0-1 range first if needed
    if enhanced.max() > 1.0:
        enhanced = enhanced / 255.0
    
    # Handle RGB and grayscale differently
    if len(enhanced.shape) == 3 and enhanced.shape[2] == 3:
        # RGB image - enhance each channel separately
        for c in range(3):
            channel = enhanced[:, :, c]
            if channel.max() == 0:
                continue
                
            channel_min = channel.min()
            channel_max = channel.max()
            channel_mean = channel.mean()
            contrast_ratio = (channel_max - channel_min) / (channel_max + 1e-8)
            
            if channel_max > channel_min:
                if contrast_ratio < 0.05:  # Very low contrast
                    channel = (channel - channel_min) / (channel_max - channel_min + 1e-8)
                    gamma = 0.2
                    channel = np.power(channel, gamma)
                    channel = np.clip((channel - 0.5) * 2.0 + 0.5, 0, 1)
                else:
                    channel = (channel - channel_min) / (channel_max - channel_min)
                    if channel_mean < 0.1:
                        gamma = 0.4
                    elif channel_mean < 0.3:
                        gamma = 0.6
                    elif channel_mean < 0.6:
                        gamma = 0.8
                    else:
                        gamma = 0.9
                    
                    channel = np.power(channel, gamma)
                    if contrast_ratio < 0.2:
                        channel = np.clip((channel - 0.5) * 1.5 + 0.5, 0, 1)
                    else:
                        channel = np.clip((channel - 0.5) * 1.2 + 0.5, 0, 1)
                
                enhanced[:, :, c] = channel
    else:
        # Grayscale image - convert to RGB after enhancement
        if len(enhanced.shape) == 3:
            enhanced = enhanced[:, :, 0]
        
        if enhanced.max() > enhanced.min():
            channel_min = enhanced.min()
            channel_max = enhanced.max()
            channel_mean = enhanced.mean()
            contrast_ratio = (channel_max - channel_min) / (channel_max + 1e-8)
            
            if contrast_ratio < 0.05:
                enhanced = (enhanced - channel_min) / (channel_max - channel_min + 1e-8)
                enhanced = np.power(enhanced, 0.2)
                enhanced = np.clip((enhanced - 0.5) * 3.0 + 0.5, 0, 1)
            else:
                enhanced = (enhanced - channel_min) / (channel_max - channel_min)
                if channel_mean < 0.1:
                    gamma = 0.4
                elif channel_mean < 0.6:
                    gamma = 0.7
                else:
                    gamma = 0.8
                
                enhanced = np.power(enhanced, gamma)
                if contrast_ratio < 0.2:
                    enhanced = np.clip((enhanced - 0.5) * 2.0 + 0.5, 0, 1)
                else:
                    enhanced = np.clip((enhanced - 0.5) * 1.3 + 0.5, 0, 1)
        
        # Convert grayscale to RGB
        enhanced = np.stack([enhanced] * 3, axis=-1)
    
    # Convert back to uint8 for display
    return (enhanced * 255).astype(np.uint8)

def load_and_enhance_image(image_name):
    """Load and enhance an image for display"""
    try:
        image_path = TEST_IMAGES_PATH / image_name
        if image_path.exists():
            if image_path.suffix.lower() in {'.tif', '.tiff'}:
                original_image = tiff.imread(str(image_path))
            else:
                original_image = np.array(Image.open(image_path))
            
            # Normalize to 0-255 range if needed (handle 16-bit images)
            if original_image.dtype == np.uint16:
                original_image = (original_image / 65535.0 * 255.0).astype(np.uint8)
            elif original_image.dtype != np.uint8:
                if original_image.max() > 255:
                    original_image = (original_image / original_image.max() * 255.0).astype(np.uint8)
                else:
                    original_image = original_image.astype(np.uint8)
            
            # Convert to RGB if grayscale
            if len(original_image.shape) == 2:
                original_image = np.stack([original_image] * 3, axis=-1)
            elif len(original_image.shape) == 3 and original_image.shape[2] == 1:
                original_image = np.repeat(original_image, 3, axis=2)
            
            # Enhance for better visibility
            return enhance_image_for_display(original_image)
        else:
            return None
    except Exception as e:
        print(f"Error loading image {image_name}: {e}")
        return None

def create_combined_success_failure_analysis(df):
    """Create combined success/failure analysis for all models"""
    # Focus on top 4 models for clarity
    top_models = ['maunet_ensemble', 'maunet_wide', 'maunet_resnet50', 'nnunet']
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Success and Failure Case Analysis Across Models', fontsize=18, fontweight='bold', y=0.95)
    
    # Create grid: 4 models x 4 columns (model name, success case, failure case, performance)
    gs = fig.add_gridspec(4, 4, width_ratios=[1, 2, 2, 1], height_ratios=[1, 1, 1, 1],
                         hspace=0.3, wspace=0.2)
    
    for row, model in enumerate(top_models):
        model_data = df[df['model'] == model]
        
        # Find best and worst cases for this model
        best_case = model_data.loc[model_data['f1_score'].idxmax()]
        worst_case = model_data.loc[model_data['f1_score'].idxmin()]
        
        # Column 1: Model name and color
        ax_name = fig.add_subplot(gs[row, 0])
        ax_name.text(0.5, 0.5, model.upper().replace('_', '-'), 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color=MODEL_COLORS.get(model, '#808080'),
                    transform=ax_name.transAxes)
        ax_name.text(0.5, 0.2, f'Avg F1: {model_data["f1_score"].mean():.3f}', 
                    ha='center', va='center', fontsize=10,
                    transform=ax_name.transAxes)
        ax_name.set_xlim(0, 1)
        ax_name.set_ylim(0, 1)
        ax_name.axis('off')
        
        # Column 2: Success case
        ax_success = fig.add_subplot(gs[row, 1])
        success_image = load_and_enhance_image(best_case['image'])
        if success_image is not None:
            ax_success.imshow(success_image)
        ax_success.set_title(f'SUCCESS: {Path(best_case["image"]).stem}\nF1: {best_case["f1_score"]:.3f}, '
                           f'GT: {best_case["gt_count"]}, Pred: {best_case["pred_count"]}', 
                           fontsize=10, color=PERFORMANCE_COLORS['excellent'], fontweight='bold')
        ax_success.axis('off')
        
        # Column 3: Failure case
        ax_failure = fig.add_subplot(gs[row, 2])
        failure_image = load_and_enhance_image(worst_case['image'])
        if failure_image is not None:
            ax_failure.imshow(failure_image)
        ax_failure.set_title(f'FAILURE: {Path(worst_case["image"]).stem}\nF1: {worst_case["f1_score"]:.3f}, '
                           f'GT: {worst_case["gt_count"]}, Pred: {worst_case["pred_count"]}', 
                           fontsize=10, color=PERFORMANCE_COLORS['failure'], fontweight='bold')
        ax_failure.axis('off')
        
        # Column 4: Performance distribution
        ax_perf = fig.add_subplot(gs[row, 3])
        ax_perf.hist(model_data['f1_score'], bins=15, alpha=0.7, 
                    color=MODEL_COLORS.get(model, '#808080'), density=True)
        ax_perf.axvline(x=model_data['f1_score'].mean(), color='black', 
                       linestyle='--', linewidth=2, label=f'Mean: {model_data["f1_score"].mean():.3f}')
        ax_perf.axvline(x=best_case['f1_score'], color=PERFORMANCE_COLORS['excellent'], 
                       linestyle='-', linewidth=2, alpha=0.8)
        ax_perf.axvline(x=worst_case['f1_score'], color=PERFORMANCE_COLORS['failure'], 
                       linestyle='-', linewidth=2, alpha=0.8)
        ax_perf.set_xlabel('F1-Score', fontsize=9)
        ax_perf.set_ylabel('Density', fontsize=9)
        ax_perf.tick_params(axis='both', which='major', labelsize=8)
        ax_perf.grid(True, alpha=0.3)
    
    # Add column headers
    col_titles = ['Model', 'Best Performance', 'Worst Performance', 'F1 Distribution']
    for col, title in enumerate(col_titles):
        fig.text(0.125 + col * 0.22, 0.92, title, ha='center', va='center', 
                fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=PERFORMANCE_COLORS['excellent'], lw=3, label='Success Case'),
        plt.Line2D([0], [0], color=PERFORMANCE_COLORS['failure'], lw=3, label='Failure Case'),
        plt.Line2D([0], [0], color='black', linestyle='--', lw=2, label='Mean Performance')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
              bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_success_failure_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created combined_success_failure_analysis.png")

def create_architectural_family_comparison(df):
    """Create comparison showing architectural families"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Architectural Family Performance Comparison', fontsize=16, fontweight='bold')
    
    # Define architectural families
    families = {
        'MAUNet Family': ['maunet_ensemble', 'maunet_wide', 'maunet_resnet50'],
        'Traditional CNN': ['unet', 'nnunet'],
        'Attention-based': ['lstmunet'],
        'Other': ['sac']
    }
    
    family_colors = {
        'MAUNet Family': PERFORMANCE_COLORS['excellent'],
        'Traditional CNN': PERFORMANCE_COLORS['good'],
        'Attention-based': PERFORMANCE_COLORS['poor'],
        'Other': PERFORMANCE_COLORS['failure']
    }
    
    # 1. Family performance comparison
    family_performance = []
    for family, models in families.items():
        family_data = df[df['model'].isin(models)]
        avg_f1 = family_data['f1_score'].mean()
        avg_pq = family_data['PQ'].mean()
        family_performance.append({'family': family, 'f1_score': avg_f1, 'PQ': avg_pq})
    
    family_df = pd.DataFrame(family_performance)
    
    x = np.arange(len(family_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, family_df['f1_score'], width, label='F1-Score',
                   color=[family_colors[f] for f in family_df['family']], alpha=0.8)
    bars2 = ax1.bar(x + width/2, family_df['PQ'], width, label='Panoptic Quality',
                   color=[family_colors[f] for f in family_df['family']], alpha=0.6)
    
    ax1.set_xlabel('Architectural Family', fontweight='bold')
    ax1.set_ylabel('Performance Score', fontweight='bold')
    ax1.set_title('Average Performance by Architectural Family', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(family_df['family'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # 2. Success rate comparison
    success_rates = []
    thresholds = [0.3, 0.4, 0.5, 0.6]
    
    for family, models in families.items():
        family_data = df[df['model'].isin(models)]
        rates = []
        for threshold in thresholds:
            rate = (family_data['f1_score'] >= threshold).mean() * 100
            rates.append(rate)
        success_rates.append(rates)
    
    for i, (family, rates) in enumerate(zip(families.keys(), success_rates)):
        ax2.plot(thresholds, rates, marker='o', linewidth=3, markersize=8,
                label=family, color=family_colors[family])
    
    ax2.set_xlabel('F1-Score Threshold', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontweight='bold')
    ax2.set_title('Success Rate by Architectural Family', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error pattern comparison
    error_patterns = []
    for family, models in families.items():
        family_data = df[df['model'].isin(models)]
        avg_fn = family_data['false_negatives'].mean()
        avg_fp = family_data['false_positives'].mean()
        avg_splits = family_data['splits'].mean()
        avg_merges = family_data['merges'].mean()
        error_patterns.append([avg_fn, avg_fp, avg_splits, avg_merges])
    
    error_patterns = np.array(error_patterns)
    
    x = np.arange(len(families))
    width = 0.2
    
    ax3.bar(x - 1.5*width, error_patterns[:, 0], width, label='False Negatives',
           color=PERFORMANCE_COLORS['failure'], alpha=0.8)
    ax3.bar(x - 0.5*width, error_patterns[:, 1], width, label='False Positives',
           color=PERFORMANCE_COLORS['poor'], alpha=0.8)
    ax3.bar(x + 0.5*width, error_patterns[:, 2], width, label='Splits',
           color=PERFORMANCE_COLORS['good'], alpha=0.8)
    ax3.bar(x + 1.5*width, error_patterns[:, 3], width, label='Merges',
           color=PERFORMANCE_COLORS['excellent'], alpha=0.8)
    
    ax3.set_xlabel('Architectural Family', fontweight='bold')
    ax3.set_ylabel('Average Error Count per Image', fontweight='bold')
    ax3.set_title('Error Patterns by Architectural Family', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(list(families.keys()), rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Robustness comparison (coefficient of variation)
    robustness_data = []
    for family, models in families.items():
        family_data = df[df['model'].isin(models)]
        cv = family_data['f1_score'].std() / family_data['f1_score'].mean()
        robustness_data.append({'family': family, 'cv': cv})
    
    robustness_df = pd.DataFrame(robustness_data)
    
    bars = ax4.bar(range(len(robustness_df)), robustness_df['cv'],
                  color=[family_colors[f] for f in robustness_df['family']], alpha=0.8)
    
    ax4.set_xlabel('Architectural Family', fontweight='bold')
    ax4.set_ylabel('Coefficient of Variation', fontweight='bold')
    ax4.set_title('Performance Consistency (Lower = More Robust)', fontweight='bold')
    ax4.set_xticks(range(len(robustness_df)))
    ax4.set_xticklabels(robustness_df['family'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'architectural_family_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created architectural_family_comparison.png")

def main():
    """Generate combined success/failure analysis figures"""
    print("Loading error analysis data...")
    
    df = load_error_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Loaded data for {len(df)} evaluations across {df['model'].nunique()} models")
    
    print("Creating combined success/failure analysis...")
    create_combined_success_failure_analysis(df)
    
    print("Creating architectural family comparison...")
    create_architectural_family_comparison(df)
    
    print(f"\n🎉 Combined analysis figures created successfully in: {output_dir}")
    print("Generated files:")
    for png_file in sorted(output_dir.glob("*combined*")) + sorted(output_dir.glob("*architectural*")):
        print(f"  - {png_file.name}")

if __name__ == "__main__":
    main()
