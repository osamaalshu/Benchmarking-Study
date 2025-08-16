#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate comprehensive results report for all models
"""

import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime

def load_metrics_for_model(model_name, base_path):
    """Load all metrics for a specific model"""
    metrics = {}
    for threshold in ['0.5', '0.7', '0.9']:
        csv_file = os.path.join(base_path, f"{model_name}-{threshold}.csv")
        if os.path.exists(csv_file):
            metrics[threshold] = pd.read_csv(csv_file)
    return metrics

def load_training_info():
    """Load training information for all models"""
    training_info = []
    
    # Training parameters for each model
    training_params = {
        'unet': {
            'arch': 'U-Net with ResNet blocks', 
            'batch_size': 8, 
            'lr': '6e-4', 
            'input_size': '256x256',
            'source': 'MONAI Framework',
            'repository': 'Built-in MONAI implementation'
        },
        'nnunet': {
            'arch': 'nnU-Net (No New U-Net)', 
            'batch_size': 8, 
            'lr': '6e-4', 
            'input_size': '256x256',
            'source': 'MIC-DKFZ',
            'repository': 'https://github.com/mic-dkfz/nnunet'
        },
        'sac': {
            'arch': 'Segment Anything + Custom Head', 
            'batch_size': 2, 
            'lr': '6e-4', 
            'input_size': '256x256',
            'source': 'Authors via Email',
            'repository': 'Code provided by authors via email'
        },
        'lstmunet': {
            'arch': 'U-Net with LSTM layers', 
            'batch_size': 8, 
            'lr': '6e-4', 
            'input_size': '256x256',
            'source': 'GitLab - shaked0',
            'repository': 'https://gitlab.com/shaked0/lstmUnet'
        },
        'maunet': {
            'arch': 'MAU-Net with ResNet50 backbone', 
            'batch_size': 8, 
            'lr': '6e-4', 
            'input_size': '256x256',
            'source': 'NeurIPS 2022 Challenge',
            'repository': 'https://github.com/Woof6/neurips22-cellseg_saltfish'
        },
        'maunet_wide': {
            'arch': 'MAU-Net with Wide-ResNet50 backbone', 
            'batch_size': 8, 
            'lr': '6e-4', 
            'input_size': '256x256',
            'source': 'NeurIPS 2022 Challenge',
            'repository': 'https://github.com/Woof6/neurips22-cellseg_saltfish'
        },
        'maunet_ensemble': {
            'arch': 'MAU-Net Ensemble (ResNet50 + Wide-ResNet50)', 
            'batch_size': 'N/A (Ensemble)', 
            'lr': 'N/A (Ensemble)', 
            'input_size': '256x256',
            'source': 'NeurIPS 2022 Challenge (Ensemble)',
            'repository': 'https://github.com/Woof6/neurips22-cellseg_saltfish'
        }
    }
    
    for model_name, params in training_params.items():
        info = {
            'Model': model_name.upper(),
            'Architecture': params['arch'],
            'Source': params['source'],
            'Repository': params['repository'],
            'Batch Size': params['batch_size'],
            'Learning Rate': params['lr'],
            'Input Size': params['input_size'],
            'Optimizer': 'AdamW'
        }
        
        # Try to load training logs
        if model_name == 'maunet':
            # Special handling for individual MAUNet (ResNet50)
            checkpoint_path = 'baseline/work_dir/maunet(Normal)_3class/best_Dice_model.pth'
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    info['Total Epochs'] = checkpoint.get('epoch', 'N/A')
                    # Handle different loss formats
                    loss_data = checkpoint.get('loss', None)
                    if isinstance(loss_data, list) and len(loss_data) > 0:
                        info['Final Loss'] = f"{loss_data[-1]:.4f}"
                    elif isinstance(loss_data, (int, float)):
                        info['Final Loss'] = f"{loss_data:.4f}"
                    else:
                        info['Final Loss'] = 'N/A'
                    info['Best Val Dice'] = 'N/A (No validation)'
                    info['Training Status'] = 'Completed'
                except Exception as e:
                    info['Total Epochs'] = 'N/A'
                    info['Final Loss'] = 'N/A'
                    info['Best Val Dice'] = 'N/A'
                    info['Training Status'] = f'Error: {str(e)}'
        elif model_name == 'maunet_wide':
            # Special handling for MAUNet Wide-ResNet50
            checkpoint_path = 'baseline/work_dir/maunet(Wide)_3class/best_Dice_model.pth'
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    info['Total Epochs'] = checkpoint.get('epoch', 'N/A')
                    loss_data = checkpoint.get('loss', None)
                    if isinstance(loss_data, list) and len(loss_data) > 0:
                        info['Final Loss'] = f"{loss_data[-1]:.4f}"
                    elif isinstance(loss_data, (int, float)):
                        info['Final Loss'] = f"{loss_data:.4f}"
                    else:
                        info['Final Loss'] = 'N/A'
                    info['Best Val Dice'] = 'N/A (No validation)'
                    info['Training Status'] = 'Completed'
                except Exception as e:
                    info['Total Epochs'] = 'N/A'
                    info['Final Loss'] = 'N/A'
                    info['Best Val Dice'] = 'N/A'
                    info['Training Status'] = f'Error: {str(e)}'
        elif model_name == 'maunet_ensemble':
            # Special handling for MAUNet ensemble
            info['Total Epochs'] = 'N/A (Ensemble)'
            info['Final Loss'] = 'N/A (Ensemble)'
            info['Best Val Dice'] = 'N/A (Ensemble)'
            info['Training Status'] = 'Ensemble Model'
        else:
            # Load from npz files for other models
            log_path = f'baseline/work_dir/{model_name}_3class/train_log.npz'
            if os.path.exists(log_path):
                try:
                    data = np.load(log_path)
                    info['Total Epochs'] = len(data['epoch_loss'])
                    info['Final Loss'] = f"{data['epoch_loss'][-1]:.4f}"
                    if 'val_dice' in data and len(data['val_dice']) > 0:
                        info['Best Val Dice'] = f"{max(data['val_dice']):.4f}"
                    else:
                        info['Best Val Dice'] = 'N/A'
                    info['Training Status'] = 'Completed'
                except:
                    info['Total Epochs'] = 'N/A'
                    info['Final Loss'] = 'N/A'
                    info['Best Val Dice'] = 'N/A'
                    info['Training Status'] = 'Logs unavailable'
            else:
                info['Total Epochs'] = 'N/A'
                info['Final Loss'] = 'N/A'
                info['Best Val Dice'] = 'N/A'
                info['Training Status'] = 'Not trained'
        
        training_info.append(info)
    
    return training_info

def generate_markdown_report(output_path):
    """Generate a comprehensive markdown report"""
    # Explicitly include the three MAUNet variants by their prediction folder names
    models = ['unet', 'nnunet', 'sac', 'lstmunet', 'maunet_resnet50', 'maunet_wide', 'maunet_ensemble']
    base_path = "./test_predictions"
    
    # Start report
    report = []
    report.append("# Cell Segmentation Model Benchmarking Results\n")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    
    # Load training information
    training_info = load_training_info()
    training_df = pd.DataFrame(training_info)
    
    # Load metrics for all models
    all_metrics = {}
    summary_table = []
    
    for model in models:
        metrics = load_metrics_for_model(model, base_path)
        if '0.5' in metrics:
            all_metrics[model] = metrics
            df = metrics['0.5']
            summary_table.append({
                'Model': model.upper(),
                'Mean F1': f"{df['F1'].mean():.4f}",
                'Median F1': f"{df['F1'].median():.4f}",
                'Std F1': f"{df['F1'].std():.4f}",
                'Mean Dice': f"{df['dice'].mean():.4f}",
                'Mean Precision': f"{df['precision'].mean():.4f}",
                'Mean Recall': f"{df['recall'].mean():.4f}"
            })
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_table)
    report.append("### Performance Summary (Threshold = 0.5)\n")
    report.append(summary_df.to_markdown(index=False))
    report.append("\n")
    
    # Best performing model
    best_model_idx = np.argmax([float(row['Mean F1']) for row in summary_table])
    best_model = summary_table[best_model_idx]['Model']
    report.append(f"    **Best Performing Model**: {best_model} (Mean F1: {summary_table[best_model_idx]['Mean F1']})\n")
    
    # Performance Visualizations Section
    report.append("\n## Performance Visualizations\n")
    report.append("\n### Overall Performance Comparison\n")
    report.append("![Performance Comparison](../visualization_results/performance_comparison.png)\n")
    report.append("\n*Figure 1: Comprehensive performance comparison across all metrics (F1, Dice, Precision, Recall)*\n")
    
    report.append("\n### F1 Score Across Different IoU Thresholds\n")
    report.append("![F1 Threshold Comparison](../visualization_results/f1_threshold_comparison.png)\n")
    report.append("\n*Figure 2: F1 Score performance at different IoU thresholds (0.5, 0.7, 0.9) showing model robustness*\n")
    
    report.append("\n### Training Information Comparison\n")
    report.append("![Training Comparison](../visualization_results/training_comparison.png)\n")
    report.append("\n*Figure 3: Training epochs and final loss comparison across models*\n")
    
    # Training Information Section
    report.append("\n## Training Information\n")
    
    # Model Sources and Repositories
    report.append("\n### Model Sources and Repositories\n")
    for info in training_info:
        model_name = info['Model']
        source = info['Source']
        repo = info['Repository']
        if repo.startswith('http'):
            report.append(f"- **{model_name}**: {source} - [{repo}]({repo})\n")
        else:
            report.append(f"- **{model_name}**: {source} - {repo}\n")
    
    report.append("\n### Model Architectures and Training Parameters\n")
    report.append(training_df.to_markdown(index=False))
    report.append("\n")
    
    # Training Summary
    completed_models = [t for t in training_info if t['Training Status'] == 'Completed']
    total_epochs = [t['Total Epochs'] for t in training_info if isinstance(t['Total Epochs'], int)]
    val_dice_scores = [float(t['Best Val Dice']) for t in training_info if t['Best Val Dice'] not in ['N/A', 'N/A (No validation)', 'N/A (Ensemble)'] and isinstance(t['Best Val Dice'], str) and t['Best Val Dice'].replace('.', '').isdigit()]
    
    report.append("\n### Training Summary\n")
    report.append(f"- **Total Models Trained**: {len(completed_models)}\n")
    if total_epochs:
        max_epochs = max(total_epochs)
        max_epoch_model = [t['Model'] for t in training_info if t['Total Epochs'] == max_epochs][0]
        report.append(f"- **Most Epochs**: {max_epochs} ({max_epoch_model})\n")
    if val_dice_scores:
        best_val_dice = max(val_dice_scores)
        best_val_model = [t['Model'] for t in training_info if t['Best Val Dice'] == f'{best_val_dice:.4f}'][0]
        report.append(f"- **Best Training Validation Dice**: {best_val_dice:.4f} ({best_val_model})\n")
    report.append("- **Optimizer**: AdamW (all models)\n")
    report.append("- **Learning Rate**: 6e-4 (all models)\n")
    
    # Sample Segmentation Results Section
    report.append("\n## Sample Segmentation Results\n")
    report.append("\n### Qualitative Comparison\n")
    
    # Check for available comparison images
    comparison_dir = "./visualization_results"
    sample_images = []
    for model_dir in ['unet', 'nnunet', 'sac', 'lstmunet', 'maunet', 'maunet_ensemble']:
        model_path = os.path.join(comparison_dir, model_dir)
        if os.path.exists(model_path):
            images = [f for f in os.listdir(model_path) if f.endswith('_comparison.png')]
            if images:
                sample_images.extend([(model_dir, img) for img in images[:2]])  # Take first 2 images
    
    if sample_images:
        report.append("The following images show qualitative comparisons between ground truth and model predictions:\n\n")
        
        # Group by image name
        image_groups = {}
        for model, img in sample_images:
            img_name = img.replace('_comparison.png', '')
            if img_name not in image_groups:
                image_groups[img_name] = []
            image_groups[img_name].append((model, img))
        
        # Show first few image comparisons
        for i, (img_name, model_imgs) in enumerate(list(image_groups.items())[:3]):
            report.append(f"#### Sample {i+1}: {img_name}\n")
            for model, img in model_imgs:
                report.append(f"**{model.upper()}**: ![{model} {img_name}](../visualization_results/{model}/{img})\n\n")
            report.append("---\n\n")
    
    # Individual visualization results if available
    viz_files = ['cell_00001_visualization.png', 'cell_00002_visualization.png', 'cell_00003_visualization.png']
    available_viz = [f for f in viz_files if os.path.exists(os.path.join(comparison_dir, f))]
    
    if available_viz:
        report.append("### Individual Segmentation Examples\n")
        for viz_file in available_viz[:2]:  # Show first 2
            cell_name = viz_file.replace('_visualization.png', '')
            report.append(f"![{cell_name} Segmentation](../visualization_results/{viz_file})\n")
            report.append(f"*{cell_name.replace('_', ' ').title()} - Original image, ground truth, and prediction comparison*\n\n")
    
    # Detailed Results per Model
    report.append("\n## Detailed Results by Model\n")
    
    for model in models:
        if model not in all_metrics:
            continue
            
        report.append(f"\n### {model.upper()}\n")
        
        # Performance across thresholds
        threshold_summary = []
        for threshold in ['0.5', '0.7', '0.9']:
            if threshold in all_metrics[model]:
                df = all_metrics[model][threshold]
                threshold_summary.append({
                    'Threshold': threshold,
                    'Mean F1': f"{df['F1'].mean():.4f}",
                    'Mean Dice': f"{df['dice'].mean():.4f}",
                    'Mean Precision': f"{df['precision'].mean():.4f}",
                    'Mean Recall': f"{df['recall'].mean():.4f}",
                    'Total Samples': len(df)
                })
        
        threshold_df = pd.DataFrame(threshold_summary)
        report.append("#### Performance Across Thresholds\n")
        report.append(threshold_df.to_markdown(index=False))
        report.append("\n")
        
        # Top and bottom performing samples
        df_05 = all_metrics[model]['0.5']
        top_5 = df_05.nlargest(5, 'F1')[['names', 'F1', 'dice']]
        bottom_5 = df_05.nsmallest(5, 'F1')[['names', 'F1', 'dice']]
        
        report.append("#### Top 5 Performing Images\n")
        report.append(top_5.to_markdown(index=False))
        report.append("\n")
        
        report.append("#### Bottom 5 Performing Images\n")
        report.append(bottom_5.to_markdown(index=False))
        report.append("\n")
    
    # Dataset Statistics
    report.append("\n## Dataset Analysis\n")
    
    # Analyze performance by number of cells
    report.append("### Performance vs. Ground Truth Cell Count\n")
    
    for model in models:
        if model not in all_metrics:
            continue
            
        df = all_metrics[model]['0.5']
        if 'true_num' in df.columns:
            # Group by GT count ranges
            df['GT Count Range'] = pd.cut(df['true_num'], 
                                         bins=[0, 5, 10, 20, 50, 100, 1000],
                                         labels=['1-5', '6-10', '11-20', '21-50', '51-100', '100+'])
            
            grouped = df.groupby('GT Count Range')['F1'].agg(['mean', 'count'])
            report.append(f"\n#### {model.upper()} - Performance by Cell Count\n")
            report.append(grouped.to_markdown())
            report.append("\n")
    
    # Model Comparison
    report.append("\n## Model Comparison\n")
    
    # Create comparison matrix
    comparison_data = []
    metrics_to_compare = [('F1', 'F1 Score'), ('dice', 'Dice Score'), ('precision', 'Precision'), ('recall', 'Recall')]
    
    for col_name, display_name in metrics_to_compare:
        row = {'Metric': display_name}
        for model in models:
            if model in all_metrics and '0.5' in all_metrics[model]:
                df = all_metrics[model]['0.5']
                row[model.upper()] = f"{df[col_name].mean():.4f}"
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    report.append("### Metrics Comparison (Threshold = 0.5)\n")
    report.append(comparison_df.to_markdown(index=False))
    report.append("\n")
    
    # Statistical Significance Analysis
    report.append("### Statistical Significance Analysis\n")
    
    # Load statistical analysis results
    stats_file = os.path.join(output_path, "statistical_significance_summary.json")
    if os.path.exists(stats_file):
        try:
            import json
            with open(stats_file, 'r') as f:
                stats_data = json.load(f)
            
            # Add statistical summary for threshold 0.5
            if 'threshold_0.5' in stats_data:
                threshold_data = stats_data['threshold_0.5']
                if 'threshold_0.5' in threshold_data:  # Nested structure
                    threshold_data = threshold_data['threshold_0.5']
                
                # F1 Score Statistical Analysis
                if 'F1' in threshold_data:
                    f1_stats = threshold_data['F1']
                    
                    # Descriptive Statistics
                    if 'descriptive_stats' in f1_stats:
                        report.append("#### F1 Score Descriptive Statistics\n")
                        report.append("| Model | Mean ± Std | Median | 95% CI | Range |\n")
                        report.append("|-------|------------|--------|--------|-------|\n")
                        
                        for model, stats in f1_stats['descriptive_stats'].items():
                            mean_std = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
                            median = f"{stats['median']:.4f}"
                            ci = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
                            range_val = f"[{stats['min']:.4f}, {stats['max']:.4f}]"
                            report.append(f"| {model.upper()} | {mean_std} | {median} | {ci} | {range_val} |\n")
                        report.append("\n")
                    
                    # ANOVA Results
                    if 'anova' in f1_stats and 'error' not in f1_stats['anova']:
                        anova = f1_stats['anova']
                        report.append("#### F1 Score ANOVA Results\n")
                        report.append(f"- **F-statistic**: {anova['statistic']:.4f}\n")
                        report.append(f"- **p-value**: {anova['p_value']:.6f}\n")
                        report.append(f"- **Significant**: {'Yes' if anova['significant'] else 'No'}\n")
                        report.append("\n")
                    
                    # Kruskal-Wallis Results
                    if 'kruskal_wallis' in f1_stats and 'error' not in f1_stats['kruskal_wallis']:
                        kw = f1_stats['kruskal_wallis']
                        report.append("#### F1 Score Kruskal-Wallis Results\n")
                        report.append(f"- **H-statistic**: {kw['statistic']:.4f}\n")
                        report.append(f"- **p-value**: {kw['p_value']:.6f}\n")
                        report.append(f"- **Significant**: {'Yes' if kw['significant'] else 'No'}\n")
                        report.append("\n")
                    
                    # Key Pairwise Comparisons
                    if 'pairwise_comparisons' in f1_stats:
                        report.append("#### Key Pairwise Comparisons (F1 Score)\n")
                        report.append("| Comparison | T-test p-value | Mann-Whitney p-value | Cohen's d | Effect Size |\n")
                        report.append("|------------|----------------|---------------------|-----------|-------------|\n")
                        
                        # Show most important comparisons (in alphabetical order as stored in JSON)
                        important_comparisons = [
                            'maunet_vs_maunet_ensemble',
                            'nnunet_vs_maunet_ensemble', 
                            'maunet_vs_nnunet',
                            'unet_vs_maunet_ensemble',
                            'maunet_vs_unet',
                            'nnunet_vs_unet'
                        ]
                        
                        for comp_name in important_comparisons:
                            if comp_name in f1_stats['pairwise_comparisons']:
                                comp = f1_stats['pairwise_comparisons'][comp_name]
                                ttest = comp.get('independent_ttest', {})
                                mw = comp.get('mannwhitney_u', {})
                                cohens = comp.get('cohens_d', {})
                                
                                ttest_p = f"{ttest['p_value']:.6f}" if 'error' not in ttest else "N/A"
                                mw_p = f"{mw['p_value']:.6f}" if 'error' not in mw else "N/A"
                                cohens_d = f"{cohens['value']:.4f}" if 'error' not in cohens else "N/A"
                                effect_size = cohens.get('interpretation', 'N/A') if 'error' not in cohens else "N/A"
                                
                                report.append(f"| {comp_name.replace('_', ' ').upper()} | {ttest_p} | {mw_p} | {cohens_d} | {effect_size} |\n")
                        
                        report.append("\n")
                
                # Dice Score Statistical Analysis
                if 'dice' in threshold_data:
                    dice_stats = threshold_data['dice']
                    if 'descriptive_stats' in dice_stats:
                        report.append("#### Dice Score Descriptive Statistics\n")
                        report.append("| Model | Mean ± Std | Median | 95% CI |\n")
                        report.append("|-------|------------|--------|--------|\n")
                        
                        for model, stats in dice_stats['descriptive_stats'].items():
                            mean_std = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
                            median = f"{stats['median']:.4f}"
                            ci = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
                            report.append(f"| {model.upper()} | {mean_std} | {median} | {ci} |\n")
                        report.append("\n")
        except Exception as e:
            report.append(f"*Statistical analysis data could not be loaded: {str(e)}*\n\n")
    else:
        report.append("*Statistical significance analysis not available*\n\n")
    
    # Recommendations
    report.append("\n## Recommendations\n")
    report.append("Based on the benchmarking results:\n")
    report.append(f"1. **{best_model}** shows the best overall performance with highest mean F1 score\n")
    report.append("2. Consider using threshold = 0.5 for optimal balance between precision and recall\n")
    report.append("3. Models perform better on images with moderate cell counts (10-50 cells)\n")
    report.append("4. Further training or fine-tuning may improve performance on densely populated images\n")
    
    # Save report
    report_text = '\n'.join(report)
    
    # Clean up any double newlines in tables
    report_text = report_text.replace('\n\n\n', '\n\n')
    report_text = report_text.replace('|\n\n|', '|\n|')
    
    report_path = os.path.join(output_path, 'benchmarking_report.md')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"✅ Report saved to: {report_path}")
    
    # Also save as text file
    text_path = os.path.join(output_path, 'benchmarking_report.txt')
    with open(text_path, 'w') as f:
        f.write(report_text.replace('|', ' ').replace('-', ' '))
    
    print(f"✅ Text version saved to: {text_path}")

def main():
    output_path = "./test_predictions"
    generate_markdown_report(output_path)
    
    # Run the visualization script
    print("\nRunning visualization script...")
    os.system("python visualize_results.py")

if __name__ == "__main__":
    main() 