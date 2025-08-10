#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate performance comparison plots with MAUNet ensemble included
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_metrics_for_model(model_name, base_path):
    """Load metrics for a specific model across all thresholds"""
    metrics = {}
    for threshold in ['0.5', '0.7', '0.9']:
        file_path = os.path.join(base_path, f"{model_name}_metrics-{threshold}.csv")
        if os.path.exists(file_path):
            metrics[threshold] = pd.read_csv(file_path)
    return metrics

def create_performance_comparison_plot():
    """Create comprehensive performance comparison plot"""
    models = ['unet', 'nnunet', 'sac', 'lstmunet', 'maunet', 'maunet_ensemble']
    metrics_path = "./test_predictions"
    output_path = "./visualization_results"
    
    # Load metrics for all models
    all_metrics = {}
    for model in models:
        all_metrics[model] = load_metrics_for_model(model, metrics_path)
    
    # Create comparison data
    comparison_data = []
    for model in models:
        if '0.5' in all_metrics[model]:
            df = all_metrics[model]['0.5']
            comparison_data.append({
                'Model': model.upper(),
                'F1': df['F1'].mean(),
                'Dice': df['dice'].mean(),
                'Precision': df['precision'].mean(),
                'Recall': df['recall'].mean()
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # F1 Score
    bars1 = ax1.bar(comparison_df['Model'], comparison_df['F1'], color='skyblue', alpha=0.8)
    ax1.set_title('F1 Score Comparison (Threshold=0.5)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1 Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Dice Score
    bars2 = ax2.bar(comparison_df['Model'], comparison_df['Dice'], color='lightcoral', alpha=0.8)
    ax2.set_title('Dice Score Comparison (Threshold=0.5)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Dice Score')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Precision
    bars3 = ax3.bar(comparison_df['Model'], comparison_df['Precision'], color='lightgreen', alpha=0.8)
    ax3.set_title('Precision Comparison (Threshold=0.5)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Precision')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Recall
    bars4 = ax4.bar(comparison_df['Model'], comparison_df['Recall'], color='gold', alpha=0.8)
    ax4.set_title('Recall Comparison (Threshold=0.5)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Recall')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Performance comparison plot saved to: {output_path}/performance_comparison.png")
    return comparison_df

def create_f1_threshold_comparison_plot():
    """Create F1 score comparison across different thresholds"""
    models = ['unet', 'nnunet', 'sac', 'lstmunet', 'maunet', 'maunet_ensemble']
    metrics_path = "./test_predictions"
    output_path = "./visualization_results"
    thresholds = ['0.5', '0.7', '0.9']
    
    # Load metrics for all models
    all_metrics = {}
    for model in models:
        all_metrics[model] = load_metrics_for_model(model, metrics_path)
    
    # Create comparison data
    comparison_data = []
    for model in models:
        for threshold in thresholds:
            if threshold in all_metrics[model]:
                df = all_metrics[model][threshold]
                comparison_data.append({
                    'Model': model.upper(),
                    'Threshold': threshold,
                    'F1': df['F1'].mean(),
                    'Std': df['F1'].std()
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, model in enumerate(models):
        model_data = comparison_df[comparison_df['Model'] == model.upper()]
        if not model_data.empty:
            plt.errorbar(model_data['Threshold'], model_data['F1'], 
                        yerr=model_data['Std'], marker='o', linewidth=2, 
                        markersize=8, label=model.upper(), color=colors[i])
    
    plt.xlabel('IoU Threshold', fontsize=12)
    plt.ylabel('Mean F1 Score', fontsize=12)
    plt.title('F1 Score Performance Across Different IoU Thresholds', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, 'f1_threshold_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ F1 threshold comparison plot saved to: {output_path}/f1_threshold_comparison.png")

def create_training_comparison_plot():
    """Create training information comparison plot"""
    # Load training information
    training_info = {
        'UNET': {'epochs': 58, 'final_loss': 0.7364, 'val_dice': 0.6130},
        'NNUNET': {'epochs': 86, 'final_loss': 0.5139, 'val_dice': 0.6744},
        'SAC': {'epochs': 52, 'final_loss': 1.3622, 'val_dice': 0.2128},
        'LSTMUNET': {'epochs': 39, 'final_loss': 0.9203, 'val_dice': 0.5898},
        'MAUNET': {'epochs': 194, 'final_loss': 0.3911, 'val_dice': None},
        'MAUNET_ENSEMBLE': {'epochs': None, 'final_loss': None, 'val_dice': None}
    }
    
    # Create comparison data
    comparison_data = []
    for model, info in training_info.items():
        if info['epochs'] is not None:
            comparison_data.append({
                'Model': model,
                'Epochs': info['epochs'],
                'Final Loss': info['final_loss'],
                'Val Dice': info['val_dice'] if info['val_dice'] is not None else 0
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training epochs
    bars1 = ax1.bar(comparison_df['Model'], comparison_df['Epochs'], color='lightblue', alpha=0.8)
    ax1.set_title('Training Epochs', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Epochs')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Final loss
    bars2 = ax2.bar(comparison_df['Model'], comparison_df['Final Loss'], color='lightcoral', alpha=0.8)
    ax2.set_title('Final Training Loss', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Loss Value')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join("./visualization_results", 'training_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training comparison plot saved to: ./visualization_results/training_comparison.png")

def main():
    print("Regenerating performance comparison plots with MAUNet ensemble...")
    
    # Create output directory
    os.makedirs("./visualization_results", exist_ok=True)
    
    # Generate all plots
    create_performance_comparison_plot()
    create_f1_threshold_comparison_plot()
    create_training_comparison_plot()
    
    print("\n✅ All performance comparison plots regenerated with MAUNet ensemble included!")

if __name__ == "__main__":
    main()
