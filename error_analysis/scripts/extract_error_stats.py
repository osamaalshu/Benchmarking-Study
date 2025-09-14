#!/usr/bin/env python3
"""
Extract error statistics from detailed_results.json and calculate averages
"""

import json
import pandas as pd
from collections import defaultdict

def extract_error_stats():
    # Load the detailed results
    with open('results/error_analysis/detailed_results.json', 'r') as f:
        data = json.load(f)
    
    # Initialize counters for each model
    model_stats = defaultdict(lambda: {
        'false_negatives': [],
        'false_positives': [],
        'splits': [],
        'merges': [],
        'total_errors': [],
        'gt_count': [],
        'pred_count': []
    })
    
    # Process each image
    for image_name, image_data in data.items():
        # Skip nested entries (some images have duplicate entries)
        if isinstance(image_data, dict) and any(key in image_data for key in ['unet', 'nnunet', 'sac', 'lstmunet', 'maunet_resnet50', 'maunet_wide', 'maunet_ensemble']):
            for model_name, model_data in image_data.items():
                if model_name in ['unet', 'nnunet', 'sac', 'lstmunet', 'maunet_resnet50', 'maunet_wide', 'maunet_ensemble']:
                    # Extract error counts
                    fn = int(model_data.get('false_negatives', 0))
                    fp = int(model_data.get('false_positives', 0))
                    splits = int(model_data.get('splits', 0))
                    merges = int(model_data.get('merges', 0))
                    total = fn + fp + splits + merges
                    
                    # Store values
                    model_stats[model_name]['false_negatives'].append(fn)
                    model_stats[model_name]['false_positives'].append(fp)
                    model_stats[model_name]['splits'].append(splits)
                    model_stats[model_name]['merges'].append(merges)
                    model_stats[model_name]['total_errors'].append(total)
                    model_stats[model_name]['gt_count'].append(int(model_data.get('gt_count', 0)))
                    model_stats[model_name]['pred_count'].append(int(model_data.get('pred_count', 0)))
    
    # Calculate averages
    results = {}
    for model_name, stats in model_stats.items():
        if stats['false_negatives']:  # Only process models with data
            results[model_name] = {
                'false_negatives': round(sum(stats['false_negatives']) / len(stats['false_negatives']), 1),
                'false_positives': round(sum(stats['false_positives']) / len(stats['false_positives']), 1),
                'splits': round(sum(stats['splits']) / len(stats['splits']), 1),
                'merges': round(sum(stats['merges']) / len(stats['merges']), 1),
                'total_errors': round(sum(stats['total_errors']) / len(stats['total_errors']), 1),
                'avg_gt_count': round(sum(stats['gt_count']) / len(stats['gt_count']), 1),
                'num_images': len(stats['false_negatives'])
            }
    
    return results

def print_latex_table(results):
    """Print the results in LaTeX table format"""
    
    # Model name mapping for display
    model_display_names = {
        'maunet_ensemble': 'MAUNet-Ens',
        'maunet_wide': 'MAUNet-Wide',
        'maunet_resnet50': 'MAUNet-R50',
        'nnunet': 'nnU-Net',
        'unet': 'U-Net',
        'lstmunet': 'LSTM-UNet',
        'sac': 'SAC'
    }
    
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{Error analysis summary – average error counts per image (\\textit{Mean values across " + str(results['maunet_ensemble']['num_images']) + " test images with an average of " + str(results['maunet_ensemble']['avg_gt_count']) + " ground-truth cells per image})}")
    print("\\label{tab:error_analysis_counts}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{False Negatives} & \\textbf{False Positives} & \\textbf{Splits} & \\textbf{Merges} & \\textbf{Total Errors} \\\\")
    print("\\hline")
    
    # Sort models by total errors (ascending)
    sorted_models = sorted(results.items(), key=lambda x: x[1]['total_errors'])
    
    for model_name, stats in sorted_models:
        display_name = model_display_names.get(model_name, model_name)
        
        # Find the best performer for each metric to bold
        best_fn = min(r['false_negatives'] for r in results.values())
        best_fp = min(r['false_positives'] for r in results.values())
        best_splits = min(r['splits'] for r in results.values())
        best_merges = min(r['merges'] for r in results.values())
        best_total = min(r['total_errors'] for r in results.values())
        
        fn_str = f"\\textbf{{{stats['false_negatives']}}}" if stats['false_negatives'] == best_fn else str(stats['false_negatives'])
        fp_str = f"\\textbf{{{stats['false_positives']}}}" if stats['false_positives'] == best_fp else str(stats['false_positives'])
        splits_str = f"\\textbf{{{stats['splits']}}}" if stats['splits'] == best_splits else str(stats['splits'])
        merges_str = f"\\textbf{{{stats['merges']}}}" if stats['merges'] == best_merges else str(stats['merges'])
        total_str = f"\\textbf{{{stats['total_errors']}}}" if stats['total_errors'] == best_total else str(stats['total_errors'])
        
        print(f"{display_name} & {fn_str} & {fp_str} & {splits_str} & {merges_str} & {total_str} \\\\")
    
    print("\\hline")
    print("\\end{tabular}%")
    print("}")
    print("\\vspace{0.8em}")
    print("\\begin{minipage}{0.95\\textwidth}")
    print("\\small")
    print("Note: Splits indicate ground truth cells divided into multiple predictions (over-segmentation). Merges indicate multiple ground truth cells combined into a single prediction (under-segmentation).")
    print("\\end{minipage}")
    print("\\end{table}")

if __name__ == "__main__":
    results = extract_error_stats()
    print_latex_table(results)
    
    # Also print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS:")
    print("="*80)
    for model_name, stats in results.items():
        print(f"{model_name}:")
        print(f"  Images processed: {stats['num_images']}")
        print(f"  Average GT cells per image: {stats['avg_gt_count']}")
        print(f"  Average errors per image: {stats['total_errors']}")
        print(f"  Average splits per image: {stats['splits']}")
        print(f"  Average merges per image: {stats['merges']}")
        print()
