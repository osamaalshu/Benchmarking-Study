#!/usr/bin/env python3
"""
Extract PQ decomposition metrics from detailed_results.json and use benchmark F1, Precision, Recall
"""

import json
from collections import defaultdict

def extract_pq_metrics():
    # Load the detailed results
    with open('results/error_analysis/detailed_results.json', 'r') as f:
        data = json.load(f)
    
    # Initialize counters for each model
    model_stats = defaultdict(lambda: {
        'pq': [],
        'rq': [],
        'sq': []
    })
    
    # Process each image
    for image_name, image_data in data.items():
        # Skip nested entries (some images have duplicate entries)
        if isinstance(image_data, dict) and any(key in image_data for key in ['unet', 'nnunet', 'sac', 'lstmunet', 'maunet_resnet50', 'maunet_wide', 'maunet_ensemble']):
            for model_name, model_data in image_data.items():
                if model_name in ['unet', 'nnunet', 'sac', 'lstmunet', 'maunet_resnet50', 'maunet_wide', 'maunet_ensemble']:
                    # Extract metrics
                    pq = float(model_data.get('PQ', 0))
                    rq = float(model_data.get('RQ', 0))
                    sq = float(model_data.get('SQ', 0))
                    
                    # Store values
                    model_stats[model_name]['pq'].append(pq)
                    model_stats[model_name]['rq'].append(rq)
                    model_stats[model_name]['sq'].append(sq)
    
    # Calculate averages
    results = {}
    for model_name, stats in model_stats.items():
        if stats['pq']:  # Only process models with data
            results[model_name] = {
                'pq': round(sum(stats['pq']) / len(stats['pq']), 3),
                'rq': round(sum(stats['rq']) / len(stats['rq']), 3),
                'sq': round(sum(stats['sq']) / len(stats['sq']), 3),
                'num_images': len(stats['pq'])
            }
    
    return results

def get_benchmark_metrics():
    """Get F1, Precision, and Recall values from benchmark results"""
    benchmark_metrics = {
        'unet': {'f1': 0.3341, 'precision': 0.3242, 'recall': 0.3854},
        'nnunet': {'f1': 0.3833, 'precision': 0.3619, 'recall': 0.4808},
        'sac': {'f1': 0.0037, 'precision': 0.0067, 'recall': 0.0107},
        'lstmunet': {'f1': 0.2889, 'precision': 0.2584, 'recall': 0.4549},
        'maunet_resnet50': {'f1': 0.5685, 'precision': 0.5722, 'recall': 0.5803},
        'maunet_wide': {'f1': 0.5561, 'precision': 0.5445, 'recall': 0.5985},
        'maunet_ensemble': {'f1': 0.6015, 'precision': 0.6654, 'recall': 0.5638}
    }
    return benchmark_metrics

def print_latex_table(results, benchmark_metrics):
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
    print("\\caption{Panoptic Quality (PQ) decomposition into Recognition Quality (RQ), Segmentation Quality (SQ), and classification metrics (F1, Precision, Recall).}")
    print("\\label{tab:pq_decomposition}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{PQ} & \\textbf{RQ (Detection)} & \\textbf{SQ (Segmentation)} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall} \\\\")
    print("\\hline")
    
    # Sort models by PQ (descending)
    sorted_models = sorted(results.items(), key=lambda x: x[1]['pq'], reverse=True)
    
    for model_name, stats in sorted_models:
        display_name = model_display_names.get(model_name, model_name)
        
        # Get benchmark metrics
        benchmark = benchmark_metrics.get(model_name, {'f1': 0, 'precision': 0, 'recall': 0})
        
        # Find the best performer for each metric to bold
        best_pq = max(r['pq'] for r in results.values())
        best_rq = max(r['rq'] for r in results.values())
        best_sq = max(r['sq'] for r in results.values())
        best_f1 = max(b['f1'] for b in benchmark_metrics.values())
        best_precision = max(b['precision'] for b in benchmark_metrics.values())
        best_recall = max(b['recall'] for b in benchmark_metrics.values())
        
        pq_str = f"\\textbf{{{stats['pq']}}}" if stats['pq'] == best_pq else str(stats['pq'])
        rq_str = f"\\textbf{{{stats['rq']}}}" if stats['rq'] == best_rq else str(stats['rq'])
        sq_str = f"\\textbf{{{stats['sq']}}}" if stats['sq'] == best_sq else str(stats['sq'])
        f1_str = f"\\textbf{{{benchmark['f1']}}}" if benchmark['f1'] == best_f1 else str(benchmark['f1'])
        precision_str = f"\\textbf{{{benchmark['precision']}}}" if benchmark['precision'] == best_precision else str(benchmark['precision'])
        recall_str = f"\\textbf{{{benchmark['recall']}}}" if benchmark['recall'] == best_recall else str(benchmark['recall'])
        
        print(f"{display_name} & {pq_str} & {rq_str} & {sq_str} & {f1_str} & {precision_str} & {recall_str} \\\\")
    
    print("\\hline")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}")

if __name__ == "__main__":
    results = extract_pq_metrics()
    benchmark_metrics = get_benchmark_metrics()
    print_latex_table(results, benchmark_metrics)
    
    # Also print summary statistics
    print("\n" + "="*80)
    print("PQ DECOMPOSITION SUMMARY:")
    print("="*80)
    for model_name, stats in results.items():
        benchmark = benchmark_metrics.get(model_name, {'f1': 0, 'precision': 0, 'recall': 0})
        print(f"{model_name}:")
        print(f"  Images processed: {stats['num_images']}")
        print(f"  PQ: {stats['pq']}")
        print(f"  RQ: {stats['rq']}")
        print(f"  SQ: {stats['sq']}")
        print(f"  F1 (benchmark): {benchmark['f1']}")
        print(f"  Precision (benchmark): {benchmark['precision']}")
        print(f"  Recall (benchmark): {benchmark['recall']}")
        print()
