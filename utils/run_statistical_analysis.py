#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to run statistical significance tests on existing evaluation results
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from compute_metric import perform_cross_model_statistical_analysis

def generate_statistical_summary_report(analysis_results, output_dir):
    """Generate a comprehensive summary report of all statistical analyses"""
    print(f"\n{'='*80}")
    print(f"GENERATING STATISTICAL SUMMARY REPORT")
    print(f"{'='*80}")
    
    report_lines = []
    report_lines.append("# Statistical Significance Analysis Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for threshold_key, threshold_results in analysis_results.items():
        if not threshold_results:
            continue
            
        threshold = threshold_key.split('_')[1]
        report_lines.append(f"## Threshold {threshold}")
        report_lines.append("-" * 40)
        report_lines.append("")
        
        for metric in ['F1', 'precision', 'recall', 'dice']:
            if metric in threshold_results:
                report_lines.append(f"### {metric.upper()} Metric Analysis")
                report_lines.append("")
                
                # Descriptive statistics summary
                desc_stats = threshold_results[metric].get('descriptive_stats', {})
                if desc_stats:
                    report_lines.append("#### Descriptive Statistics")
                    report_lines.append("")
                    report_lines.append("| Model | Mean ± Std | Median | 95% CI | Range |")
                    report_lines.append("|-------|------------|--------|--------|-------|")
                    
                    for model_name, stats in desc_stats.items():
                        mean_std = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
                        median = f"{stats['median']:.4f}"
                        ci = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
                        range_val = f"[{stats['min']:.4f}, {stats['max']:.4f}]"
                        report_lines.append(f"| {model_name} | {mean_std} | {median} | {ci} | {range_val} |")
                    report_lines.append("")
                
                # ANOVA results
                anova_result = threshold_results[metric].get('anova', {})
                if 'error' not in anova_result and anova_result:
                    report_lines.append("#### ANOVA Results")
                    report_lines.append("")
                    report_lines.append(f"- **F-statistic**: {anova_result['statistic']:.4f}")
                    report_lines.append(f"- **p-value**: {anova_result['p_value']:.6f}")
                    report_lines.append(f"- **Significant**: {'Yes' if anova_result['significant'] else 'No'}")
                    report_lines.append("")
                
                # Kruskal-Wallis results
                kw_result = threshold_results[metric].get('kruskal_wallis', {})
                if 'error' not in kw_result and kw_result:
                    report_lines.append("#### Kruskal-Wallis Results")
                    report_lines.append("")
                    report_lines.append(f"- **H-statistic**: {kw_result['statistic']:.4f}")
                    report_lines.append(f"- **p-value**: {kw_result['p_value']:.6f}")
                    report_lines.append(f"- **Significant**: {'Yes' if kw_result['significant'] else 'No'}")
                    report_lines.append("")
                
                # Pairwise comparisons
                pairwise_results = threshold_results[metric].get('pairwise_comparisons', {})
                if pairwise_results:
                    report_lines.append("#### Pairwise Comparisons")
                    report_lines.append("")
                    report_lines.append("| Comparison | T-test p-value | Mann-Whitney p-value | Cohen's d | Effect Size |")
                    report_lines.append("|------------|----------------|---------------------|-----------|-------------|")
                    
                    for comparison_name, comparison_result in pairwise_results.items():
                        ttest_result = comparison_result.get('independent_ttest', {})
                        mw_result = comparison_result.get('mannwhitney_u', {})
                        cohens_result = comparison_result.get('cohens_d', {})
                        
                        ttest_p = f"{ttest_result['p_value']:.6f}" if 'error' not in ttest_result else "N/A"
                        mw_p = f"{mw_result['p_value']:.6f}" if 'error' not in mw_result else "N/A"
                        cohens_d = f"{cohens_result['value']:.4f}" if 'error' not in cohens_result else "N/A"
                        effect_size = cohens_result.get('interpretation', 'N/A') if 'error' not in cohens_result else "N/A"
                        
                        report_lines.append(f"| {comparison_name} | {ttest_p} | {mw_p} | {cohens_d} | {effect_size} |")
                    report_lines.append("")
                
                report_lines.append("---")
                report_lines.append("")
    
    # Save the report
    report_file = os.path.join(output_dir, "statistical_significance_report.md")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Statistical summary report saved to: {report_file}")
    
    # Also save as JSON for programmatic access
    json_file = os.path.join(output_dir, "statistical_significance_summary.json")
    with open(json_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"✅ Statistical summary JSON saved to: {json_file}")
    
    return report_file

def main():
    parser = argparse.ArgumentParser(description='Run statistical significance tests on existing evaluation results')
    parser.add_argument('--metrics_dir', type=str, default='./test_predictions',
                       help='Directory containing metric CSV files')
    parser.add_argument('--output_dir', type=str, default='./test_predictions',
                       help='Directory to save statistical analysis results')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.5, 0.7, 0.9],
                       help='Thresholds to analyze')
    
    args = parser.parse_args()
    
    # Check if metrics directory exists
    if not os.path.exists(args.metrics_dir):
        print(f"❌ Metrics directory not found: {args.metrics_dir}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Metrics directory: {args.metrics_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Thresholds to analyze: {args.thresholds}")
    
    # Check for metric files
    import glob
    metric_files = []
    for threshold in args.thresholds:
        files = glob.glob(os.path.join(args.metrics_dir, f"*-{threshold}.csv"))
        metric_files.extend(files)
    
    if not metric_files:
        print(f"❌ No metric files found in {args.metrics_dir}")
        print("Expected files: *_metrics-{threshold}.csv")
        return
    
    print(f"Found {len(metric_files)} metric files:")
    for file in metric_files:
        print(f"  - {os.path.basename(file)}")
    
    # Run statistical analysis for each threshold
    all_results = {}
    
    for threshold in args.thresholds:
        print(f"\n{'='*60}")
        print(f"ANALYZING THRESHOLD {threshold}")
        print(f"{'='*60}")
        
        try:
            results = perform_cross_model_statistical_analysis(args.metrics_dir, args.output_dir, [threshold])
            all_results[f'threshold_{threshold}'] = results
            
            if results:
                print(f"✅ Statistical analysis completed for threshold {threshold}")
            else:
                print(f"⚠️  No statistical analysis results for threshold {threshold}")
                
        except Exception as e:
            print(f"❌ Error in statistical analysis for threshold {threshold}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comprehensive summary report
    if all_results:
        generate_statistical_summary_report(all_results, args.output_dir)
        print(f"\n✅ Statistical analysis completed successfully!")
        print(f"Results saved in: {args.output_dir}")
    else:
        print(f"\n❌ No statistical analysis results generated")

if __name__ == "__main__":
    main()
