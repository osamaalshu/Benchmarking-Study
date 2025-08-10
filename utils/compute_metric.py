"""
Created on Thu Mar 31 18:10:52 2022
adapted form https://github.com/stardist/stardist/blob/master/stardist/matching.py
Thanks the authors of Stardist for sharing the great code

"""

import argparse
import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import pandas as pd
from skimage import segmentation, io
import tifffile as tif
import os
join = os.path.join
from tqdm import tqdm
import traceback

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    x = x.ravel()
    y = y.ravel()
    
    # preallocate a 'contact map' matrix
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    
    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image 
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def dice(gt, seg):
    if np.count_nonzero(gt)==0 and np.count_nonzero(seg)==0:
        dice_score = 1.0
    elif np.count_nonzero(gt)==0 and np.count_nonzero(seg)>0:
        dice_score = 0.0
    else:
        union = np.count_nonzero(np.logical_and(gt, seg))
        intersection = np.count_nonzero(gt) + np.count_nonzero(seg)
        dice_score = 2*union/intersection
    return dice_score

def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold
    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp

def eval_tp_fp_fn(masks_true, masks_pred, threshold=0.5):
    num_inst_gt = np.max(masks_true)
    num_inst_seg = np.max(masks_pred)
    if num_inst_seg>0:
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
            # for k,th in enumerate(threshold):
        tp = _true_positive(iou, threshold)
        fp = num_inst_seg - tp
        fn = num_inst_gt - tp
    else:
        # print('No segmentation results!')
        tp = 0
        fp = 0
        fn = 0
        
    return tp, fp, fn

def remove_boundary_cells(mask):
    "We do not consider boundary cells during evaluation"
    W, H = mask.shape
    bd = np.ones((W, H))
    bd[2:W-2, 2:H-2] = 0
    bd_cells = np.unique(mask*bd)
    for i in bd_cells[1:]:
        mask[mask==i] = 0
    new_label,_,_ = segmentation.relabel_sequential(mask)
    return new_label

def perform_cross_model_statistical_analysis(metrics_dir, output_dir, thresholds=[0.5, 0.7, 0.9]):
    """
    Perform comprehensive statistical analysis across all models and thresholds
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    from scipy.stats import f_oneway, kruskal
    import os
    
    analysis_results = {}
    
    for threshold in thresholds:
        threshold_key = f"threshold_{threshold}"
        analysis_results[threshold_key] = {}
        
        # Load all model metrics for this threshold
        all_metrics = {}
        models = ['unet', 'nnunet', 'sac', 'lstmunet', 'maunet', 'maunet_ensemble']
        
        for model in models:
            metrics_file = os.path.join(metrics_dir, f"{model}_metrics-{threshold}.csv")
            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    all_metrics[model] = df
                except Exception as e:
                    print(f"Warning: Could not load {metrics_file}: {e}")
                    continue
        
        if not all_metrics:
            print(f"No metrics found for threshold {threshold}")
            continue
            
        # Perform analysis for each metric
        metrics_to_analyze = ['F1', 'precision', 'recall', 'dice']
        
        for metric in metrics_to_analyze:
            if metric not in analysis_results[threshold_key]:
                analysis_results[threshold_key][metric] = {}
            
            # Collect data for this metric
            metric_data = {}
            for model, df in all_metrics.items():
                if metric in df.columns:
                    metric_data[model] = df[metric].dropna().values
            
            if len(metric_data) < 2:
                continue
                
            # Descriptive statistics
            desc_stats = {}
            for model, data in metric_data.items():
                if len(data) > 0:
                    desc_stats[model] = {
                        'mean': np.mean(data),
                        'std': np.std(data),
                        'median': np.median(data),
                        'min': np.min(data),
                        'max': np.max(data),
                        'ci_lower': np.percentile(data, 2.5),
                        'ci_upper': np.percentile(data, 97.5)
                    }
            
            analysis_results[threshold_key][metric]['descriptive_stats'] = desc_stats
            
            # ANOVA test
            try:
                groups = [data for data in metric_data.values() if len(data) > 0]
                if len(groups) >= 2:
                    f_stat, p_value = f_oneway(*groups)
                    analysis_results[threshold_key][metric]['anova'] = {
                        'statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                else:
                    analysis_results[threshold_key][metric]['anova'] = {'error': 'Insufficient groups'}
            except Exception as e:
                analysis_results[threshold_key][metric]['anova'] = {'error': str(e)}
            
            # Kruskal-Wallis test
            try:
                groups = [data for data in metric_data.values() if len(data) > 0]
                if len(groups) >= 2:
                    h_stat, p_value = kruskal(*groups)
                    analysis_results[threshold_key][metric]['kruskal_wallis'] = {
                        'statistic': h_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                else:
                    analysis_results[threshold_key][metric]['kruskal_wallis'] = {'error': 'Insufficient groups'}
            except Exception as e:
                analysis_results[threshold_key][metric]['kruskal_wallis'] = {'error': str(e)}
            
            # Pairwise comparisons
            pairwise_results = {}
            model_names = list(metric_data.keys())
            
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    comparison_name = f"{model1}_vs_{model2}"
                    
                    data1 = metric_data[model1]
                    data2 = metric_data[model2]
                    
                    if len(data1) > 0 and len(data2) > 0:
                        pairwise_results[comparison_name] = {}
                        
                        # Independent t-test
                        try:
                            t_stat, p_value = stats.ttest_ind(data1, data2)
                            pairwise_results[comparison_name]['independent_ttest'] = {
                                'statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                        except Exception as e:
                            pairwise_results[comparison_name]['independent_ttest'] = {'error': str(e)}
                        
                        # Mann-Whitney U test
                        try:
                            u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            pairwise_results[comparison_name]['mannwhitney_u'] = {
                                'statistic': u_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
                        except Exception as e:
                            pairwise_results[comparison_name]['mannwhitney_u'] = {'error': str(e)}
                        
                        # Cohen's d effect size
                        try:
                            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                                (len(data1) + len(data2) - 2))
                            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                            
                            # Interpret effect size
                            if abs(cohens_d) < 0.2:
                                interpretation = "Negligible"
                            elif abs(cohens_d) < 0.5:
                                interpretation = "Small"
                            elif abs(cohens_d) < 0.8:
                                interpretation = "Medium"
                            else:
                                interpretation = "Large"
                            
                            pairwise_results[comparison_name]['cohens_d'] = {
                                'value': cohens_d,
                                'interpretation': interpretation
                            }
                        except Exception as e:
                            pairwise_results[comparison_name]['cohens_d'] = {'error': str(e)}
            
            analysis_results[threshold_key][metric]['pairwise_comparisons'] = pairwise_results
    
    return analysis_results

def main():
    parser = argparse.ArgumentParser('Compute F1 score for cell segmentation results', add_help=False)
    # Dataset parameters
    parser.add_argument('-g', '--gt_path', default='labelsTr_GT', type=str, help='path to ground truth')
    parser.add_argument('-s', '--seg_path', type=str, default='', help='path to segmentation results; file names are the same as ground truth', required=False)
    parser.add_argument('--gt_suffix', default='.tif', type=str, help='suffix of ground truth names')
    parser.add_argument('--seg_suffix', default='_label.tiff', type=str, help='suffix of segmentation names')
    parser.add_argument('-thre', '--thresholds', nargs='+', default=[0.5], type=float, help='threshold to count correct cells')
    parser.add_argument('-o', '--output_path', default='./', type=str, help='path where to save metrics')
    parser.add_argument('-n', '--save_name', default='demo', type=str, help='name of the csv file')
    # we opt to remove the boundary cells by default because these cells are usually not complete and the annotations have large variations
    parser.add_argument('--count_bd_cells', default=False, action='store_true', required=False, help='remove the boundary cells when computing metrics by default')
    args = parser.parse_args()

    gt_path = args.gt_path
    seg_path = args.seg_path
   
    names = sorted(os.listdir(seg_path))
    names = [i for i in names if i.endswith(args.seg_suffix)]
    
    for threshold in args.thresholds:
        print('compute metrics at threshold:', threshold)
        metrics = OrderedDict()
        metrics['names'] = []
        metrics['true_num'] = []
        metrics['pred_num'] = []
        metrics['correct_num(TP)'] = []
        metrics['missed_num(FN)'] = []
        metrics['wrong_num(FP)'] = []
        metrics['precision'] = []
        metrics['recall'] = []
        metrics['dice'] = []
        metrics['F1'] = []
        failed = []
        for name in tqdm(names):
            try:
                if name.endswith('.tif') or name.endswith('.tiff'):
                    gt_name = name.split(args.seg_suffix)[0] + args.gt_suffix
                    gt = tif.imread(join(gt_path, gt_name))
                    seg = tif.imread(join(seg_path, name))
                else:
                    gt_name = name.split(args.seg_suffix)[0] + args.gt_suffix
                    gt = io.imread(join(gt_path, gt_name))
                    seg = io.imread(join(seg_path, name))
                dice_score = dice(gt>0, seg>0)
                # Score the cases
                # do not consider cells on the boundaries during evaluation
                if np.prod(gt.shape)<25000000:
                    if not args.count_bd_cells:
                        gt = remove_boundary_cells(gt.astype(np.int32)) 
                        seg = remove_boundary_cells(seg.astype(np.int32))    
                    gt, _, _ = segmentation.relabel_sequential(gt)
                    seg, _, _ = segmentation.relabel_sequential(seg)
                    cell_true_num = np.max(gt)
                    cell_pred_num = np.max(seg)
                    tp, fp, fn = eval_tp_fp_fn(gt, seg, threshold=threshold)
                else: # for large images (>5000x5000), the F1 score is computed by a patch-based way
                    # this is because the grand-challenge platfrom has a limitation of RAM
                    # directly computing the metrics will have OOM issue.
                    H, W = gt.shape
                    roi_size = 2000
                
                    if H % roi_size != 0:
                        n_H = H // roi_size + 1
                        new_H = roi_size * n_H
                    else:
                        n_H = H // roi_size
                        new_H = H
                
                    if W % roi_size != 0:
                        n_W = W // roi_size + 1
                        new_W = roi_size * n_W    
                    else:
                        n_W = W // roi_size
                        new_W = W    
                
                    gt_pad = np.zeros((new_H, new_W), dtype=gt.dtype)
                    seg_pad = np.zeros((new_H, new_W), dtype=gt.dtype)
                    gt_pad[:H, :W] = gt
                    seg_pad[:H, :W] = seg
                      
                    tp = 0
                    fp = 0
                    fn = 0
                    cell_true_num = 0
                    cell_pred_num = 0
                    for i in range(n_H):
                        for j in range(n_W):
                            if not args.count_bd_cells:
                                gt_roi  = remove_boundary_cells(gt_pad[roi_size*i:roi_size*(i+1), roi_size*j:roi_size*(j+1)])
                                seg_roi = remove_boundary_cells(seg_pad[roi_size*i:roi_size*(i+1), roi_size*j:roi_size*(j+1)])
                            gt_roi, _, _ = segmentation.relabel_sequential(gt_roi)
                            seg_roi, _, _ = segmentation.relabel_sequential(seg_roi)
                            cell_true_num += np.max(gt_roi)
                            cell_pred_num += np.max(seg_roi)
                            tp_i, fp_i, fn_i = eval_tp_fp_fn(gt_roi, seg_roi, threshold=threshold)
                            tp += tp_i
                            fp += fp_i
                            fn += fn_i            
                if tp == 0:
                    precision = 0
                    recall = 0
                    f1 = 0
                else:
                    precision = tp / cell_pred_num
                    recall = tp / cell_true_num
                    f1 = 2 * (precision * recall)/ (precision + recall)
                    
                metrics['names'].append(name)
                metrics['true_num'].append(cell_true_num)
                metrics['pred_num'].append(cell_pred_num)
                metrics['correct_num(TP)'].append(tp)
                metrics['missed_num(FN)'].append(fn)    
                metrics['wrong_num(FP)'].append(fp)
                metrics['precision'].append(np.round(precision,4))
                metrics['recall'].append(np.round(recall, 4))
                metrics['dice'].append(np.round(dice_score, 4))
                metrics['F1'].append(np.round(f1, 4))
            except Exception:
                print('!'*20)
                print(name, 'evaluation error!')
                traceback.print_exc()
                failed.append(name)
            
        seg_metric_df = pd.DataFrame(metrics)
        if args.save_name.endswith('.csv'):
            args.save_name = args.save_name.split('.csv')[0]
        save_name = args.save_name + '-' + str(threshold) + '.csv'
        seg_metric_df.to_csv(join(args.output_path, save_name), index=False)
        print('threshold:', threshold, 'mean F1 Score:', np.mean(metrics['F1']), 'median F1 Score:', np.median(metrics['F1']))
        print('failed cases:', failed)

if __name__ == '__main__':
    main()
