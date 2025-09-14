
import os
import sys
join = os.path.join
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import numpy as np
import torch
import monai
from monai.inferers import sliding_window_inference
from models.unetr2d import UNETR2D
from models.sac_model import SACModel, create_default_points
from models.nnunet import create_nnunet_model
from models.lstmunet import create_lstmunet_model
from models.maunet import create_maunet_model, create_maunet_ensemble_model
from models.maunet_error_aware import create_maunet_error_aware_model
import time
from skimage import io, segmentation, morphology, measure, exposure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import gaussian
import scipy.ndimage as ndi
import tifffile as tif
import warnings

# Suppress PyTorch deprecation warnings
warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing is deprecated")

def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    if len(non_zero_vals) == 0:
        return img.astype(np.uint8)
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)

# Per-modality presets for smart post-processing
PRESETS = {
    'bf': dict(t_fg=0.60, t_core=0.65, t_cent=0.40, min_dist=6),
    'gs': dict(t_fg=0.60, t_core=0.65, t_cent=0.35, min_dist=6),
    'fl': dict(t_fg=0.50, t_core=0.60, t_cent=0.30, min_dist=5),
}

def _merge_params(modality: str, overrides: dict | None):
    base = {**PRESETS.get('gs', {})}
    base.update(PRESETS.get(modality, {}))
    if overrides:
        base.update(overrides)
    base.setdefault('min_area', 16)
    base.setdefault('hmin', 0.02)
    base.setdefault('compactness', 0.001)
    base.setdefault('max_splits_per_comp', 4)
    return base

def detect_modality(img_path: str) -> str:
    f = os.path.basename(img_path).lower()
    if 'bf' in f or 'bright' in f: return 'bf'
    if 'fl' in f or 'fluor' in f:  return 'fl'
    return 'gs'

def instance_quality_ok(mask: np.ndarray, dt_pred01: np.ndarray, fg_prob01: np.ndarray,
                        max_mse=0.30, min_fg=0.70) -> bool:
    if mask.max() == 0: return False
    dist = ndi.distance_transform_edt(mask > 0)
    if dist.max() > 0: dist = dist / (dist.max() + 1e-6)
    m = mask > 0
    if not m.any(): return False
    mse  = ((dt_pred01 - dist)**2)[m].mean()
    conf = fg_prob01[m].mean()
    return (mse <= max_mse) and (conf >= min_fg)

def instance_nms(labels: np.ndarray, score_map: np.ndarray, iou_th=0.5) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int32)
    ids = [i for i in range(1, labels.max()+1)]
    if not ids: return out.astype(np.uint16)
    boxes, scores, id_list = [], [], []
    for lab_id in ids:
        m = (labels == lab_id)
        if m.sum() == 0: continue
        ys, xs = np.where(m)
        boxes.append([ys.min(), xs.min(), ys.max()+1, xs.max()+1])
        scores.append(float(score_map[m].mean()))
        id_list.append(lab_id)
    if not boxes: return out.astype(np.uint16)
    order = np.argsort(scores)[::-1]
    taken = np.zeros(len(order), bool)
    def iou(b1, b2):
        y0=max(b1[0],b2[0]); x0=max(b1[1],b2[1]); y1=min(b1[2],b2[2]); x1=min(b1[3],b2[3])
        inter=max(0,y1-y0)*max(0,x1-x0)
        a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/(a1+a2-inter+1e-6)
    keep=[]
    for i in order:
        if taken[i]: continue
        keep.append(i)
        for j in order:
            if taken[j] or j==i: continue
            if iou(boxes[i], boxes[j]) > iou_th: taken[j]=True
    nid=1
    for idx in keep:
        lab_id = id_list[idx]
        out[labels == lab_id] = nid
        nid += 1
    return out.astype(np.uint16)

def estimate_scene(seg_interior: np.ndarray,
                   dist_prob: np.ndarray | None,
                   cent_prob: np.ndarray | None,
                   presets: dict,
                   modality: str) -> dict:
    """
    Returns a dict with:
      - seed_count: number of confident peaks (estimate of instances)
      - r_med: median radius proxy from dist map at peaks (pixels)
      - sparse: bool flag for low-cell scenes
    """
    p = {**presets.get(modality, presets['gs'])}
    t_cent = p.get('t_cent', 0.4)
    min_dist = p.get('min_dist', 6)

    # Smooth centroid for stable peak detection
    if cent_prob is not None:
        cent_s = gaussian(cent_prob, sigma=1.0, preserve_range=True)
        peaks = peak_local_max(
            cent_s, min_distance=min_dist, threshold_abs=t_cent,
            labels=(seg_interior >= max(0.5, p.get('t_fg', 0.5))).astype(np.uint8),
            exclude_border=False
        )
    else:
        peaks = np.empty((0, 2), dtype=int)

    # Radius proxy from dist map at peaks (fallback to EDT if needed)
    if dist_prob is not None and len(peaks) > 0:
        r_vals = dist_prob[peaks[:, 0], peaks[:, 1]]
    else:
        edt = ndi.distance_transform_edt(seg_interior >= 0.5)
        r_vals = edt[edt > 0]
    r_med = float(np.median(r_vals)) if r_vals.size > 0 else 6.0

    seed_count = int(peaks.shape[0])
    sparse = seed_count < 50  # your failure regime

    return dict(seed_count=seed_count, r_med=r_med, sparse=sparse)

def adapt_params_for_scene(base_params: dict, scene: dict, modality: str, seg_interior: np.ndarray = None) -> dict:
    """
    Tighten thresholds on sparse scenes; relax slightly on very dense ones.
    """
    p = {**base_params}
    k_area = max(25.0, np.pi * (max(scene['r_med'], 2.0) ** 2))  # typical cell area proxy

    if scene['sparse']:
        # More moderate adjustments for sparse scenes - prevent over-prediction but allow detection
        p['t_fg']   = max(p.get('t_fg', 0.5), 0.65 if modality != 'fl' else 0.55)  # Less aggressive
        p['t_core'] = max(p.get('t_core', 0.7), p['t_fg'] + 0.03)  # Smaller gap
        p['t_cent'] = max(0.2, p.get('t_cent', 0.4) - 0.1)  # Lower threshold for sparse scenes
        p['min_area'] = max(32, int(0.5 * k_area))  # Less aggressive size filter
        p['max_splits_per_comp'] = min(3, p.get('max_splits_per_comp', 4))
        p['hmin'] = min(0.025, p.get('hmin', 0.02))  # Smaller smoothing
        p['compactness'] = min(0.0012, p.get('compactness', 0.001))

        # Only force empty for truly empty scenes
        if scene['seed_count'] == 0 and seg_interior is not None and float(seg_interior.mean()) < 0.05:
            p['_force_empty'] = True
    else:
        # Slight relaxation for dense scenes (helps recall)
        if scene['seed_count'] >= 500:
            p['t_fg']   = max(0.48, p.get('t_fg', 0.5) - 0.02)
            p['min_area'] = max(16, int(0.35 * k_area))
            p['max_splits_per_comp'] = min(4, p.get('max_splits_per_comp', 4))

    # Ensure defaults exist
    p.setdefault('min_area', 32)
    p.setdefault('hmin', 0.02)
    p.setdefault('compactness', 0.001)
    p.setdefault('max_splits_per_comp', 4)
    return p

def adaptive_threshold(seg_interior: np.ndarray, initial_threshold: float = 0.6) -> float:
    """Adaptively adjust threshold based on over-prediction detection"""
    # Quick prediction count at different thresholds
    pred_50 = measure.label(seg_interior >= 0.5).max()
    pred_60 = measure.label(seg_interior >= 0.6).max()
    pred_70 = measure.label(seg_interior >= 0.7).max()
    
    image_area_mp = (seg_interior.shape[0] * seg_interior.shape[1]) / 1000000
    
    # Expected reasonable cell density ranges
    expected_max_density = 200  # cells per megapixel (very generous)
    expected_min_density = 5   # cells per megapixel (very sparse)
    
    current_density = pred_50 / image_area_mp
    
    # If massively over-predicting, use much higher threshold
    if current_density > expected_max_density * 5:  # 1000+ cells/MP
        return 0.85
    elif current_density > expected_max_density * 2:  # 400+ cells/MP  
        return 0.80
    elif current_density > expected_max_density:  # 200+ cells/MP
        return 0.75
    elif current_density > expected_max_density * 0.5:  # 100+ cells/MP
        return 0.70
    else:
        return initial_threshold

def conservative_instance_postprocess(
    seg_prob_interior: np.ndarray,
    seg_prob_boundary: np.ndarray | float = 0.0,
    dist_prob: np.ndarray | None = None,
    centroid_prob: np.ndarray | None = None,
    modality: str = 'gs',
    **overrides
) -> np.ndarray:
    p = _merge_params(modality, overrides)
    t_fg, t_core, t_cent = p['t_fg'], p['t_core'], p['t_cent']
    min_dist, min_area = p['min_dist'], p['min_area']
    hmin, compactness, max_splits = p['hmin'], p['compactness'], p['max_splits_per_comp']

    # 0) Optional early exit for very sparse empty scenes
    if p.get('_force_empty', False):
        return np.zeros_like(seg_prob_interior, dtype=np.uint16)

    # 1) Foreground from INTERIOR ONLY (don't union boundary up front)
    fg = (seg_prob_interior >= t_fg)
    if not fg.any():
        return np.zeros_like(seg_prob_interior, dtype=np.uint16)

    # 2) Require each FG component to contain 'core' pixels
    core = (seg_prob_interior >= t_core)
    labels = measure.label(fg)
    keep = np.zeros(labels.max() + 1, dtype=bool)
    for i in range(1, labels.max() + 1):
        comp = (labels == i)
        if (core & comp).any():
            keep[i] = True
    fg = keep[labels]
    if not fg.any():
        return np.zeros_like(seg_prob_interior, dtype=np.uint16)

    # 3) If no aux heads, return CC like baseline
    if dist_prob is None or centroid_prob is None:
        cleaned = morphology.remove_small_objects(morphology.remove_small_holes(fg), min_area)
        return measure.label(cleaned).astype(np.uint16)

    # 4) Find candidate seeds (centroid first, EDT fallback if too few)
    cent_s = gaussian(centroid_prob, sigma=1.0, preserve_range=True)
    peaks = peak_local_max(cent_s, min_distance=min_dist, threshold_abs=t_cent,
                           labels=core.astype(np.uint8), exclude_border=False)

    seeds_all = [(r, c) for (r, c) in peaks if seg_prob_interior[r, c] >= t_core]
    if len(seeds_all) < 2:
        edt_global = ndi.distance_transform_edt(fg)
        if edt_global.max() > 0:
            edt01 = edt_global / (edt_global.max() + 1e-6)
            pks2 = peak_local_max(gaussian(edt01, 1.0, preserve_range=True),
                                  min_distance=min_dist, threshold_rel=0.25,
                                  labels=fg.astype(np.uint8), exclude_border=False)
            seeds_all = [(r, c) for (r, c) in pks2]

    # 5) Per-component watershed with adaptive spacing/splits
    cc_labels = measure.label(fg)
    final_labels = np.zeros_like(cc_labels, dtype=np.uint16)
    cur = 1

    # Normalize predicted DT once
    local_dt = dist_prob.copy()
    if local_dt.max() > 0:
        local_dt = local_dt / (local_dt.max() + 1e-6)

    for cc_id in range(1, cc_labels.max() + 1):
        cc_mask = (cc_labels == cc_id)
        if cc_mask.sum() < min_area:
            continue

        # Adaptive min_dist ~ 0.4 × estimated radius
        est_r = max(3, int(0.4 * np.sqrt(cc_mask.sum() / np.pi)))
        local_min_dist = max(3, min_dist, est_r)

        # Keep seeds inside this component and space them
        cand = np.array([(r, c) for (r, c) in seeds_all if cc_mask[r, c]])
        if cand.shape[0] <= 1:
            final_labels[cc_mask] = cur; cur += 1; continue

        # Cap over-splitting
        if cand.shape[0] > max_splits:
            vals = cent_s[cand[:, 0], cand[:, 1]]
            order = np.argsort(vals)[::-1][:max_splits]
            cand = cand[order]

        # Re-enforce spacing: greedily keep farthest next peak
        kept = []
        taken = np.zeros(cand.shape[0], dtype=bool)
        for i in np.argsort(cent_s[cand[:, 0], cand[:, 1]])[::-1]:
            if taken[i]: continue
            kept.append(cand[i])
            # mark too-close peaks
            d2 = (cand[:, 0] - cand[i, 0])**2 + (cand[:, 1] - cand[i, 1])**2
            taken |= (d2 < (local_min_dist ** 2))
        seeds_in_cc = kept

        seeds = np.zeros_like(cc_mask, dtype=np.int32)
        for k, (r, c) in enumerate(seeds_in_cc, start=1):
            seeds[r, c] = k

        # Energy: down-weight predicted DT (more conservative)
        edt = ndi.distance_transform_edt(cc_mask)
        if edt.max() > 0:
            edt = edt / (edt.max() + 1e-6)
        energy = -(0.3 * local_dt + 0.7 * edt)

        # h-min suppression within the component
        e = energy.copy()
        e_min, e_max = e[cc_mask].min(), e[cc_mask].max()
        rng = (e_max - e_min) + 1e-8
        e01 = np.zeros_like(e, dtype=np.float32)
        e01[cc_mask] = (e[cc_mask] - e_min) / rng
        marks = morphology.h_minima(e01, h=hmin)
        e01_sm = e01.copy()
        e01_sm[cc_mask] = ndi.gaussian_filter(e01[cc_mask], sigma=1.0)
        energy_final = - (np.where(marks, e01_sm, e01) * rng + e_min)

        ws = watershed(energy_final, markers=seeds, mask=cc_mask, compactness=compactness)
        for k in range(1, ws.max() + 1):
            m = (ws == k)
            if m.sum() >= min_area:
                final_labels[m] = cur
                cur += 1

    final_labels = morphology.remove_small_objects(final_labels, min_size=min_area)
    return final_labels.astype(np.uint16)

def smart_instance_postprocess(
    seg_prob_interior, seg_prob_boundary=0.0, dist_prob=None, centroid_prob=None,
    modality='gs', **kwargs
):
    """
    Simple rule: label FG; only run a *local* watershed inside components that
    have >=2 clear centroid peaks. Everywhere else keep the component intact.
    """
    # 1) Foreground & labeling (union interior+boundary avoids holes)
    fg = (seg_prob_interior + (seg_prob_boundary if isinstance(seg_prob_boundary, np.ndarray) else 0.0)) >= t_fg
    if not fg.any():
        return np.zeros_like(seg_prob_interior, dtype=np.uint16)
    base_labels = measure.label(morphology.remove_small_holes(fg, area_threshold=16))
    base_labels = morphology.remove_small_objects(base_labels, min_size=min_area)

    # If no aux heads, just return the connected components (matches your baseline)
    if dist_prob is None or centroid_prob is None:
        return base_labels.astype(np.uint16)

    # 2) For each component: split only if we *see* multiple peaks
    out = np.zeros_like(base_labels, dtype=np.int32)
    next_id = 1
    for comp_id in range(1, base_labels.max() + 1):
        comp_mask = (base_labels == comp_id)
        if comp_mask.sum() < min_area:
            continue

        # Peaks only inside confident core (safer)
        core = (seg_prob_interior >= t_core) & comp_mask
        if core.sum() == 0:
            out[comp_mask] = next_id
            next_id += 1
            continue

        peaks = peak_local_max(
            centroid_prob, min_distance=min_dist, threshold_abs=t_cent, labels=core
        )

        # If <2 peaks => keep as single instance (no split)
        if peaks.shape[0] < 2:
            out[comp_mask] = next_id
            next_id += 1
            continue

        # Cap splits to avoid over-fragmentation in very large blobs
        if peaks.shape[0] > max_splits_per_comp:
            peaks = peaks[:max_splits_per_comp]

        # Local energy: blend predicted DT (if valid) with EDT inside the component
        local_dt = dist_prob.copy()
        if local_dt.max() > 0:
            local_dt = local_dt / local_dt.max()
        edt = ndi.distance_transform_edt(comp_mask)
        if edt.max() > 0:
            edt = edt / edt.max()
        energy = -(0.6 * local_dt + 0.4 * edt)

        # h-min suppression (within component only)
        e = energy.copy()
        e_min, e_max = e[comp_mask].min(), e[comp_mask].max()
        e01 = np.zeros_like(e); rng = (e_max - e_min) + 1e-8
        e01[comp_mask] = (e[comp_mask] - e_min) / rng
        # mark shallow minima, smooth just those areas
        marks = morphology.h_minima(e01, h=hmin)
        e01_smooth = e01.copy()
        e01_smooth[comp_mask] = ndi.gaussian_filter(e01[comp_mask].astype(np.float32), sigma=1.0)
        e_final = np.where(marks, e01_smooth, e01)
        energy_final = -(e_final * rng + e_min)

        # Build markers
        markers = np.zeros_like(base_labels, dtype=np.int32)
        sid = 1
        for r, c in peaks:
            if comp_mask[r, c]:
                markers[r, c] = sid
                sid += 1

        # Local watershed
        ws = watershed(energy_final, markers=markers, mask=comp_mask, compactness=0.002)

        # Relabel into global map
        for k in range(1, ws.max() + 1):
            m = (ws == k)
            if m.sum() >= min_area:
                out[m] = next_id
                next_id += 1

    # Final cleanup (merge/dismiss any tiny leftovers)
    out = morphology.remove_small_objects(out, min_size=min_area)
    return out.astype(np.uint16)

def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='./inputs', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./outputs', type=str, help='output path')
    parser.add_argument('--model_path', default='./work_dir/swinunetr_3class', help='path where to save models and segmentation results')
    parser.add_argument('--show_overlay', required=False, default=False, action="store_true", help='save segmentation overlay')
    parser.add_argument('--skip_101', action='store_true', help='Skip cell_00101 (very large image)')

    # Model parameters
    parser.add_argument('--model_name', default='swinunetr', help='select mode: unet, unetr, swinunetr, sac, nnunet, lstmunet, maunet, maunet_error_aware')
    parser.add_argument('--num_class', default=3, type=int, help='segmentation classes')
    parser.add_argument('--input_size', default=256, type=int, help='segmentation classes')
    parser.add_argument('--backbone', default=None, type=str, choices=[None, 'resnet50', 'wide_resnet50'], help='Backbone for MAUNet (overrides inference from model_path if provided)')
    parser.add_argument('--ensemble', action='store_true', help='Use MAUNet ensemble for inference')
    parser.add_argument('--model_paths', type=str, default=None, help='Comma-separated list of model checkpoint directories for ensemble members')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    img_names = sorted(os.listdir(join(input_path)))
    
    # Skip cell_00101 if requested (very large image)
    if args.skip_101:
        img_names = [img for img in img_names if 'cell_00101' not in img]
        print("Skipping cell_00101.tif (very large image)")


    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.model_name.lower() == 'unet':
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=args.num_class,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)


    if args.model_name.lower() == 'unetr':
        model = UNETR2D(
            in_channels=3,
            out_channels=args.num_class,
            img_size=(args.input_size, args.input_size),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)


    if args.model_name.lower() == 'swinunetr':
        model = monai.networks.nets.SwinUNETR(
            img_size=(args.input_size, args.input_size), 
            in_channels=3, 
            out_channels=args.num_class,
            feature_size=24, # should be divisible by 12
            spatial_dims=2
            ).to(device)

    if args.model_name.lower() == 'sac':
        model = SACModel(device=device, num_classes=args.num_class, freeze_encoder_layers=6, use_lora=True, lora_rank=16)
        # Note: SACModel handles device internally and has its own decoder head

    if args.model_name.lower() == 'nnunet':
        model = create_nnunet_model(
            image_size=(args.input_size, args.input_size),
            in_channels=3,
            out_channels=args.num_class,
            gpu_memory_gb=8.0
        ).to(device)

    if args.model_name.lower() == 'lstmunet':
        model = create_lstmunet_model(
            image_size=(args.input_size, args.input_size),
            in_channels=3,
            out_channels=args.num_class,
            base_filters=64,
            depth=4,
            lstm_hidden_channels=64,
            lstm_layers=2,
            dropout_rate=0.1
        ).to(device)

    if args.model_name.lower() == 'maunet':
        # MAUNet can use either resnet50 or wide_resnet50 backbone or an ensemble
        if args.ensemble:
            if not args.model_paths:
                raise ValueError("--ensemble requires --model_paths with comma-separated checkpoint directories")
            paths = [p.strip() for p in args.model_paths.split(',') if p.strip()]
            if len(paths) < 2:
                raise ValueError("Provide at least two paths in --model_paths for ensemble")
            # Deduce backbones per path
            backbones = [('wide_resnet50' if 'wide' in p.lower() else 'resnet50') for p in paths]
            model = create_maunet_ensemble_model(
                num_classes=args.num_class,
                input_size=args.input_size,
                in_channels=3,
                backbones=backbones,
                average=True,
            ).to(device)
            # Load each member's checkpoint
            for idx, member in enumerate(model.models):
                ckpt_file = join(paths[idx], 'best_Dice_model.pth')
                checkpoint = torch.load(ckpt_file, map_location=torch.device(device))
                member.load_state_dict(checkpoint['model_state_dict'])
        else:
            if args.backbone is not None:
                backbone = args.backbone
            else:
                # Infer from model path if possible
                backbone = 'wide_resnet50' if 'wide' in args.model_path.lower() else 'resnet50'
            model = create_maunet_model(
                num_classes=args.num_class,
                input_size=args.input_size,
                in_channels=3,
                backbone=backbone
            ).to(device)

    if args.model_name.lower() == 'maunet_error_aware':
        # Error-aware MAUNet with specified backbone
        if args.backbone is not None:
            backbone = args.backbone
        else:
            # Infer from model path if possible
            backbone = 'wide_resnet50' if 'wide' in args.model_path.lower() else 'resnet50'
        model = create_maunet_error_aware_model(
            num_classes=args.num_class,
            input_size=args.input_size,
            in_channels=3,
            backbone=backbone
        ).to(device)

    if not args.ensemble:
        # Check if model_path already includes the filename
        if args.model_path.endswith('.pth'):
            checkpoint_path = args.model_path
        else:
            checkpoint_path = join(args.model_path, 'best_Dice_model.pth')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
    #%%
    roi_size = (args.input_size, args.input_size)
    sw_batch_size = 4
    model.eval()
    with torch.no_grad():
        for img_name in img_names:
            if img_name.endswith('.tif') or img_name.endswith('.tiff'):
                img_data = tif.imread(join(input_path, img_name))
            else:
                img_data = io.imread(join(input_path, img_name))
            
            # normalize image data
            if len(img_data.shape) == 2:
                img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
            elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
                img_data = img_data[:,:, :3]
            else:
                pass
            pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
            for i in range(3):
                img_channel_i = img_data[:,:,i]
                if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                    pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
            
            t0 = time.time()
            # Convert to tensor with zero-division guard
            mx = np.max(pre_img_data)
            test_npy01 = pre_img_data / (mx if mx > 0 else 1)
            test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
            
            # Handle SAC model differently (requires points)
            if args.model_name.lower() == 'sac':
                batch_size = test_tensor.shape[0]
                points = create_default_points(batch_size, (args.input_size, args.input_size))
                points = points.to(device)
                test_pred_out = model(test_tensor, points=points)
                # SAC model outputs 256x256, need to resize to original image size
                original_size = (test_tensor.shape[2], test_tensor.shape[3])
                test_pred_out = torch.nn.functional.interpolate(test_pred_out, size=original_size, mode='bilinear', align_corners=False)
            else:
                # Use sliding window inference for UNet, UNetR, SwinUNetR, and nnU-Net
                if args.model_name.lower() in ['maunet', 'maunet_error_aware']:
                    # MAUNet and Error-aware MAUNet return three outputs: segmentation, distance transform, and centroid/embeddings
                    # Create a wrapper to handle triple outputs
                    def maunet_predictor(x):
                        seg_out, _, _ = model(x)  # Get segmentation output
                        return seg_out
                    test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, maunet_predictor, padding_mode="reflect")
                    
                    # Get dist & centroid heads (still lightweight; one SWI each) for MAUNet variants
                    dist_pred = sliding_window_inference(test_tensor, roi_size, sw_batch_size, lambda x: model(x)[1], padding_mode="reflect")
                    center_pred = sliding_window_inference(test_tensor, roi_size, sw_batch_size, lambda x: model(x)[2], padding_mode="reflect")
                else:
                    test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)
                    dist_pred = center_pred = None
                
            test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1)  # (B,C,H,W)
            seg_interior = test_pred_out[0, 1].cpu().numpy()
            seg_boundary = test_pred_out[0, 2].cpu().numpy() if test_pred_out.shape[1] > 2 else 0.0
            dist_prob = torch.sigmoid(dist_pred)[0,0].cpu().numpy() if dist_pred is not None else None
            cent_prob = torch.sigmoid(center_pred)[0,0].cpu().numpy() if center_pred is not None else None

            modality = detect_modality(join(input_path, img_name))

            # HYBRID APPROACH: Use different strategies for different image types
            
            # Quick over-prediction detection
            initial_cc = measure.label(seg_interior >= 0.5).max()
            image_area_mp = (seg_interior.shape[0] * seg_interior.shape[1]) / 1000000
            pred_density = initial_cc / image_area_mp
            
            # ULTRA-CATASTROPHIC (>1000 cells/MP) - need 50-100x reduction
            if pred_density > 1000:
                # Ultra-high threshold - only very confident cells
                fg = (seg_interior >= 0.85) & (seg_boundary >= 0.80 if isinstance(seg_boundary, np.ndarray) else True)
                fg = morphology.remove_small_holes(fg, area_threshold=128)
                fg = morphology.remove_small_objects(fg, min_size=64)
                pred_mask = measure.label(fg).astype(np.uint16)
                approach = "ultra_conservative"
                
            # CATASTROPHIC (500-1000 cells/MP) - need 20-50x reduction
            elif pred_density > 500:
                # Very high threshold with strict filtering
                fg = (seg_interior >= 0.80) & (seg_boundary >= 0.75 if isinstance(seg_boundary, np.ndarray) else True)
                fg = morphology.remove_small_holes(fg, area_threshold=64)
                fg = morphology.remove_small_objects(fg, min_size=48)
                pred_mask = measure.label(fg).astype(np.uint16)
                approach = "catastrophic_conservative"
                
            # SEVERE (200-500 cells/MP) - need 5-20x reduction
            elif pred_density > 200:
                # High threshold
                adaptive_t = min(0.78, max(0.70, 0.65 + (pred_density - 200) / 1000))
                fg = (seg_interior >= adaptive_t)
                fg = morphology.remove_small_holes(fg, area_threshold=32)  
                fg = morphology.remove_small_objects(fg, min_size=32)
                pred_mask = measure.label(fg).astype(np.uint16)
                approach = f"severe_t={adaptive_t:.2f}"
                
            # MODERATE (100-200 cells/MP) - need 2-5x reduction
            elif pred_density > 100:
                adaptive_t = min(0.70, max(0.60, 0.55 + (pred_density - 100) / 500))
                fg = (seg_interior >= adaptive_t)
                fg = morphology.remove_small_holes(fg, area_threshold=16)  
                fg = morphology.remove_small_objects(fg, min_size=24)
                pred_mask = measure.label(fg).astype(np.uint16)
                approach = f"moderate_t={adaptive_t:.2f}"
                
            # MILD (50-100 cells/MP) - need small reduction
            elif pred_density > 50:
                adaptive_t = min(0.60, max(0.52, 0.5 + (pred_density - 50) / 250))
                fg = (seg_interior >= adaptive_t) | (seg_boundary >= adaptive_t if isinstance(seg_boundary, np.ndarray) else False)
                fg = morphology.remove_small_holes(fg, area_threshold=16)
                fg = morphology.remove_small_objects(fg, min_size=16)
                pred_mask = measure.label(fg).astype(np.uint16)
                approach = f"mild_t={adaptive_t:.2f}"
                
            # Normal images - use fixed smart post-processing
            else:
                pred_mask = conservative_instance_postprocess(
                    seg_interior, seg_boundary, dist_prob, cent_prob, modality
                )
                approach = "fixed_smart"
            
            final_count = int(pred_mask.max())
            print(f"[{img_name}] CC:{initial_cc} density:{pred_density:.1f} → {final_count} inst ({approach})")

            test_pred_mask = pred_mask
            tif.imwrite(join(output_path, img_name.split('.')[0]+'_label.tiff'), test_pred_mask, compression='zlib')
            t1 = time.time()
            print(f'Prediction finished: {img_name}; img size = {pre_img_data.shape}; costing: {t1-t0:.2f}s')
            
            if args.show_overlay:
                boundary = segmentation.find_boundaries(test_pred_mask, connectivity=1, mode='inner')
                boundary = morphology.binary_dilation(boundary, morphology.disk(2))
                img_data[boundary, :] = 255
                io.imsave(join(output_path, 'overlay_' + img_name), img_data, check_contrast=False)
            
        
if __name__ == "__main__":
    main()





