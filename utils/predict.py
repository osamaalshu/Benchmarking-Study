
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
import time
from skimage import io, segmentation, morphology, measure, exposure
try:
    from skimage.feature import peak_local_max
except Exception:
    from skimage.morphology import local_maxima as peak_local_max
import tifffile as tif

def normalize_channel_float(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    if non_zero_vals.size == 0:
        return img.astype(np.float32)
    vmin, vmax = np.percentile(non_zero_vals, [lower, upper])
    if vmax <= vmin + 1e-6:
        return img.astype(np.float32)
    out = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return out.astype(np.float32)

def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation', add_help=False)
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='./inputs', type=str, help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./outputs', type=str, help='output path')
    parser.add_argument('--model_path', default='./work_dir/swinunetr_3class', help='path where to save models and segmentation results')
    parser.add_argument('--show_overlay', required=False, default=False, action="store_true", help='save segmentation overlay')

    # Model parameters
    parser.add_argument('--model_name', default='swinunetr', help='select mode: unet, unetr, swinunetr, sac, nnunet, lstmunet, maunet')
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

    if not args.ensemble:
        checkpoint = torch.load(join(args.model_path, 'best_Dice_model.pth'), map_location=torch.device(device))
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
            pre_img_data = np.zeros(img_data.shape, dtype=np.float32)
            for i in range(3):
                img_channel_i = img_data[:,:,i]
                if len(img_channel_i[np.nonzero(img_channel_i)])>0:
                    pre_img_data[:,:,i] = normalize_channel_float(img_channel_i, lower=1, upper=99)
            
            t0 = time.time()
            # Convert to tensor (already in [0,1] floats)
            test_tensor = torch.from_numpy(np.expand_dims(pre_img_data, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
            
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
                if args.model_name.lower() == 'maunet':
                    # Fetch both heads via SWI
                    seg_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, lambda x: model(x)[0])
                    dist_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, lambda x: model(x)[1])
                    test_pred_out = seg_out
                else:
                    test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)
                
            test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
            if args.model_name.lower() == 'maunet' and args.num_class >= 2:
                seg_probs = test_pred_out[0].cpu().numpy()
                dist_map = dist_out[0,0].cpu().numpy().astype(np.float32)
                prob_background = seg_probs[0]
                prob_interior = seg_probs[1] if seg_probs.shape[0] > 1 else seg_probs[0]
                # Instance post-processing via watershed(-distance)
                interior_mask = prob_interior > 0.7
                background_mask = prob_background < 0.3
                cell_mask = interior_mask & background_mask
                cell_mask = morphology.remove_small_objects(cell_mask, 16)
                if cell_mask.any():
                    # Smooth and detect peaks
                    import cv2
                    dm = dist_map / (np.max(dist_map) if np.max(dist_map) > 0 else 1.0)
                    blurred = cv2.GaussianBlur(dm, (5,5), 0)
                    try:
                        coords = peak_local_max(blurred, min_distance=2, threshold_rel=0.5, exclude_border=False)
                        peaks = np.zeros_like(blurred, dtype=bool)
                        if coords.size>0:
                            peaks[tuple(coords.T)] = True
                    except Exception:
                        peaks = peak_local_max(blurred, min_distance=2, threshold_rel=0.5)
                    markers = measure.label(peaks & cell_mask).astype(np.int32)
                    if markers.max() > 0:
                        ws = segmentation.watershed(-blurred, markers=markers, mask=cell_mask.astype(np.uint8))
                        test_pred_mask = ws.astype(np.uint16)
                    else:
                        test_pred_mask = measure.label(cell_mask).astype(np.uint16)
                else:
                    test_pred_mask = np.zeros(cell_mask.shape, dtype=np.uint16)
            else:
                test_pred_npy = test_pred_out[0,1].cpu().numpy() if test_pred_out.shape[1]>1 else test_pred_out[0,0].cpu().numpy()
                test_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(test_pred_npy>0.5),16))
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





