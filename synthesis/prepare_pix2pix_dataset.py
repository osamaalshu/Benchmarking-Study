#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def ensure_rgb(image_array: np.ndarray) -> np.ndarray:
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.ndim == 3 and image_array.shape[-1] > 3:
        image_array = image_array[:, :, :3]
    return image_array.astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    # Supports binary or 3-class masks (0,1,2)
    mask = mask.astype(np.uint8)
    if mask.ndim == 3:
        # Already RGB, just ensure uint8
        if mask.shape[-1] == 3:
            return mask
        mask = mask[:, :, 0]
    unique_vals = np.unique(mask)
    if np.array_equal(unique_vals, np.array([0, 1, 2])) or np.any(unique_vals == 2):
        palette = {
            0: (0, 0, 0),       # background
            1: (0, 255, 0),     # interior
            2: (255, 0, 0),     # boundary
        }
    else:
        palette = {
            0: (0, 0, 0),       # background
            1: (255, 255, 255), # foreground
        }
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, color in palette.items():
        rgb[mask == k] = color
    return rgb


def tile_image(image: np.ndarray, tile_size: int, stride: int):
    h, w = image.shape[:2]
    for y in range(0, max(1, h - tile_size + 1), stride):
        for x in range(0, max(1, w - tile_size + 1), stride):
            tile = image[y:y + tile_size, x:x + tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                yield y, x, tile


def save_ab(a_rgb: np.ndarray, b_rgb: np.ndarray, out_path: Path):
    assert a_rgb.shape == b_rgb.shape
    h, w, _ = a_rgb.shape
    ab = np.zeros((h, w * 2, 3), dtype=np.uint8)
    ab[:, :w, :] = a_rgb
    ab[:, w:, :] = b_rgb
    Image.fromarray(ab).save(out_path)


def process_split(img_dir: Path, label_dir: Path, out_dir: Path, tile_size: int, stride: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted([p for p in img_dir.iterdir() if p.is_file()])
    for img_fp in tqdm(image_files, desc=f"{img_dir.name}"):
        name = img_fp.stem
        # labels follow *_label.png convention
        label_fp = label_dir / f"{name}_label.png"
        if not label_fp.exists():
            # Try without suffix
            candidates = list(label_dir.glob(f"{name}*.png"))
            if not candidates:
                continue
            label_fp = candidates[0]
        img = np.array(Image.open(img_fp))
        mask = np.array(Image.open(label_fp))
        img = ensure_rgb(img)
        mask_rgb = colorize_mask(mask)
        for y, x, tile_b in tile_image(img, tile_size, stride):
            tile_a = mask_rgb[y:y + tile_size, x:x + tile_size]
            if tile_a.shape[:2] != (tile_size, tile_size):
                continue
            out_name = f"{name}_y{y}_x{x}_AB.png"
            save_ab(tile_a, tile_b, out_dir / out_name)


def main():
    parser = argparse.ArgumentParser("Prepare aligned AB dataset for pix2pix (A=mask, B=image)")
    parser.add_argument("--train_path", required=True, help="Path with images/ and labels/")
    parser.add_argument("--val_path", required=False, help="Optional val path with images/ and labels/")
    parser.add_argument("--test_path", required=False, help="Optional test path with images/ and labels/")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[0]
    ds_root = root / "datasets" / args.dataset_name
    (ds_root / "train").mkdir(parents=True, exist_ok=True)
    if args.val_path:
        (ds_root / "val").mkdir(parents=True, exist_ok=True)
    if args.test_path:
        (ds_root / "test").mkdir(parents=True, exist_ok=True)

    process_split(Path(args.train_path) / "images", Path(args.train_path) / "labels", ds_root / "train", args.tile_size, args.stride)
    if args.val_path:
        process_split(Path(args.val_path) / "images", Path(args.val_path) / "labels", ds_root / "val", args.tile_size, args.stride)
    if args.test_path:
        process_split(Path(args.test_path) / "images", Path(args.test_path) / "labels", ds_root / "test", args.tile_size, args.stride)

    print("pix2pix dataset ready at:", ds_root)


if __name__ == "__main__":
    main()


