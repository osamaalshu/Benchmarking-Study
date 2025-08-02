#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to split training data into train and validation sets
"""

import os
import shutil
import random
from pathlib import Path
import argparse

def split_data(data_path, val_frac=0.1, seed=2022):
    """
    Split data into train and validation sets
    
    Parameters:
    -----------
    data_path : str
        Path to the data directory (should contain images/ and labels/ subfolders)
    val_frac : float
        Fraction of data to use for validation (default: 0.1)
    seed : int
        Random seed for reproducibility
    """
    random.seed(seed)
    
    data_path = Path(data_path)
    images_path = data_path / "images"
    labels_path = data_path / "labels"
    
    # Create validation directories
    val_images_path = data_path.parent / "val" / "images"
    val_labels_path = data_path.parent / "val" / "labels"
    
    val_images_path.mkdir(parents=True, exist_ok=True)
    val_labels_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted([f for f in images_path.iterdir() if f.is_file()])
    
    # Calculate number of validation samples
    n_val = int(len(image_files) * val_frac)
    
    # Randomly select validation files
    val_files = random.sample(image_files, n_val)
    
    print(f"Total images: {len(image_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Training images: {len(image_files) - len(val_files)}")
    
    # Move files to validation directory
    for img_file in val_files:
        # Get corresponding label file
        label_file = labels_path / f"{img_file.stem}_label.png"
        
        if label_file.exists():
            # Move image
            shutil.move(str(img_file), str(val_images_path / img_file.name))
            # Move label
            shutil.move(str(label_file), str(val_labels_path / label_file.name))
            print(f"Moved {img_file.name} and {label_file.name} to validation set")
        else:
            print(f"Warning: Label file {label_file} not found for {img_file}")
    
    print(f"\nData split complete!")
    print(f"Training data: {data_path}")
    print(f"Validation data: {data_path.parent / 'val'}")

def main():
    parser = argparse.ArgumentParser('Split training data into train and validation sets')
    parser.add_argument('--data_path', default='./data/train-preprocessed', type=str, 
                       help='Path to the data directory')
    parser.add_argument('--val_frac', default=0.1, type=float,
                       help='Fraction of data to use for validation')
    parser.add_argument('--seed', default=2022, type=int,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    split_data(args.data_path, args.val_frac, args.seed)

if __name__ == "__main__":
    main() 