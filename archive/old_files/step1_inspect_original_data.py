#!/usr/bin/env python3
"""
Step 1: Comprehensive inspection of original training data
Understanding resolution, format, quality, and characteristics for synthesis pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json

def inspect_original_data():
    """Comprehensive analysis of original training data"""
    
    print("🔍 Step 1: Inspecting Original Training Data")
    print("=" * 60)
    
    # Data directories
    train_images_dir = Path("data/train-preprocessed/images")
    train_labels_dir = Path("data/train-preprocessed/labels")
    
    if not train_images_dir.exists() or not train_labels_dir.exists():
        print("❌ Training data directories not found!")
        return False
    
    # Get file lists
    image_files = sorted(list(train_images_dir.glob("*.png")))
    label_files = sorted(list(train_labels_dir.glob("*.png")))
    
    print(f"📁 Found {len(image_files)} images and {len(label_files)} labels")
    
    # Sample analysis on first 10 images
    sample_images = image_files[:10]
    sample_labels = label_files[:10]
    
    print("\n📊 RESOLUTION ANALYSIS:")
    print("-" * 30)
    
    resolutions = []
    file_sizes = []
    
    for i, img_path in enumerate(sample_images):
        img = Image.open(img_path)
        size_bytes = img_path.stat().st_size
        
        resolutions.append(img.size)  # (width, height)
        file_sizes.append(size_bytes)
        
        if i < 3:  # Show details for first 3
            print(f"  {img_path.name}: {img.size[0]}×{img.size[1]} ({size_bytes/1024/1024:.1f}MB)")
    
    # Check if all images have same resolution
    unique_resolutions = set(resolutions)
    if len(unique_resolutions) == 1:
        standard_res = list(unique_resolutions)[0]
        print(f"✅ All images have consistent resolution: {standard_res[0]}×{standard_res[1]}")
    else:
        print(f"⚠️  Multiple resolutions found: {unique_resolutions}")
    
    print(f"📏 Average file size: {np.mean(file_sizes)/1024/1024:.1f}MB")
    
    print("\n🎨 IMAGE FORMAT ANALYSIS:")
    print("-" * 30)
    
    # Analyze first image in detail
    sample_img = Image.open(sample_images[0])
    sample_array = np.array(sample_img)
    
    print(f"  Mode: {sample_img.mode}")
    print(f"  Shape: {sample_array.shape}")
    print(f"  Data type: {sample_array.dtype}")
    print(f"  Value range: {sample_array.min()} - {sample_array.max()}")
    
    if len(sample_array.shape) == 3:
        print(f"  Channels: {sample_array.shape[2]}")
        # Check if grayscale disguised as RGB
        if sample_array.shape[2] == 3:
            r, g, b = sample_array[:,:,0], sample_array[:,:,1], sample_array[:,:,2]
            if np.array_equal(r, g) and np.array_equal(g, b):
                print("  ℹ️  RGB format but appears to be grayscale")
            else:
                print("  ℹ️  True color RGB image")
    
    print("\n🏷️  LABEL FORMAT ANALYSIS:")
    print("-" * 30)
    
    # Analyze corresponding label
    sample_label_path = train_labels_dir / f"{sample_images[0].stem}_label.png"
    if sample_label_path.exists():
        sample_label = Image.open(sample_label_path)
        label_array = np.array(sample_label)
        
        print(f"  Mode: {sample_label.mode}")
        print(f"  Shape: {label_array.shape}")
        print(f"  Data type: {label_array.dtype}")
        print(f"  Unique values: {sorted(np.unique(label_array))}")
        
        # Analyze class distribution
        unique_vals, counts = np.unique(label_array, return_counts=True)
        total_pixels = label_array.size
        
        print("  Class distribution:")
        for val, count in zip(unique_vals, counts):
            percentage = (count / total_pixels) * 100
            class_name = {0: "Background", 1: "Cell Interior", 2: "Cell Boundary"}.get(val, f"Class {val}")
            print(f"    {class_name} ({val}): {percentage:.1f}%")
    
    print("\n📈 QUALITY ASSESSMENT:")
    print("-" * 30)
    
    # Load a few images for quality analysis
    sample_arrays = []
    for img_path in sample_images[:5]:
        img_array = np.array(Image.open(img_path))
        sample_arrays.append(img_array)
    
    # Calculate statistics
    all_pixels = np.concatenate([arr.flatten() for arr in sample_arrays])
    
    print(f"  Mean intensity: {np.mean(all_pixels):.1f}")
    print(f"  Std deviation: {np.std(all_pixels):.1f}")
    print(f"  Dynamic range: {np.min(all_pixels)} - {np.max(all_pixels)}")
    
    # Check for potential issues
    if np.mean(all_pixels) < 50:
        print("  ⚠️  Images appear quite dark")
    elif np.mean(all_pixels) > 200:
        print("  ⚠️  Images appear quite bright")
    else:
        print("  ✅ Good brightness distribution")
    
    if np.std(all_pixels) < 20:
        print("  ⚠️  Low contrast images")
    else:
        print("  ✅ Good contrast")
    
    print("\n📋 SYNTHESIS REQUIREMENTS:")
    print("-" * 30)
    
    # Extract key requirements for synthesis
    requirements = {
        "target_resolution": list(unique_resolutions)[0] if len(unique_resolutions) == 1 else "MIXED",
        "image_mode": sample_img.mode,
        "image_channels": sample_array.shape[2] if len(sample_array.shape) == 3 else 1,
        "label_classes": len(unique_vals),
        "label_values": unique_vals.tolist(),
        "mean_intensity": float(np.mean(all_pixels)),
        "std_intensity": float(np.std(all_pixels)),
        "file_count": len(image_files)
    }
    
    print(f"  🎯 Target resolution: {requirements['target_resolution']}")
    print(f"  🖼️  Image format: {requirements['image_mode']}, {requirements['image_channels']} channels")
    print(f"  🏷️  Label classes: {requirements['label_classes']} ({requirements['label_values']})")
    print(f"  📊 Intensity stats: μ={requirements['mean_intensity']:.1f}, σ={requirements['std_intensity']:.1f}")
    
    # Save requirements for next steps
    with open("synthesis_requirements.json", "w") as f:
        json.dump(requirements, f, indent=2)
    
    print(f"\n✅ Analysis complete! Requirements saved to synthesis_requirements.json")
    print(f"📝 Ready for Step 2: Test Mask Generation Pipeline")
    
    return True

if __name__ == "__main__":
    success = inspect_original_data()
    exit(0 if success else 1)
