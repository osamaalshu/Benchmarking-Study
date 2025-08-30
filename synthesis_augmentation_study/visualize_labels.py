#!/usr/bin/env python3
"""
Simple script to visualize synthetic labels with proper contrast
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_image_and_label(image_path, label_path, save_path=None):
    """Visualize an image alongside its properly contrasted label"""
    
    # Load the image and label
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    
    # Create a properly contrasted version of the label
    # Scale values: 0->0, 1->127, 2->255 for better visibility
    label_vis = np.zeros_like(label)
    label_vis[label == 1] = 127  # Interior cells -> gray
    label_vis[label == 2] = 255  # Boundaries -> white
    
    # Create the visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Synthetic Image')
    axes[0].axis('off')
    
    # Original label (dark)
    axes[1].imshow(label, cmap='gray')
    axes[1].set_title('Original Label (Dark)')
    axes[1].axis('off')
    
    # Enhanced label
    axes[2].imshow(label_vis, cmap='gray')
    axes[2].set_title('Enhanced Label\n(0=Black, 1=Gray, 2=White)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    # Print label statistics
    unique_vals, counts = np.unique(label, return_counts=True)
    print(f"\nLabel statistics for {label_path.name}:")
    for val, count in zip(unique_vals, counts):
        percentage = (count / label.size) * 100
        label_name = {0: "Background", 1: "Cell Interior", 2: "Cell Boundary"}.get(val, f"Unknown({val})")
        print(f"  {label_name}: {count} pixels ({percentage:.1f}%)")

if __name__ == "__main__":
    # Paths for synthetic image 2
    base_dir = Path(__file__).parent / "synthetic_data_500"
    image_path = base_dir / "synthetic_images_500" / "synthetic_0002.png"
    label_path = base_dir / "synthetic_labels_grayscale" / "synthetic_0002_label.png"
    
    if image_path.exists() and label_path.exists():
        visualize_image_and_label(image_path, label_path, "synthetic_0002_visualization.png")
    else:
        print(f"Files not found:")
        print(f"  Image: {image_path} (exists: {image_path.exists()})")
        print(f"  Label: {label_path} (exists: {label_path.exists()})")
