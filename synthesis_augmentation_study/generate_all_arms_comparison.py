#!/usr/bin/env python3
"""
Generate All Dataset Arms Comparison
Shows cell_00077 predictions across all dataset arms: R, R+S@10, R+S@25, R+S@50, S
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image
import cv2
from skimage import measure
import sys

# Add the models directory to path
sys.path.append(str(Path(__file__).parent.parent / "models"))
from nnunet import nnUNet

class AllArmsComparator:
    """Generate comparison across all dataset arms"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup paths
        self.test_images_dir = Path("../data/test/images")
        self.test_labels_dir = Path("../data/test/labels")
        
        # Model paths for all arms
        self.model_paths = {
            'R': self.results_dir / "training_results/nnunet/R_seed0",
            'R+S@10': self.results_dir / "training_results/nnunet/R+S@10_seed0",
            'R+S@25': self.results_dir / "training_results/nnunet/R+S@25_seed0", 
            'R+S@50': self.results_dir / "training_results/nnunet/R+S@50_seed0",
            'S': self.results_dir / "training_results/nnunet/S_seed0"
        }
        
        # Load all models
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models"""
        print("Loading all dataset arm models...")
        
        for arm_name, model_path in self.model_paths.items():
            try:
                # Load model config
                config_path = model_path / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Create model
                    model = self.create_model_from_config(config)
                    
                    # Load weights
                    weights_path = model_path / "best_model.pth"
                    if weights_path.exists():
                        self.load_model_weights(model, weights_path)
                        self.models[arm_name] = model
                        print(f"✅ {arm_name} model loaded")
                    else:
                        print(f"⚠️ Weights not found for {arm_name}")
                else:
                    print(f"⚠️ Config not found for {arm_name}")
                    
            except Exception as e:
                print(f"❌ Error loading {arm_name}: {e}")
    
    def create_model_from_config(self, config):
        """Create model from configuration"""
        in_channels = config.get('in_channels', 3)
        out_channels = config.get('out_channels', 3)
        base_filters = config.get('base_filters', 32)
        depth = config.get('depth', 5)
        
        model = nnUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            depth=depth
        )
        
        return model.to(self.device)
    
    def load_model_weights(self, model, weights_path):
        """Load trained weights into model"""
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
    
    def preprocess_image(self, image_path: Path, target_size: int = 256):
        """Preprocess image for model input"""
        if image_path.suffix.lower() in ['.tif', '.tiff']:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                image = np.array(Image.open(image_path).convert('RGB'))
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.array(Image.open(image_path).convert('RGB'))
        
        # Resize
        image = cv2.resize(image, (target_size, target_size))
        
        # Store original for display
        original_display = image.copy()
        
        # Normalize for model
        image_normalized = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float().unsqueeze(0)
        
        return image_tensor.to(self.device), original_display
    
    def load_ground_truth(self, label_path: Path, target_size: int = 256):
        """Load ground truth label"""
        try:
            if label_path.suffix.lower() in ['.tif', '.tiff']:
                label = np.array(Image.open(label_path))
                if len(label.shape) > 2:
                    label = label[:,:,0]
            else:
                label = np.array(Image.open(label_path).convert('L'))
            
            # Resize
            label = cv2.resize(label, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            return label
            
        except Exception as e:
            print(f"Error loading label {label_path}: {e}")
            return np.zeros((target_size, target_size), dtype=np.uint8)
    
    def predict_segmentation(self, model, image_tensor):
        """Generate segmentation prediction"""
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).cpu().numpy()[0]
            
            # Convert to instances
            cell_mask = (pred_class > 0).astype(np.uint8)
            instances = measure.label(cell_mask)
            
            return instances.astype(np.uint8)
    
    def colorize_instances(self, instances):
        """Convert instance segmentation to colored visualization"""
        colored = np.zeros((*instances.shape, 3), dtype=np.uint8)
        
        unique_instances = np.unique(instances)
        unique_instances = unique_instances[unique_instances > 0]
        
        # Use consistent colors
        np.random.seed(42)
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3] * 255
        
        for i, instance_id in enumerate(unique_instances):
            mask = instances == instance_id
            color_idx = i % len(colors)
            colored[mask] = colors[color_idx].astype(np.uint8)
        
        return colored
    
    def generate_all_arms_comparison(self, save_path: str = "all_arms_comparison_cell_00077.png"):
        """Generate comparison across all dataset arms"""
        
        # Find cell_00077
        test_images = list(self.test_images_dir.glob("*.tif")) + list(self.test_images_dir.glob("*.png"))
        
        target_image = None
        for img_path in test_images:
            if "cell_00077" in img_path.name:
                target_image = img_path
                break
        
        if not target_image:
            print("cell_00077 not found!")
            return
        
        # Find corresponding label
        label_name = f"{target_image.stem}_label.tiff"
        label_path = self.test_labels_dir / label_name
        
        print(f"Processing: {target_image.name}")
        
        # Load data
        image_tensor, original_image = self.preprocess_image(target_image)
        ground_truth = self.load_ground_truth(label_path)
        
        # Generate predictions for all arms
        predictions = {}
        for arm_name, model in self.models.items():
            try:
                pred = self.predict_segmentation(model, image_tensor)
                predictions[arm_name] = pred
                num_instances = len(np.unique(pred)) - 1
                print(f"{arm_name}: {num_instances} instances detected")
            except Exception as e:
                print(f"Error predicting {arm_name}: {e}")
                predictions[arm_name] = np.zeros_like(ground_truth)
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Remove spacing for clean look
        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.05, hspace=0.15)
        
        # Ground truth cell count
        num_gt_cells = len(np.unique(ground_truth)) - 1
        
        # Plot original image and ground truth
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold', pad=10)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(self.colorize_instances(ground_truth))
        axes[0, 1].set_title(f'Ground Truth\n({num_gt_cells} cells)', fontsize=14, fontweight='bold', pad=10)
        axes[0, 1].axis('off')
        
        # Plot predictions for each arm
        arm_positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
        arm_order = ['R', 'R+S@10', 'R+S@25', 'R+S@50']
        
        for i, arm_name in enumerate(arm_order):
            if arm_name in predictions:
                row, col = arm_positions[i]
                pred = predictions[arm_name]
                num_pred_cells = len(np.unique(pred)) - 1
                
                # Color predictions
                pred_colored = self.colorize_instances(pred)
                axes[row, col].imshow(pred_colored)
                
                # Calculate improvement vs baseline
                if arm_name != 'R' and 'R' in predictions:
                    baseline_cells = len(np.unique(predictions['R'])) - 1
                    improvement = num_pred_cells - baseline_cells
                    improvement_text = f" ({improvement:+d})" if improvement != 0 else ""
                else:
                    improvement_text = " (baseline)"
                
                axes[row, col].set_title(f'{arm_name}\n({num_pred_cells} cells{improvement_text})', 
                                       fontsize=14, fontweight='bold', pad=10)
                axes[row, col].axis('off')
        
        # Handle S (Synthetic-only) separately if available
        if 'S' in predictions:
            # Use the empty slot or create additional space
            axes[0, 2].clear()  # Clear the R position and use it for S
            pred = predictions['S']
            num_pred_cells = len(np.unique(pred)) - 1
            
            pred_colored = self.colorize_instances(pred)
            axes[0, 2].imshow(pred_colored)
            axes[0, 2].set_title(f'S (Synthetic Only)\n({num_pred_cells} cells)', 
                               fontsize=14, fontweight='bold', pad=10)
            axes[0, 2].axis('off')
        
        # Add overall title
        fig.suptitle('Dataset Arms Comparison: cell_00077.tif', fontsize=16, fontweight='bold')
        
        # Add summary text
        summary_text = f"Ground Truth: {num_gt_cells} cells | "
        for arm_name in ['R', 'R+S@10', 'R+S@25', 'R+S@50', 'S']:
            if arm_name in predictions:
                num_cells = len(np.unique(predictions[arm_name])) - 1
                summary_text += f"{arm_name}: {num_cells} | "
        
        fig.text(0.5, 0.02, summary_text.rstrip(" | "), ha='center', fontsize=12, fontweight='bold')
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ All arms comparison saved to: {save_path}")
        
        plt.show()
        return save_path

def main():
    """Main function"""
    results_dir = Path("final_augmentation_results")
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    comparator = AllArmsComparator(results_dir)
    
    if not comparator.models:
        print("No models loaded successfully!")
        return
    
    output_path = comparator.generate_all_arms_comparison()
    
    print(f"\n🎉 All arms comparison complete!")
    print(f"📁 Figure saved to: {output_path}")
    print(f"\n📊 This shows cell_00077 across all dataset arms:")
    print(f"   • Original image and ground truth")
    print(f"   • R (real-only baseline)")
    print(f"   • R+S@10 (best performing)")
    print(f"   • R+S@25 (moderate improvement)")
    print(f"   • R+S@50 (diminishing returns)")
    print(f"   • S (synthetic-only)")

if __name__ == "__main__":
    main()
