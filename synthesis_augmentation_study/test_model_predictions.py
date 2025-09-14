#!/usr/bin/env python3
"""
Test Model Predictions
Check which test images have meaningful predictions from the trained models
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from PIL import Image
import cv2
from skimage import measure
import sys

# Add the models directory to path
sys.path.append(str(Path(__file__).parent.parent / "models"))
from nnunet import nnUNet

class ModelTester:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup paths
        self.test_images_dir = Path("../data/test/images")
        self.test_labels_dir = Path("../data/test/labels")
        
        # Model paths
        self.baseline_model_path = self.results_dir / "training_results/nnunet/R_seed0"
        self.improved_model_path = self.results_dir / "training_results/nnunet/R+S@10_seed0"
        
        # Load models
        self.baseline_model = None
        self.improved_model = None
        self.load_models()
    
    def load_models(self):
        """Load the trained models"""
        try:
            print("Loading trained models...")
            
            # Load baseline model
            baseline_config_path = self.baseline_model_path / "config.json"
            if baseline_config_path.exists():
                with open(baseline_config_path, 'r') as f:
                    baseline_config = json.load(f)
                
                self.baseline_model = self.create_model_from_config(baseline_config)
                self.load_model_weights(self.baseline_model, self.baseline_model_path / "best_model.pth")
                print("✅ Baseline model loaded")
            
            # Load improved model
            improved_config_path = self.improved_model_path / "config.json"
            if improved_config_path.exists():
                with open(improved_config_path, 'r') as f:
                    improved_config = json.load(f)
                
                self.improved_model = self.create_model_from_config(improved_config)
                self.load_model_weights(self.improved_model, self.improved_model_path / "best_model.pth")
                print("✅ Improved model loaded")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
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
        if weights_path.exists():
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
        
        # Normalize to [0, 1] for model input
        image_normalized = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float().unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict_and_analyze(self, model, image_tensor):
        """Generate prediction and analyze activity"""
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).cpu().numpy()[0]
            
            # Analyze prediction activity
            unique_classes = np.unique(pred_class)
            class_counts = {cls: np.sum(pred_class == cls) for cls in unique_classes}
            
            # Check if model predicts anything other than background (class 0)
            non_bg_pixels = np.sum(pred_class > 0)
            total_pixels = pred_class.size
            activity_ratio = non_bg_pixels / total_pixels
            
            # Convert to instances
            cell_mask = (pred_class > 0).astype(np.uint8)
            instances = measure.label(cell_mask)
            num_instances = len(np.unique(instances)) - 1  # Exclude background
            
            return {
                'pred_class': pred_class,
                'unique_classes': unique_classes,
                'class_counts': class_counts,
                'activity_ratio': activity_ratio,
                'non_bg_pixels': non_bg_pixels,
                'num_instances': num_instances,
                'instances': instances
            }
    
    def test_multiple_images(self, num_images: int = 20):
        """Test predictions on multiple images to find active ones"""
        test_images = list(self.test_images_dir.glob("*.png")) + list(self.test_images_dir.glob("*.tif"))
        test_images = test_images[:num_images]  # Limit for testing
        
        results = []
        
        print(f"\nTesting {len(test_images)} images...")
        print("=" * 80)
        
        for i, img_path in enumerate(test_images):
            print(f"Testing {img_path.name}...")
            
            try:
                # Preprocess image
                image_tensor = self.preprocess_image(img_path)
                
                # Test baseline model
                baseline_result = self.predict_and_analyze(self.baseline_model, image_tensor)
                
                # Test improved model  
                improved_result = self.predict_and_analyze(self.improved_model, image_tensor)
                
                # Store results
                result = {
                    'image_name': img_path.name,
                    'image_path': img_path,
                    'baseline': baseline_result,
                    'improved': improved_result
                }
                results.append(result)
                
                # Print summary
                print(f"  Baseline: {baseline_result['activity_ratio']:.3f} active, {baseline_result['num_instances']} instances")
                print(f"  Improved: {improved_result['activity_ratio']:.3f} active, {improved_result['num_instances']} instances")
                print(f"  Classes: {baseline_result['unique_classes']}")
                
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
        
        # Find most active images
        print("\n" + "=" * 80)
        print("SUMMARY - Most Active Predictions:")
        print("=" * 80)
        
        # Sort by activity (baseline + improved activity)
        results.sort(key=lambda x: x['baseline']['activity_ratio'] + x['improved']['activity_ratio'], reverse=True)
        
        for i, result in enumerate(results[:10]):  # Top 10
            baseline = result['baseline']
            improved = result['improved']
            total_activity = baseline['activity_ratio'] + improved['activity_ratio']
            
            print(f"{i+1:2d}. {result['image_name']:20s} | "
                  f"Total Activity: {total_activity:.3f} | "
                  f"Baseline: {baseline['num_instances']:2d} instances ({baseline['activity_ratio']:.3f}) | "
                  f"Improved: {improved['num_instances']:2d} instances ({improved['activity_ratio']:.3f})")
        
        return results

def main():
    """Main function"""
    results_dir = Path("final_augmentation_results")
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    tester = ModelTester(results_dir)
    results = tester.test_multiple_images(30)  # Test 30 images
    
    print(f"\n🎯 Recommendation: Use the top images for visualization")
    print(f"   These show the most model activity and meaningful predictions")

if __name__ == "__main__":
    main()
