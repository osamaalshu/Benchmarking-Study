#!/usr/bin/env python3
"""
Find images where R+S@10 model shows improvements over R baseline
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

class ImprovementFinder:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.test_images_dir = Path("../data/test/images")
        self.test_labels_dir = Path("../data/test/labels")
        
        # Model paths
        self.baseline_model_path = self.results_dir / "training_results/nnunet/R_seed0"
        self.improved_model_path = self.results_dir / "training_results/nnunet/R+S@10_seed0"
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load the trained models"""
        try:
            print("Loading trained models...")
            
            # Load baseline model
            baseline_config_path = self.baseline_model_path / "config.json"
            with open(baseline_config_path, 'r') as f:
                baseline_config = json.load(f)
            
            self.baseline_model = self.create_model_from_config(baseline_config)
            self.load_model_weights(self.baseline_model, self.baseline_model_path / "best_model.pth")
            
            # Load improved model
            improved_config_path = self.improved_model_path / "config.json"
            with open(improved_config_path, 'r') as f:
                improved_config = json.load(f)
            
            self.improved_model = self.create_model_from_config(improved_config)
            self.load_model_weights(self.improved_model, self.improved_model_path / "best_model.pth")
            print("✅ Models loaded")
                
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
    
    def load_ground_truth(self, label_path: Path, target_size: int = 256):
        """Load ground truth for comparison"""
        try:
            if label_path.suffix.lower() in ['.tif', '.tiff']:
                label = np.array(Image.open(label_path))
                if len(label.shape) > 2:
                    label = label[:,:,0]
            else:
                label = np.array(Image.open(label_path).convert('L'))
            
            # Resize
            label = cv2.resize(label, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            
            # Count ground truth instances
            gt_instances = len(np.unique(label)) - 1
            return label, gt_instances
            
        except Exception as e:
            print(f"Error loading label {label_path}: {e}")
            return np.zeros((target_size, target_size), dtype=np.uint8), 0
    
    def predict_and_evaluate(self, model, image_tensor):
        """Generate prediction and evaluate quality"""
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).cpu().numpy()[0]
            
            # Convert to instances
            cell_mask = (pred_class > 0).astype(np.uint8)
            instances = measure.label(cell_mask)
            num_instances = len(np.unique(instances)) - 1
            
            # Calculate activity metrics
            non_bg_pixels = np.sum(pred_class > 0)
            total_pixels = pred_class.size
            activity_ratio = non_bg_pixels / total_pixels
            
            # Calculate confidence (average probability of predicted class)
            confidence = np.mean(np.max(probs.cpu().numpy()[0], axis=0))
            
            return {
                'pred_class': pred_class,
                'instances': instances,
                'num_instances': num_instances,
                'activity_ratio': activity_ratio,
                'non_bg_pixels': non_bg_pixels,
                'confidence': confidence
            }
    
    def find_improvements(self):
        """Find images where improved model performs better"""
        test_images = list(self.test_images_dir.glob("*.png")) + list(self.test_images_dir.glob("*.tif"))
        
        improvements = []
        
        print(f"Analyzing {len(test_images)} images for improvements...")
        print("=" * 100)
        
        for img_path in test_images:
            try:
                # Load image and ground truth
                image_tensor = self.preprocess_image(img_path)
                
                # Find corresponding label
                label_name = f"{img_path.stem}_label.tiff"
                label_path = self.test_labels_dir / label_name
                
                if not label_path.exists():
                    continue
                
                ground_truth, gt_instances = self.load_ground_truth(label_path)
                
                # Get predictions
                baseline_result = self.predict_and_evaluate(self.baseline_model, image_tensor)
                improved_result = self.predict_and_evaluate(self.improved_model, image_tensor)
                
                # Calculate different types of improvements
                instance_improvement = improved_result['num_instances'] - baseline_result['num_instances']
                activity_improvement = improved_result['activity_ratio'] - baseline_result['activity_ratio']
                confidence_improvement = improved_result['confidence'] - baseline_result['confidence']
                
                # Calculate accuracy (how close to ground truth)
                baseline_accuracy = 1.0 - abs(baseline_result['num_instances'] - gt_instances) / max(gt_instances, 1)
                improved_accuracy = 1.0 - abs(improved_result['num_instances'] - gt_instances) / max(gt_instances, 1)
                accuracy_improvement = improved_accuracy - baseline_accuracy
                
                result = {
                    'image_name': img_path.name,
                    'image_path': img_path,
                    'label_path': label_path,
                    'gt_instances': gt_instances,
                    'baseline': baseline_result,
                    'improved': improved_result,
                    'instance_improvement': instance_improvement,
                    'activity_improvement': activity_improvement,
                    'confidence_improvement': confidence_improvement,
                    'accuracy_improvement': accuracy_improvement,
                    'baseline_accuracy': baseline_accuracy,
                    'improved_accuracy': improved_accuracy
                }
                
                improvements.append(result)
                
                print(f"{img_path.name:20s} | GT: {gt_instances:3d} | "
                      f"B: {baseline_result['num_instances']:4d} ({baseline_result['activity_ratio']:.3f}) | "
                      f"I: {improved_result['num_instances']:4d} ({improved_result['activity_ratio']:.3f}) | "
                      f"Δinst: {instance_improvement:+4d} | Δact: {activity_improvement:+.3f} | "
                      f"Δacc: {accuracy_improvement:+.3f}")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        # Find best improvements
        print("\n" + "=" * 100)
        print("BEST IMPROVEMENTS BY DIFFERENT METRICS:")
        print("=" * 100)
        
        # Sort by instance improvement
        print("\n🔢 INSTANCE COUNT IMPROVEMENTS (R+S@10 detects more cells):")
        instance_improvements = sorted(improvements, key=lambda x: x['instance_improvement'], reverse=True)
        for i, result in enumerate(instance_improvements[:10]):
            if result['instance_improvement'] > 0:
                print(f"{i+1:2d}. {result['image_name']:20s} | "
                      f"GT: {result['gt_instances']:3d} | "
                      f"Baseline: {result['baseline']['num_instances']:4d} → "
                      f"Improved: {result['improved']['num_instances']:4d} "
                      f"({result['instance_improvement']:+4d})")
        
        # Sort by activity improvement
        print("\n📈 ACTIVITY RATIO IMPROVEMENTS (R+S@10 has higher cell detection ratio):")
        activity_improvements = sorted(improvements, key=lambda x: x['activity_improvement'], reverse=True)
        for i, result in enumerate(activity_improvements[:10]):
            if result['activity_improvement'] > 0:
                print(f"{i+1:2d}. {result['image_name']:20s} | "
                      f"Baseline: {result['baseline']['activity_ratio']:.3f} → "
                      f"Improved: {result['improved']['activity_ratio']:.3f} "
                      f"({result['activity_improvement']:+.3f})")
        
        # Sort by accuracy improvement
        print("\n🎯 ACCURACY IMPROVEMENTS (R+S@10 closer to ground truth):")
        accuracy_improvements = sorted(improvements, key=lambda x: x['accuracy_improvement'], reverse=True)
        for i, result in enumerate(accuracy_improvements[:10]):
            if result['accuracy_improvement'] > 0:
                print(f"{i+1:2d}. {result['image_name']:20s} | "
                      f"GT: {result['gt_instances']:3d} | "
                      f"Baseline acc: {result['baseline_accuracy']:.3f} → "
                      f"Improved acc: {result['improved_accuracy']:.3f} "
                      f"({result['accuracy_improvement']:+.3f})")
        
        return improvements

def main():
    """Main function"""
    results_dir = Path("final_augmentation_results")
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    finder = ImprovementFinder(results_dir)
    improvements = finder.find_improvements()
    
    # Find the best overall improvement
    best_candidates = sorted(improvements, key=lambda x: x['instance_improvement'] + x['activity_improvement'] + x['accuracy_improvement'], reverse=True)
    
    print(f"\n🏆 TOP CANDIDATES FOR VISUALIZATION:")
    for i, result in enumerate(best_candidates[:5]):
        if any([result['instance_improvement'] > 0, result['activity_improvement'] > 0, result['accuracy_improvement'] > 0]):
            print(f"{i+1}. {result['image_name']} - Shows improvement in multiple metrics")

if __name__ == "__main__":
    main()
