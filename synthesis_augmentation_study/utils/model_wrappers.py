"""
Model wrappers for the augmentation study
Provides consistent interface to different model architectures
"""

import torch
import torch.nn as nn
import monai
import sys
import os
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.nnunet import create_nnunet_model
from models.lstmunet import create_lstmunet_model
from models.maunet import MAUNet
from models.sac_model import SACModel


def create_unet_model(input_channels: int = 3, 
                     num_classes: int = 3,
                     input_size: int = 256) -> nn.Module:
    """Create a U-Net model using MONAI"""
    return monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=input_channels,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def create_nnunet_wrapper(input_channels: int = 3,
                         num_classes: int = 3,
                         input_size: int = 256) -> nn.Module:
    """Create an nnU-Net model"""
    return create_nnunet_model(
        image_size=(input_size, input_size),
        in_channels=input_channels,
        out_channels=num_classes,
        gpu_memory_gb=8.0
    )


def create_lstmunet_wrapper(input_channels: int = 3,
                           num_classes: int = 3,
                           input_size: int = 256) -> nn.Module:
    """Create an LSTM U-Net model"""
    return create_lstmunet_model(
        image_size=(input_size, input_size),
        in_channels=input_channels,
        out_channels=num_classes,
        base_filters=64,
        depth=4,
        lstm_hidden_channels=64,
        lstm_layers=2,
        dropout_rate=0.1
    )


def create_maunet_wrapper(input_channels: int = 3,
                         num_classes: int = 3,
                         input_size: int = 256,
                         backbone: str = 'resnet50') -> nn.Module:
    """Create a MAUNet model"""
    return MAUNet(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=False
    )


def create_sac_wrapper(input_channels: int = 3,
                      num_classes: int = 3,
                      input_size: int = 256,
                      device: str = 'auto') -> nn.Module:
    """Create a SAC model"""
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    return SACModel(
        device=device,
        num_classes=num_classes,
        freeze_encoder_layers=6,
        use_lora=True,
        lora_rank=16
    )


def get_model_creator(model_name: str):
    """Get the appropriate model creator function"""
    creators = {
        'unet': create_unet_model,
        'nnunet': create_nnunet_wrapper,
        'lstmunet': create_lstmunet_wrapper,
        'maunet': create_maunet_wrapper,
        'sac': create_sac_wrapper
    }
    
    if model_name.lower() not in creators:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(creators.keys())}")
    
    return creators[model_name.lower()]


# Legacy compatibility classes
class UNetModel(nn.Module):
    """Legacy wrapper for U-Net model"""
    def __init__(self, input_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.model = create_unet_model(input_channels, num_classes)
    
    def forward(self, x):
        return self.model(x)


class nnUNetModel(nn.Module):
    """Legacy wrapper for nnU-Net model"""
    def __init__(self, input_channels: int = 3, num_classes: int = 3, input_size: tuple = (256, 256)):
        super().__init__()
        self.model = create_nnunet_wrapper(input_channels, num_classes, input_size[0])
    
    def forward(self, x):
        return self.model(x)


# Dataset class for compatibility
class CellDataset(torch.utils.data.Dataset):
    """Simple dataset class for cell segmentation"""
    def __init__(self, data_dir: str, input_size: int = 256, num_classes: int = 3, is_training: bool = True):
        import os
        from PIL import Image
        import torchvision.transforms as transforms
        import numpy as np
        
        self.data_dir = data_dir
        self.input_size = input_size
        self.num_classes = num_classes
        self.is_training = is_training
        
        # Find image and label files
        images_dir = os.path.join(data_dir, 'images')
        labels_dir = os.path.join(data_dir, 'labels')
        
        if not (os.path.exists(images_dir) and os.path.exists(labels_dir)):
            raise ValueError(f"Expected images and labels directories in {data_dir}")
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))])
        
        self.samples = []
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            
            # Find corresponding label (handle _label suffix)
            base_name = os.path.splitext(img_file)[0]
            label_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                # Try with _label suffix first
                potential_label = os.path.join(labels_dir, base_name + '_label' + ext)
                if os.path.exists(potential_label):
                    label_path = potential_label
                    break
                # Try without _label suffix as fallback
                potential_label = os.path.join(labels_dir, base_name + ext)
                if os.path.exists(potential_label):
                    label_path = potential_label
                    break
            
            if label_path:
                self.samples.append((img_path, label_path))
        
        # Setup transforms
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ])
        
        self.label_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Load label as integer array (preserve class IDs)
        label = Image.open(label_path).convert('L')
        label = label.resize((self.input_size, self.input_size), resample=Image.NEAREST)
        label_array = np.array(label)
        
        # Handle different label formats
        if self.num_classes == 2:
            # Binary case: ensure labels are {0, 1}
            label_array = (label_array > 127).astype(np.uint8)
        elif self.num_classes == 3:
            # 3-class case: map from common formats
            if label_array.max() > 2:
                # If labels are 0/128/255, map to 0/1/2
                if np.unique(label_array).tolist() == [0, 128, 255] or np.unique(label_array).tolist() == [0, 255]:
                    label_array = (label_array / 127.5).astype(np.uint8)
                    label_array = np.clip(label_array, 0, 2)
                else:
                    # If labels are 0/255 binary, treat as 0/2 for background/foreground
                    label_array = (label_array > 127).astype(np.uint8) * 2
        else:
            # Multi-class: assume labels are already in correct format
            label_array = np.clip(label_array, 0, self.num_classes - 1)
        
        label = torch.from_numpy(label_array).long()
        
        return image, label


if __name__ == "__main__":
    # Test model creation
    print("Testing model wrappers...")
    
    models_to_test = ['unet', 'nnunet']  # Test basic models
    
    for model_name in models_to_test:
        try:
            creator = get_model_creator(model_name)
            model = creator(input_channels=3, num_classes=3, input_size=256)
            
            # Test forward pass
            x = torch.randn(1, 3, 256, 256)
            output = model(x)
            print(f"✓ {model_name}: Input {x.shape} -> Output {output.shape}")
        except Exception as e:
            print(f"✗ {model_name}: {e}")
    
    print("Model wrapper testing completed.")
