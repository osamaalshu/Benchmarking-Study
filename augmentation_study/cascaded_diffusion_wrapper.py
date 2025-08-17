"""
Cascaded Diffusion Model Wrapper for Cell Microscopy Synthesis
Integrates the cascaded diffusion model for generating synthetic cell images and masks
"""

import os
import sys
import torch
import numpy as np
from typing import Tuple, List, Optional, Union
from pathlib import Path
import cv2
from PIL import Image
import logging
from skimage.draw import disk, ellipse

# Add cascaded diffusion to path
CASCADED_DIFFUSION_PATH = os.path.join(os.path.dirname(__file__), '..', 'synthesis', 'cascaded_diffusion')
sys.path.insert(0, CASCADED_DIFFUSION_PATH)

try:
    from diffusers import (
        StableDiffusionPipeline, 
        StableDiffusionInpaintPipeline,
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        DDPMPipeline
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import UNet2DConditionModel, PNDMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Diffusion libraries not available: {e}")
    DIFFUSERS_AVAILABLE = False

class CascadedDiffusionWrapper:
    """
    Wrapper for the cascaded diffusion model to generate synthetic cell microscopy images
    """
    
    def __init__(self, 
                 mask_model_path: Optional[str] = None,
                 texture_model_path: Optional[str] = None,
                 device: str = 'auto',
                 conditioning_method: str = 'inpainting'):
        """
        Initialize the cascaded diffusion wrapper
        
        Args:
            mask_model_path: Path to trained mask generation model
            texture_model_path: Path to trained texture generation model  
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            conditioning_method: Method for texture conditioning ('inpainting' or 'controlnet')
        """
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("Diffusion libraries not available. Install: pip install diffusers transformers accelerate")
        
        self.device = self._get_device(device)
        self.mask_model_path = mask_model_path
        self.texture_model_path = texture_model_path
        self.conditioning_method = conditioning_method
        self.style_reference_dir = "./data/train-preprocessed/images"
        
        # Initialize models
        self.mask_generator = None
        self.texture_generator = None
        self.controlnet = None
        
        # Style reference images for image-based conditioning
        self.style_references = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load style reference images
        self._load_style_references()
        
    def _get_device(self, device: str) -> torch.device:
        """Automatically detect best available device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_style_references(self, num_references: int = 8):
        """Load a set of real images to use as style references for image-based conditioning"""
        try:
            from pathlib import Path
            import random
            
            ref_dir = Path(self.style_reference_dir)
            if not ref_dir.exists():
                self.logger.warning(f"Style reference directory {ref_dir} not found")
                return
            
            # Get all image files
            image_files = list(ref_dir.glob('*.png')) + list(ref_dir.glob('*.jpg')) + list(ref_dir.glob('*.tiff'))
            
            if len(image_files) < num_references:
                self.logger.warning(f"Only found {len(image_files)} reference images, using all")
                selected_files = image_files
            else:
                # Randomly sample reference images
                selected_files = random.sample(image_files, num_references)
            
            # Load and preprocess reference images
            for img_path in selected_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    # Resize to standard size for consistency
                    img = img.resize((256, 256), Image.LANCZOS)
                    self.style_references.append(img)
                except Exception as e:
                    self.logger.warning(f"Failed to load reference image {img_path}: {e}")
            
            self.logger.info(f"Loaded {len(self.style_references)} style reference images")
            
        except Exception as e:
            self.logger.error(f"Failed to load style references: {e}")
    
    def setup_mask_generator(self, model_path: Optional[str] = None):
        """Setup the mask generation model (DDPM)"""
        if model_path is None:
            model_path = self.mask_model_path
            
        if model_path and os.path.exists(model_path):
            try:
                self.mask_generator = DDPMPipeline.from_pretrained(model_path)
                self.mask_generator = self.mask_generator.to(self.device)
                self.logger.info(f"Loaded mask generator from {model_path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to load mask generator: {e}")
                self.mask_generator = None
                return False
        else:
            # No external checkpoint provided; use simple fallback generator
            self.logger.warning("No valid mask generator model path provided; using simple fallback generator")
            self.mask_generator = "fallback"
            return True
    
    def setup_texture_generator(self, model_path: Optional[str] = None):
        """Setup the texture generation model with proper conditioning"""
        if model_path is None:
            model_path = self.texture_model_path
            
        # Determine base model path
        base_model = model_path if (model_path and os.path.exists(model_path)) else "runwayml/stable-diffusion-v1-5"
        
        try:
            if self.conditioning_method == 'inpainting':
                # Use inpainting pipeline for mask-conditioned generation
                if model_path and os.path.exists(model_path):
                    # Try to load as inpainting model
                    try:
                        self.texture_generator = StableDiffusionInpaintPipeline.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
                        )
                        self.logger.info(f"Loaded custom inpainting model from {model_path}")
                    except:
                        # Fallback to base inpainting model
                        self.texture_generator = StableDiffusionInpaintPipeline.from_pretrained(
                            "runwayml/stable-diffusion-inpainting",
                            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                            safety_checker=None,  # Disable NSFW filter for scientific images
                            requires_safety_checker=False
                        )
                        self.logger.info("Using base SD inpainting model (safety checker disabled)")
                else:
                    self.texture_generator = StableDiffusionInpaintPipeline.from_pretrained(
                        "runwayml/stable-diffusion-inpainting",
                        torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                        safety_checker=None,  # Disable NSFW filter for scientific images
                        requires_safety_checker=False
                    )
                    self.logger.info("Using base SD inpainting model (safety checker disabled)")
                    
            elif self.conditioning_method == 'controlnet':
                # Use ControlNet with scribble/segmentation conditioning
                try:
                    self.controlnet = ControlNetModel.from_pretrained(
                        "lllyasviel/sd-controlnet-scribble",
                        torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
                    )
                    self.texture_generator = StableDiffusionControlNetPipeline.from_pretrained(
                        base_model,
                        controlnet=self.controlnet,
                        torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
                    )
                    self.logger.info("Using ControlNet-based texture generation")
                except Exception as e:
                    self.logger.warning(f"ControlNet setup failed: {e}, falling back to inpainting")
                    self.conditioning_method = 'inpainting'
                    return self.setup_texture_generator(model_path)
            else:
                raise ValueError(f"Unknown conditioning method: {self.conditioning_method}")
                
            self.texture_generator = self.texture_generator.to(self.device)
            self.logger.info(f"Texture generator setup complete using {self.conditioning_method}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup texture generator: {e}")
            self.texture_generator = None
            return False
    
    def _generate_fallback_mask(self, image_size: Tuple[int, int]) -> np.ndarray:
        """Generate a simple synthetic mask using basic shapes"""
        import random
        from skimage.draw import disk, ellipse
        
        mask = np.zeros(image_size, dtype=np.uint8)
        
        # Generate 1-3 cell-like shapes
        num_shapes = random.randint(1, 3)
        
        for _ in range(num_shapes):
            # Random ellipse parameters
            center_y = random.randint(image_size[0] // 4, 3 * image_size[0] // 4)
            center_x = random.randint(image_size[1] // 4, 3 * image_size[1] // 4)
            radius_y = random.randint(20, min(60, image_size[0] // 4))
            radius_x = random.randint(20, min(60, image_size[1] // 4))
            
            # Create ellipse
            rr, cc = ellipse(center_y, center_x, radius_y, radius_x, shape=image_size)
            mask[rr, cc] = 1
        
        return mask

    def generate_synthetic_masks(self, 
                                num_masks: int,
                                image_size: Tuple[int, int] = (256, 256),
                                num_inference_steps: int = 50) -> List[np.ndarray]:
        """
        Generate synthetic cell masks using DDPM or fallback
        
        Args:
            num_masks: Number of masks to generate
            image_size: Output image size (H, W)
            num_inference_steps: Number of denoising steps
            
        Returns:
            List of generated masks as numpy arrays
        """
        if self.mask_generator is None:
            self.logger.error("Mask generator not initialized")
            return []
        
        if self.mask_generator == "fallback":
            self.logger.info("Using fallback mask generator")
        
        masks = []
        for i in range(num_masks):
            try:
                if self.mask_generator == "fallback":
                    # Use fallback mask generation
                    mask_array = self._generate_fallback_mask(image_size)
                else:
                    # Use DDPM
                    image = self.mask_generator(
                        num_inference_steps=num_inference_steps,
                        output_type="pil"
                    ).images[0]
                    
                    # Resize to target size
                    image = image.resize(image_size[::-1], Image.LANCZOS)  # PIL uses (W, H)
                    
                    # Convert to binary mask - use threshold to create cell-like shapes
                    mask_array = np.array(image.convert('L'))
                    # Use adaptive thresholding to create more realistic cell shapes
                    threshold = np.percentile(mask_array, 70)  # Use 70th percentile as threshold
                    mask_array = (mask_array > threshold).astype(np.uint8)
                
                masks.append(mask_array)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Generated {i + 1}/{num_masks} masks")
                    
            except Exception as e:
                self.logger.error(f"Failed to generate mask {i}: {e}")
                # Try fallback for this mask
                try:
                    mask_array = self._generate_fallback_mask(image_size)
                    masks.append(mask_array)
                except:
                    continue
        
        return masks
    
    def generate_synthetic_textures(self, 
                                   masks: List[np.ndarray],
                                   prompt: str = "",  # Minimal prompt for image-based conditioning
                                   num_inference_steps: int = 50,
                                   guidance_scale: float = 3.0) -> List[np.ndarray]:  # Lower CFG for image conditioning
        """
        Generate synthetic textures properly conditioned on masks
        
        Args:
            masks: List of mask arrays to condition on
            prompt: Text prompt for generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            List of generated texture images as numpy arrays
        """
        if self.texture_generator is None:
            self.logger.error("Texture generator not initialized")
            return []
        
        textures = []
        for i, mask in enumerate(masks):
            try:
                if self.conditioning_method == 'inpainting':
                    # Use inpainting for proper mask conditioning with image-based style control
                    mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')
                    
                    # Use image-based style conditioning instead of text prompts
                    import random
                    
                    # Select a random style reference image
                    if self.style_references:
                        style_ref = random.choice(self.style_references)
                        # Use the style reference as the initial image for better style matching
                        init_image = style_ref.copy()
                        # Ensure it matches the mask size
                        if init_image.size != (mask.shape[1], mask.shape[0]):
                            init_image = init_image.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
                    else:
                        # Fallback to neutral background if no references
                        init_image = Image.new('RGB', (mask.shape[1], mask.shape[0]), color=(85, 85, 85))
                    
                    # Use minimal or empty prompt - let the image reference dominate
                    style_prompt = prompt if prompt else "microscopy image"
                    
                    # Generate texture using inpainting with image-based style control
                    try:
                        image = self.texture_generator(
                            prompt=style_prompt,
                            image=init_image,
                            mask_image=mask_pil,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,  # Low CFG for image-dominated generation
                            strength=0.7  # Moderate strength to preserve style reference
                        ).images[0]
                    except Exception as e:
                        # Fallback: disable safety checker if available
                        if hasattr(self.texture_generator, 'safety_checker'):
                            original_safety = self.texture_generator.safety_checker
                            self.texture_generator.safety_checker = None
                            try:
                                image = self.texture_generator(
                                    prompt=style_prompt,
                                    image=init_image,
                                    mask_image=mask_pil,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    strength=0.7
                                ).images[0]
                            finally:
                                self.texture_generator.safety_checker = original_safety
                        else:
                            raise e
                    
                elif self.conditioning_method == 'controlnet':
                    # Use ControlNet for mask conditioning with image-based style
                    # Convert binary mask to edge/scribble format
                    mask_rgb = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_rgb)
                    
                    # Use minimal prompt for image-based style control
                    style_prompt = prompt if prompt else "microscopy image"
                    
                    # Generate with ControlNet conditioning
                    image = self.texture_generator(
                        prompt=style_prompt,
                        image=mask_pil,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,  # Low CFG for image-dominated generation
                        controlnet_conditioning_scale=1.0
                    ).images[0]
                
                else:
                    raise ValueError(f"Unknown conditioning method: {self.conditioning_method}")
                
                # Convert to numpy array and resize to match mask size
                texture_array = np.array(image)
                
                # Resize texture to match mask size if needed
                if texture_array.shape[:2] != mask.shape:
                    image_resized = image.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
                    texture_array = np.array(image_resized)
                
                # Ensure texture respects mask boundaries (additional safety)
                if len(texture_array.shape) == 3:
                    # Apply mask to each channel
                    mask_3d = np.stack([mask] * 3, axis=-1)
                    texture_array = texture_array * mask_3d
                else:
                    texture_array = texture_array * mask
                
                textures.append(texture_array)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Generated {i + 1}/{len(masks)} conditioned textures")
                    
            except Exception as e:
                self.logger.error(f"Failed to generate texture {i}: {e}")
                continue
        
        return textures
    
    def generate_paired_synthetic_data(self, 
                                      num_pairs: int,
                                      image_size: Tuple[int, int] = (256, 256),
                                      prompt: str = "",  # Minimal prompt for image-based style control
                                      mask_steps: int = 50,
                                      texture_steps: int = 50) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate paired (image, mask) synthetic data
        
        Args:
            num_pairs: Number of pairs to generate
            image_size: Output image size
            prompt: Text prompt for texture generation
            mask_steps: Denoising steps for mask generation
            texture_steps: Denoising steps for texture generation
            
        Returns:
            List of (image, mask) tuples
        """
        self.logger.info(f"Generating {num_pairs} paired synthetic samples...")
        
        # Generate masks first
        masks = self.generate_synthetic_masks(
            num_masks=num_pairs,
            image_size=image_size,
            num_inference_steps=mask_steps
        )
        
        if not masks:
            self.logger.error("No masks generated")
            return []
        
        # Generate textures conditioned on masks
        textures = self.generate_synthetic_textures(
            masks=masks,
            prompt=prompt,
            num_inference_steps=texture_steps
        )
        
        if len(textures) != len(masks):
            self.logger.warning(f"Mismatch in generated samples: {len(textures)} textures vs {len(masks)} masks")
        
        # Pair them up
        pairs = []
        for i in range(min(len(masks), len(textures))):
            pairs.append((textures[i], masks[i]))
        
        self.logger.info(f"Successfully generated {len(pairs)} paired samples")
        return pairs
    
    def generate_texture_for_real_mask(self, 
                                      mask: np.ndarray,
                                      prompt: str = "",  # Minimal prompt for image-based style control
                                      num_inference_steps: int = 50,
                                      guidance_scale: float = 3.0) -> Optional[np.ndarray]:  # Lower CFG
        """
        Generate synthetic texture for a real mask (for Rmask+SynthTex control)
        
        Args:
            mask: Real mask array
            prompt: Text prompt for generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Generated texture image as numpy array
        """
        if self.texture_generator is None:
            self.logger.error("Texture generator not initialized")
            return None
        
        try:
            textures = self.generate_synthetic_textures(
                masks=[mask],
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            return textures[0] if textures else None
        except Exception as e:
            self.logger.error(f"Failed to generate texture for real mask: {e}")
            return None
    
    def save_synthetic_data(self, 
                           pairs: List[Tuple[np.ndarray, np.ndarray]], 
                           output_dir: str,
                           prefix: str = "synth") -> None:
        """
        Save synthetic image-mask pairs to disk
        
        Args:
            pairs: List of (image, mask) tuples
            output_dir: Output directory
            prefix: File prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        image_dir = os.path.join(output_dir, 'images')
        mask_dir = os.path.join(output_dir, 'labels')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        for i, (image, mask) in enumerate(pairs):
            # Save image
            if len(image.shape) == 3:
                image_pil = Image.fromarray(image.astype(np.uint8))
            else:
                image_pil = Image.fromarray(image.astype(np.uint8), mode='L')
            image_pil.save(os.path.join(image_dir, f"{prefix}_{i:05d}.png"))
            
            # Save mask
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            mask_pil.save(os.path.join(mask_dir, f"{prefix}_{i:05d}.png"))
        
        self.logger.info(f"Saved {len(pairs)} synthetic pairs to {output_dir}")


def test_cascaded_diffusion():
    """Test function for the cascaded diffusion wrapper"""
    wrapper = CascadedDiffusionWrapper()
    
    # Test device detection
    print(f"Using device: {wrapper.device}")
    
    # Test mask generation (without actual model)
    print("Cascaded diffusion wrapper initialized successfully")
    return wrapper


if __name__ == "__main__":
    test_cascaded_diffusion()
