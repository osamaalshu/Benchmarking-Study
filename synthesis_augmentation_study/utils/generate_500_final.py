#!/usr/bin/env python3
"""
Generate 500 synthetic image-mask pairs using trained pix2pix model

This script generates high-quality synthetic microscopy images and corresponding
segmentation masks using a trained pix2pix model. The generated data is used
in the controlled augmentation study.

Usage:
    python generate_500_final.py --model_name your_model --output_dir synthetic_data_500

Based on the training configuration:
- Model: pix2pix with unet_256 generator
- Resolution: 512x512
- Format: RGB images + grayscale masks
"""

import argparse
import subprocess
import random
import shutil
import json
from pathlib import Path
from PIL import Image
import numpy as np

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic microscopy data using trained pix2pix model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of trained pix2pix model (in checkpoints directory)"
    )
    parser.add_argument(
        "--output_dir", 
        default="synthetic_data_500",
        help="Output directory for generated synthetic data"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of synthetic image-mask pairs to generate"
    )
    parser.add_argument(
        "--dataset_dir",
        default="datasets/high_quality_pix2pix",
        help="Path to prepared pix2pix dataset"
    )
    parser.add_argument(
        "--checkpoints_dir",
        default="checkpoints",
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--external_dir",
        default="external",
        help="Directory containing pix2pix implementation"
    )
    
    # Model parameters (should match training configuration)
    parser.add_argument("--netG", default="unet_256", help="Generator architecture")
    parser.add_argument("--ngf", type=int, default=64, help="Generator filters")
    parser.add_argument("--ndf", type=int, default=64, help="Discriminator filters")
    parser.add_argument("--load_size", type=int, default=512, help="Input image size")
    parser.add_argument("--crop_size", type=int, default=512, help="Crop size")
    parser.add_argument("--direction", default="AtoB", help="Translation direction")
    
    return parser.parse_args()

def validate_setup(args):
    """Validate that all required directories and files exist"""
    
    print("🔍 Validating Setup...")
    
    # Check directories
    required_dirs = [
        (args.dataset_dir, "Dataset directory"),
        (args.checkpoints_dir, "Checkpoints directory"), 
        (args.external_dir, "Pix2pix external directory")
    ]
    
    for dir_path, name in required_dirs:
        if not Path(dir_path).exists():
            print(f"   ❌ {name}: {dir_path} - MISSING!")
            return False
        print(f"   ✅ {name}: {dir_path}")
    
    # Check model checkpoint
    model_checkpoint_dir = Path(args.checkpoints_dir) / args.model_name
    if not model_checkpoint_dir.exists():
        print(f"   ❌ Model checkpoint: {model_checkpoint_dir} - MISSING!")
        return False
    print(f"   ✅ Model checkpoint: {model_checkpoint_dir}")
    
    # Check pix2pix test script
    test_script = Path(args.external_dir) / "test.py"
    if not test_script.exists():
        print(f"   ❌ Pix2pix test script: {test_script} - MISSING!")
        return False
    print(f"   ✅ Pix2pix test script: {test_script}")
    
    # Check validation data
    val_dir = Path(args.dataset_dir) / "val"
    if not val_dir.exists():
        print(f"   ❌ Validation data: {val_dir} - MISSING!")
        return False
    
    val_images = list(val_dir.glob("*_AB.png"))
    if len(val_images) < args.num_samples:
        print(f"   ⚠️ Warning: Only {len(val_images)} validation images found, need {args.num_samples}")
        print(f"   Will use random sampling with replacement")
    else:
        print(f"   ✅ Validation data: {len(val_images)} images available")
    
    return True

def prepare_test_data(args):
    """Prepare temporary test dataset for generation"""
    
    print(f"\n📁 Preparing Test Data for {args.num_samples} samples...")
    
    # Get validation images
    val_dir = Path(args.dataset_dir) / "val"
    val_images = list(val_dir.glob("*_AB.png"))
    
    # Select samples
    if len(val_images) < args.num_samples:
        # Sample with replacement if not enough images
        selected_images = random.choices(val_images, k=args.num_samples)
        print(f"   Using {len(val_images)} unique images with replacement")
    else:
        # Sample without replacement
        selected_images = random.sample(val_images, args.num_samples)
        print(f"   Selected {args.num_samples} unique images")
    
    # Create temporary test directory
    temp_test_dir = Path(args.dataset_dir) / "temp_generation"
    temp_test_dir.mkdir(exist_ok=True)
    
    # Copy selected images with consistent naming
    for i, img_path in enumerate(selected_images):
        new_name = f"synthetic_{i:05d}_AB.png"
        shutil.copy(img_path, temp_test_dir / new_name)
    
    print(f"   ✅ Created temporary test set: {temp_test_dir}")
    return temp_test_dir

def run_pix2pix_generation(args, temp_test_dir):
    """Run pix2pix generation using the trained model"""
    
    print(f"\n🎨 Running Pix2Pix Generation...")
    
    # Create results directory
    results_dir = Path(args.output_dir) / "raw_generation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command with exact training parameters
    cmd = [
        "python", "test.py",
        "--model", "pix2pix",
        "--dataset_mode", "aligned", 
        "--dataroot", str(Path(args.dataset_dir).resolve()),
        "--name", args.model_name,
        "--direction", args.direction,
        "--results_dir", str(results_dir.resolve()),
        "--phase", "temp_generation",
        "--preprocess", "none",
        "--load_size", str(args.load_size),
        "--crop_size", str(args.crop_size),
        "--netG", args.netG,
        "--ngf", str(args.ngf),
        "--ndf", str(args.ndf),
        "--no_dropout"
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Working directory: {args.external_dir}")
    
    # Run generation
    try:
        result = subprocess.run(
            cmd,
            cwd=args.external_dir,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"   ✅ Generation completed successfully")
        
        # Check output
        output_dir = results_dir / args.model_name / "temp_generation" / "images"
        if output_dir.exists():
            generated_files = list(output_dir.glob("*fake*.png"))
            print(f"   Generated {len(generated_files)} images")
            return output_dir
        else:
            print(f"   ❌ Output directory not found: {output_dir}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Generation failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return None

def process_generated_images(args, generation_output_dir):
    """Process generated images into final format"""
    
    print(f"\n🔄 Processing Generated Images...")
    
    # Create final output directories
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "synthetic_images_500"
    labels_dir = output_dir / "synthetic_labels_grayscale"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find generated images
    fake_images = list(generation_output_dir.glob("*fake*.png"))
    real_images = list(generation_output_dir.glob("*real*.png"))
    
    if len(fake_images) == 0:
        print(f"   ❌ No generated images found in {generation_output_dir}")
        return False
    
    print(f"   Found {len(fake_images)} generated images")
    print(f"   Found {len(real_images)} real images (masks)")
    
    # Process each pair
    processed_count = 0
    for i, fake_img_path in enumerate(sorted(fake_images)):
        try:
            # Find corresponding real image (mask)
            base_name = fake_img_path.stem.replace("_fake_B", "")
            real_img_path = generation_output_dir / f"{base_name}_real_A.png"
            
            if not real_img_path.exists():
                print(f"   ⚠️ Warning: No mask found for {fake_img_path.name}")
                continue
            
            # Load and process images
            fake_img = Image.open(fake_img_path).convert("RGB")
            real_img = Image.open(real_img_path).convert("L")  # Grayscale mask
            
            # Resize if necessary (ensure 512x512)
            if fake_img.size != (512, 512):
                fake_img = fake_img.resize((512, 512), Image.LANCZOS)
            if real_img.size != (512, 512):
                real_img = real_img.resize((512, 512), Image.NEAREST)
            
            # Save with consistent naming
            fake_img.save(images_dir / f"synthetic_{processed_count:05d}.png")
            real_img.save(labels_dir / f"synthetic_{processed_count:05d}_label.png")
            
            processed_count += 1
            
        except Exception as e:
            print(f"   ⚠️ Error processing {fake_img_path.name}: {e}")
            continue
    
    print(f"   ✅ Processed {processed_count} image-mask pairs")
    
    # Save generation metadata
    metadata = {
        "generation_date": str(Path().cwd()),
        "model_name": args.model_name,
        "num_generated": processed_count,
        "target_samples": args.num_samples,
        "model_config": {
            "netG": args.netG,
            "ngf": args.ngf,
            "ndf": args.ndf,
            "load_size": args.load_size,
            "crop_size": args.crop_size,
            "direction": args.direction
        },
        "output_format": {
            "images": "RGB PNG 512x512",
            "labels": "Grayscale PNG 512x512",
            "naming": "synthetic_XXXXX.png / synthetic_XXXXX_label.png"
        }
    }
    
    with open(output_dir / "generation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return processed_count == args.num_samples

def cleanup_temp_files(temp_test_dir, generation_output_dir):
    """Clean up temporary files"""
    
    print(f"\n🧹 Cleaning Up Temporary Files...")
    
    if temp_test_dir and temp_test_dir.exists():
        shutil.rmtree(temp_test_dir)
        print(f"   ✅ Removed temporary test data: {temp_test_dir}")
    
    # Optionally keep raw generation for debugging
    # if generation_output_dir and generation_output_dir.exists():
    #     shutil.rmtree(generation_output_dir.parent)
    #     print(f"   ✅ Removed raw generation: {generation_output_dir.parent}")

def validate_output(args):
    """Validate the generated output"""
    
    print(f"\n✅ Validating Generated Output...")
    
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "synthetic_images_500"
    labels_dir = output_dir / "synthetic_labels_grayscale"
    
    # Count files
    image_files = list(images_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.png"))
    
    print(f"   Images generated: {len(image_files)}")
    print(f"   Labels generated: {len(label_files)}")
    
    # Check if we have the target number
    if len(image_files) >= args.num_samples and len(label_files) >= args.num_samples:
        print(f"   ✅ Successfully generated {args.num_samples} synthetic pairs")
        
        # Sample a few for quality check
        if len(image_files) > 0:
            sample_img = Image.open(image_files[0])
            sample_label = Image.open(label_files[0])
            print(f"   Sample image size: {sample_img.size}, mode: {sample_img.mode}")
            print(f"   Sample label size: {sample_label.size}, mode: {sample_label.mode}")
        
        return True
    else:
        print(f"   ❌ Expected {args.num_samples} pairs, got {len(image_files)} images and {len(label_files)} labels")
        return False

def main():
    """Main function"""
    
    # Parse arguments
    args = parse_arguments()
    
    print("🎨 Pix2Pix Synthetic Data Generation")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Samples: {args.num_samples}")
    print("")
    
    # Validate setup
    if not validate_setup(args):
        print("❌ Setup validation failed")
        return False
    
    try:
        # Step 1: Prepare test data
        temp_test_dir = prepare_test_data(args)
        
        # Step 2: Run generation
        generation_output_dir = run_pix2pix_generation(args, temp_test_dir)
        if generation_output_dir is None:
            return False
        
        # Step 3: Process generated images
        success = process_generated_images(args, generation_output_dir)
        if not success:
            return False
        
        # Step 4: Validate output
        if not validate_output(args):
            return False
        
        # Step 5: Cleanup
        cleanup_temp_files(temp_test_dir, generation_output_dir)
        
        print(f"\n🎉 Generation Complete!")
        print(f"📁 Synthetic data saved to: {args.output_dir}")
        print(f"📊 Ready for augmentation study!")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
