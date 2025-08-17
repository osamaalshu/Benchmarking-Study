#!/usr/bin/env python3
"""
Modality Agnostic Controlled Augmentation Study - Main Entry Point

This script runs the complete augmentation study evaluating the effectiveness
of synthetic data for microscopy image segmentation.

Usage:
    python augmentation_study/run_study.py [options]

Example:
    python augmentation_study/run_study.py --models nnunet --epochs 5 --device mps
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from synthesis_augmentation_study.experiment_runner import AugmentationStudyRunner

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run Modality Agnostic Controlled Augmentation Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument(
        "--train-data-dir", 
        default="data/train-preprocessed",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--val-data-dir",
        default="data/val", 
        help="Directory containing validation data"
    )
    parser.add_argument(
        "--test-data-dir",
        default="data/test",
        help="Directory containing test data"
    )
    parser.add_argument(
        "--synthetic-data-dir",
        default="synthesis/synthetic_data_500",
        help="Directory containing synthetic data"
    )
    
    # Model configuration
    parser.add_argument(
        "--models",
        nargs="+",
        default=["nnunet"],
        choices=["nnunet", "unet"],
        help="Models to train and evaluate"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds for reproducibility"
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs per run"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device for training (auto=detect best available)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0=disable multiprocessing)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        default="augmentation_study_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=True,
        help="Save model predictions on test set"
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true", 
        default=True,
        help="Generate visualization plots"
    )
    
    # Study phases
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Skip synthetic data generation (use existing)"
    )
    parser.add_argument(
        "--arms-only",
        action="store_true",
        help="Only create dataset arms, don't train models"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation on existing models"
    )
    
    return parser.parse_args()

def create_study_config(args):
    """Create study configuration from arguments"""
    
    config = {
        # Data paths
        "train_data_dir": args.train_data_dir,
        "val_data_dir": args.val_data_dir,
        "test_data_dir": args.test_data_dir,
        "synthetic_data_dir": args.synthetic_data_dir,
        
        # Model configuration
        "models": args.models,
        "seeds": args.seeds,
        
        # Training parameters
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "num_workers": args.num_workers,
        
        # Output settings
        "output_dir": args.output_dir,
        "save_predictions": args.save_predictions,
        "generate_plots": args.generate_plots,
        
        # Study phases
        "skip_data_generation": args.skip_data_generation,
        "arms_only": args.arms_only,
        "evaluate_only": args.evaluate_only,
        
        # Fixed parameters
        "val_interval": 1,
        "initial_lr": 6e-4,
        "input_size": 256,
        "num_classes": 3,
    }
    
    return config

def validate_prerequisites(config):
    """Validate that all required data and directories exist"""
    
    print("🔍 Validating Prerequisites...")
    
    # Check data directories
    required_dirs = [
        (config["train_data_dir"], "Training data"),
        (config["val_data_dir"], "Validation data"), 
        (config["test_data_dir"], "Test data"),
    ]
    
    if not config.get("skip_data_generation", False):
        required_dirs.append((config["synthetic_data_dir"], "Synthetic data"))
    
    for dir_path, name in required_dirs:
        if not Path(dir_path).exists():
            print(f"   ❌ {name}: {dir_path} - MISSING!")
            return False
        else:
            print(f"   ✅ {name}: {dir_path}")
    
    # Check augmentation study code
    study_dir = Path("augmentation_study")
    if not study_dir.exists():
        print(f"   ❌ Augmentation study code: {study_dir} - MISSING!")
        return False
    else:
        print(f"   ✅ Augmentation study code: {study_dir}")
    
    print("✅ All prerequisites satisfied!")
    return True

def print_study_overview(config):
    """Print study overview and configuration"""
    
    print("\n🎯 Modality Agnostic Controlled Augmentation Study")
    print("=" * 70)
    print("🔬 Evaluating synthetic data effectiveness for cell segmentation")
    print("")
    
    # Dataset arms
    print("📊 Dataset Arms:")
    print("   • R (Real-only): 900 real images (baseline)")
    print("   • R+S@10: 900 real + 90 synthetic = 990 total")  
    print("   • R+S@25: 900 real + 225 synthetic = 1,125 total")
    print("   • R+S@50: 900 real + 450 synthetic = 1,350 total")
    print("   • S (Synthetic-only): 450 synthetic images")
    print("")
    
    # Training configuration
    print("🏗️ Training Configuration:")
    print(f"   • Models: {', '.join(config['models'])}")
    print(f"   • Seeds: {config['seeds']} (for statistical validity)")
    print(f"   • Epochs: {config['max_epochs']} per training run")
    print(f"   • Batch size: {config['batch_size']}")
    print(f"   • Device: {config['device']}")
    print("")
    
    # Expected runtime
    num_runs = len(config['models']) * 5 * len(config['seeds'])  # 5 arms
    estimated_time = num_runs * 5 * config['max_epochs'] / 60  # ~5 min per epoch
    
    print("⏱️ Expected Runtime:")
    print(f"   • Total training runs: {num_runs}")
    print(f"   • Estimated time: ~{estimated_time:.1f} minutes")
    print("")

def main():
    """Main function"""
    
    # Parse arguments
    args = parse_arguments()
    config = create_study_config(args)
    
    # Print overview
    print_study_overview(config)
    
    # Validate prerequisites
    if not validate_prerequisites(config):
        print("\n❌ Prerequisites not met. Please fix issues above.")
        return False
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = output_dir / "study_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📝 Configuration saved: {config_file}")
    
    print(f"\n🚀 Starting Study...")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    start_time = time.time()
    
    try:
        # Initialize and run study
        runner = AugmentationStudyRunner(base_config=config)
        
        if config.get("arms_only", False):
            # Only create dataset arms
            success = runner.create_dataset_arms()
            phase_name = "Dataset Arms Creation"
            
        elif config.get("evaluate_only", False):
            # Only run evaluation
            success = runner.evaluate_all_results()
            phase_name = "Evaluation"
            
        else:
            # Run complete study
            success = runner.run_complete_study()
            phase_name = "Complete Study"
        
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            print(f"\n🎉 {phase_name} Completed Successfully!")
            print(f"⏱️ Total Duration: {duration/3600:.2f} hours ({duration/60:.1f} minutes)")
            print(f"📁 Results saved to: {output_dir}")
            
            # Print key result files
            print(f"\n📊 Key Result Files:")
            
            if (output_dir / "evaluation_results" / "evaluation_report.md").exists():
                print(f"   📈 Detailed Analysis: evaluation_results/evaluation_report.md")
            if (output_dir / "evaluation_results" / "statistical_summary.csv").exists():
                print(f"   📊 Statistical Results: evaluation_results/statistical_summary.csv")
            if (output_dir / "STUDY_SUMMARY.md").exists():
                print(f"   📝 Executive Summary: STUDY_SUMMARY.md")
            
            print(f"\n🏆 Study completed successfully!")
            
        else:
            print(f"\n❌ {phase_name} Failed!")
            print(f"⏱️ Duration before failure: {duration/60:.1f} minutes")
            print(f"📋 Check logs in {output_dir} for details")
            return False
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Study interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n💥 Study failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
