#!/usr/bin/env python3
"""
Main script to run the Modality Agnostic Controlled Augmentation Study
Using Cascaded Diffusion for Microscopy Image Synthesis

This script orchestrates the complete study including:
1. Setting up cascaded diffusion model
2. Creating dataset arms (R, RxS@10/25/50, S, Rmask+SynthTex@25)
3. Training models (nnUNet + U-Net) on all arms with seeds {0,1,2}
4. Comprehensive evaluation with statistical analysis
5. Final report generation

Usage:
    python run_augmentation_study.py --train-data ./data/train-preprocessed --val-data ./data/val --test-data ./data/test
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add augmentation study to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'augmentation_study'))

from augmentation_study.experiment_runner import AugmentationStudyRunner, create_default_config


def setup_study_environment():
    """Setup environment for the augmentation study"""
    print("Setting up augmentation study environment...")
    
    # Check if cascaded diffusion requirements are installed
    try:
        import diffusers
        import transformers
        print("✓ Diffusion libraries available")
    except ImportError as e:
        print(f"⚠ Warning: Diffusion libraries not fully available: {e}")
        print("Please install: pip install -r augmentation_study/requirements_cascaded.txt")
    
    # Check data directories
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    
    required_dirs = ['train-preprocessed', 'val', 'test']
    for req_dir in required_dirs:
        dir_path = data_dir / req_dir
        if dir_path.exists():
            print(f"✓ Found {req_dir} directory")
        else:
            print(f"⚠ Warning: {req_dir} directory not found at {dir_path}")
    
    print("Environment check completed.\n")


def create_study_config(args) -> dict:
    """Create configuration for the study based on arguments"""
    config = create_default_config()
    
    # Update paths
    if args.train_data:
        config['train_data_dir'] = args.train_data
    if args.val_data:
        config['val_data_dir'] = args.val_data
    if args.test_data:
        config['test_data_dir'] = args.test_data
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Update model configuration
    if args.models:
        config['models'] = args.models
    if args.seeds:
        config['seeds'] = args.seeds
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.max_epochs:
        config['max_epochs'] = args.max_epochs
    
    # Cascaded diffusion model paths
    if args.mask_model:
        config['mask_model_path'] = args.mask_model
    if args.texture_model:
        config['texture_model_path'] = args.texture_model
    
    # Other settings
    config['debug'] = args.debug
    config['device'] = args.device
    
    return config


def print_study_overview(config):
    """Print overview of the study configuration"""
    print("=" * 70)
    print("MODALITY AGNOSTIC CONTROLLED AUGMENTATION STUDY")
    print("Using Cascaded Diffusion for Microscopy Image Synthesis")
    print("=" * 70)
    print()
    
    print("📊 STUDY CONFIGURATION:")
    print(f"  • Models: {', '.join(config['models'])}")
    print(f"  • Seeds: {config['seeds']}")
    print(f"  • Batch size: {config['batch_size']}")
    print(f"  • Max epochs: {config['max_epochs']}")
    print(f"  • Device: {config['device']}")
    print()
    
    print("📁 DATA DIRECTORIES:")
    print(f"  • Training: {config['train_data_dir']}")
    print(f"  • Validation: {config['val_data_dir']}")
    print(f"  • Test: {config['test_data_dir']}")
    print(f"  • Output: {config['output_dir']}")
    print()
    
    print("🔬 DATASET ARMS TO BE CREATED:")
    print("  • R (Real-only): Original training set")
    print("  • RxS@10: Replace 10% with synthetic")
    print("  • RxS@25: Replace 25% with synthetic")
    print("  • RxS@50: Replace 50% with synthetic")
    print("  • S (Synthetic-only): 100% synthetic")
    print("  • Rmask+SynthTex@25: Real masks + 25% synthetic textures")
    print()
    
    print("📈 EVALUATION METRICS:")
    print("  • Dice coefficient")
    print("  • IoU (Intersection over Union)")
    print("  • Precision & Recall")
    print("  • F1 Score")
    print("  • Boundary F1")
    print("  • Hausdorff Distance 95th percentile")
    print()
    
    print("📊 STATISTICAL ANALYSIS:")
    print("  • Paired t-tests (R vs each arm)")
    print("  • Effect size (Cohen's d)")
    print("  • Bonferroni correction for multiple comparisons")
    print("  • 95% confidence intervals")
    print()
    
    print("=" * 70)
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run Modality Agnostic Controlled Augmentation Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete study with default settings
  python run_augmentation_study.py
  
  # Run with custom data paths
  python run_augmentation_study.py --train-data ./data/train-preprocessed --val-data ./data/val --test-data ./data/test
  
  # Run with specific models and seeds
  python run_augmentation_study.py --models nnunet --seeds 0 1 2
  
  # Run with custom output directory
  python run_augmentation_study.py --output-dir ./my_study_results
  
  # Resume from a specific phase
  python run_augmentation_study.py --resume-from training_completed
        """
    )
    
    # Data directories
    parser.add_argument('--train-data', type=str, default='./data/train-preprocessed',
                       help='Path to training data directory')
    parser.add_argument('--val-data', type=str, default='./data/val',
                       help='Path to validation data directory')
    parser.add_argument('--test-data', type=str, default='./data/test',
                       help='Path to test data directory')
    parser.add_argument('--output-dir', type=str, default='./augmentation_study_results',
                       help='Output directory for study results')
    
    # Model configuration
    parser.add_argument('--models', nargs='+', default=['nnunet', 'unet'],
                       choices=['nnunet', 'unet'],
                       help='Models to train and evaluate')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                       help='Random seeds for reproducibility')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--max-epochs', type=int, default=30,
                       help='Maximum training epochs')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use for training')
    
    # Cascaded diffusion models (optional)
    parser.add_argument('--mask-model', type=str,
                       help='Path to trained mask generation model (DDPM checkpoint)')
    parser.add_argument('--texture-model', type=str,
                       help='Path to trained texture generation model (optional)')
    
    # Execution options
    parser.add_argument('--resume-from', type=str,
                       choices=['cascaded_diffusion_setup', 'dataset_arms_created', 
                               'training_completed', 'evaluation_completed'],
                       help='Resume study from specific phase')
    parser.add_argument('--config', type=str,
                       help='Path to JSON configuration file')
    parser.add_argument('--save-config', type=str,
                       help='Save configuration to JSON file and exit')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration and exit without running')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_study_environment()
    
    # Create configuration
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_study_config(args)
    
    # Save configuration if requested
    if args.save_config:
        with open(args.save_config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {args.save_config}")
        return 0
    
    # Print study overview
    print_study_overview(config)
    
    # Dry run - just show configuration
    if args.dry_run:
        print("🔍 DRY RUN - Configuration shown above")
        print("Run without --dry-run to execute the study")
        return 0
    
    # Confirm execution
    if not args.debug:
        response = input("Proceed with the study? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Study cancelled.")
            return 0
    
    print("🚀 Starting Modality Agnostic Controlled Augmentation Study...")
    print()
    
    # Create and run study
    try:
        runner = AugmentationStudyRunner(config)
        
        if args.resume_from:
            print(f"📍 Resuming from phase: {args.resume_from}")
            success = runner.resume_from_phase(args.resume_from)
        else:
            success = runner.run_complete_study()
        
        if success:
            print("\n" + "=" * 70)
            print("🎉 STUDY COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"📁 Results saved to: {config['output_dir']}")
            print(f"📋 Study summary: {config['output_dir']}/STUDY_SUMMARY.md")
            print(f"📊 Statistical results: {config['output_dir']}/evaluation_results/statistical_summary.csv")
            print(f"📈 Detailed report: {config['output_dir']}/evaluation_results/evaluation_report.md")
            print()
            print("🔬 Key files to examine:")
            print("  • STUDY_SUMMARY.md - High-level overview")
            print("  • evaluation_results/statistical_summary.csv - Statistical comparison")
            print("  • evaluation_results/evaluation_report.md - Detailed analysis")
            print("  • study_state.json - Complete study metadata")
            print("=" * 70)
            return 0
        else:
            print("\n❌ Study failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹ Study interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
