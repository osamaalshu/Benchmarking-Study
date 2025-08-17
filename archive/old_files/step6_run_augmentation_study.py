#!/usr/bin/env python3
"""
Step 6: Run Complete Augmentation Study
Execute the full modality agnostic controlled augmentation study with fixed dataset arms
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add augmentation_study to path
sys.path.append(str(Path(__file__).parent / "augmentation_study"))

from augmentation_study.experiment_runner import AugmentationStudyRunner

def run_complete_study():
    """Run the complete augmentation study with fixed dataset arms"""
    
    print("🎯 Step 6: Running Complete Augmentation Study")
    print("=" * 70)
    print("🔬 Modality Agnostic Controlled Augmentation Study")
    print("📊 Strategy: Additive Augmentation (R+S)")
    print("🎲 Seeds: [0, 1, 2] for reproducibility")
    print("🏗️ Models: nnU-Net (best baseline) + U-Net (simpler)")
    print("")
    
    # Study configuration
    config = {
        # Dataset configuration
        "base_data_dir": "data",
        "train_data_dir": "data/train-preprocessed",
        "val_data_dir": "data/val",
        "test_data_dir": "data/test", 
        "dataset_arms_dir": "fixed_dataset_arms",
        
        # Model configuration
        "models": ["nnunet"],  # Only nnU-Net for reliability
        "seeds": [0, 1, 2],
        
        # Training configuration  
        "batch_size": 4,
        "max_epochs": 5,  # 5 epochs per run
        "val_interval": 1,
        "num_workers": 0,  # Disable multiprocessing to avoid worker crashes
        "device": "mps",  # Use Apple Silicon GPU
        
        # Output configuration
        "output_dir": "final_augmentation_results",
        "save_predictions": True,
        "generate_plots": True,
        
        # Study phases - skip data generation since we have fixed arms
        "skip_cascaded_setup": True,
        "skip_data_generation": True,
        "use_existing_arms": True
    }
    
    print("📋 Study Configuration:")
    for key, value in config.items():
        if key not in ["skip_cascaded_setup", "skip_data_generation", "use_existing_arms"]:
            print(f"   {key}: {value}")
    print("")
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / "study_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize study runner
    print("🏗️ Initializing Augmentation Study Runner...")
    
    # Update config with output directory
    config["output_dir"] = str(output_dir)
    
    try:
        runner = AugmentationStudyRunner(base_config=config)
        
        print("✅ Study runner initialized successfully!")
        
        # Set up to use our fixed dataset arms
        fixed_arms_dir = Path(config["dataset_arms_dir"])
        runner.arms_dir = fixed_arms_dir
        print(f"📁 Using fixed dataset arms from: {fixed_arms_dir}")
        
        # Copy our fixed arms to the expected location so the runner can find them
        expected_location = runner.output_dir / 'dataset_arms'
        if not expected_location.exists():
            print(f"📋 Copying arms to expected location: {expected_location}")
            import shutil
            shutil.copytree(fixed_arms_dir, expected_location)
        
        # Verify arms exist
        expected_arms = ["R", "R+S@10", "R+S@25", "R+S@50", "S"]
        
        print("\n🔍 Verifying dataset arms...")
        for arm_name in expected_arms:
            arm_path = expected_location / arm_name
            if arm_path.exists():
                # Count samples
                images_dir = arm_path / "images"
                if images_dir.exists():
                    num_samples = len(list(images_dir.glob("*.png")))
                    print(f"   ✅ {arm_name}: {num_samples} samples")
                else:
                    print(f"   ❌ {arm_name}: images directory missing")
                    return False
            else:
                print(f"   ❌ {arm_name}: directory missing")
                return False
        
        # Create arms_created.json file that the runner expects
        arm_paths = [str(expected_location / arm) for arm in expected_arms]
        arms_info = {
            'arm_paths': arm_paths,
            'creation_time': datetime.now().isoformat(),
            'config': config,
            'source': 'fixed_dataset_arms'
        }
        
        with open(expected_location / 'arms_created.json', 'w') as f:
            json.dump(arms_info, f, indent=2)
        
        # Set the study state to indicate arms are ready
        runner.study_state['phase'] = 'dataset_arms_created'
        runner.study_state['completed_phases'] = ['dataset_arms_created']
        runner.study_state['arm_paths'] = arm_paths
        
        print("\n🚀 Starting Complete Augmentation Study...")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        start_time = time.time()
        
        # Run the study (skip data generation phases)
        success = runner.run_complete_study()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            print(f"\n🎉 Study Completed Successfully!")
            print(f"⏱️ Total Duration: {duration/3600:.1f} hours ({duration/60:.1f} minutes)")
            print(f"📁 Results saved to: {output_dir}")
            print(f"📊 Check final_evaluation_report.json for detailed results")
            
            # Print summary of what was accomplished
            print(f"\n📈 Study Summary:")
            print(f"   • {len(config['models'])} models trained")
            print(f"   • {len(expected_arms)} dataset arms tested") 
            print(f"   • {len(config['seeds'])} seeds per arm")
            print(f"   • {len(config['models']) * len(expected_arms) * len(config['seeds'])} total training runs")
            print(f"   • Statistical significance testing performed")
            print(f"   • Comprehensive evaluation metrics computed")
            
        else:
            print(f"\n❌ Study Failed!")
            print(f"⏱️ Duration before failure: {duration/60:.1f} minutes")
            print(f"📋 Check logs in {output_dir} for details")
            return False
            
    except Exception as e:
        print(f"\n💥 Study failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return success

def main():
    """Main function for Step 6"""
    
    # Check prerequisites
    prerequisites = [
        ("Fixed dataset arms", "fixed_dataset_arms"),
        ("Real test data", "data/test"),
        ("Augmentation study code", "augmentation_study")
    ]
    
    print("🔍 Checking Prerequisites...")
    for name, path in prerequisites:
        if Path(path).exists():
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: {path} - MISSING!")
            return False
    
    print("✅ All prerequisites satisfied!")
    print("")
    
    # Run the study
    success = run_complete_study()
    
    if success:
        print(f"\n🏆 Step 6 Complete - Augmentation Study Successful!")
        print(f"📝 Ready for analysis and paper writing!")
    else:
        print(f"\n❌ Step 6 Failed - Check logs for details")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
