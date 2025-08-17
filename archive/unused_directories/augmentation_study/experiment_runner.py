"""
Experiment Runner for Modality Agnostic Controlled Augmentation Study

Main orchestrator that:
1. Sets up cascaded diffusion model
2. Creates all dataset arms
3. Trains all models on all arms with all seeds
4. Evaluates all results
5. Generates comprehensive reports
"""

import os
import sys
import argparse
import logging
import json
from augmentation_study.evaluation_framework import NumpyEncoder
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
from datetime import datetime
import traceback

# Import study components
from .cascaded_diffusion_wrapper import CascadedDiffusionWrapper
from .data_arms_manager import DataArmsManager
from .training_protocol import MultiModelTrainer
from .evaluation_framework import ComprehensiveEvaluator


class AugmentationStudyRunner:
    """Main experiment runner for the augmentation study"""
    
    def __init__(self, 
                 base_config: Dict[str, Any]):
        """
        Initialize the experiment runner
        
        Args:
            base_config: Base configuration for the experiment
        """
        self.config = base_config
        
        # Setup logging
        log_level = logging.DEBUG if base_config.get('debug', False) else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(base_config['output_dir']) / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.output_dir = Path(base_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.arms_dir = self.output_dir / 'dataset_arms'
        self.training_dir = self.output_dir / 'training_results'
        self.evaluation_dir = self.output_dir / 'evaluation_results'
        
        # Initialize components
        self.cascaded_model = None
        self.data_arms_manager = None
        self.trainer = None
        self.evaluator = None
        
        # Study state
        self.study_state = {
            'phase': 'initialized',
            'completed_phases': [],
            'start_time': datetime.now().isoformat(),
            'config': base_config
        }
        
        self.logger.info("Augmentation Study Runner initialized")
    
    def setup_cascaded_diffusion(self) -> bool:
        """Setup the cascaded diffusion model"""
        try:
            self.logger.info("Setting up Cascaded Diffusion model...")
            
            self.cascaded_model = CascadedDiffusionWrapper(
                mask_model_path=self.config.get('mask_model_path'),
                texture_model_path=self.config.get('texture_model_path'),
                device=self.config.get('device', 'auto')
            )
            
            # Setup generators - always setup mask generator (will fallback if no model)
            self.cascaded_model.setup_mask_generator()
            
            if self.config.get('texture_model_path'):
                self.cascaded_model.setup_texture_generator()
            else:
                # Use base Stable Diffusion
                self.cascaded_model.setup_texture_generator()
            
            self.study_state['phase'] = 'cascaded_diffusion_setup'
            self.study_state['completed_phases'].append('cascaded_diffusion_setup')
            
            self.logger.info("Cascaded Diffusion model setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup Cascaded Diffusion model: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def create_dataset_arms(self) -> bool:
        """Create all dataset arms"""
        try:
            self.logger.info("Creating dataset arms...")
            
            self.data_arms_manager = DataArmsManager(
                base_data_dir=self.config['train_data_dir'],
                output_dir=str(self.arms_dir),
                cascaded_model=self.cascaded_model,
                seed=self.config.get('seed', 42)
            )
            
            # Create all arms
            arm_paths = self.data_arms_manager.create_all_arms()
            
            # Save arm information
            with open(self.arms_dir / 'arms_created.json', 'w') as f:
                json.dump({
                    'arm_paths': arm_paths,
                    'creation_time': datetime.now().isoformat(),
                    'config': self.config
                }, f, indent=2)
            
            self.study_state['phase'] = 'dataset_arms_created'
            self.study_state['completed_phases'].append('dataset_arms_created')
            self.study_state['arm_paths'] = arm_paths
            
            self.logger.info(f"Created {len(arm_paths)} dataset arms")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create dataset arms: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def train_all_models(self) -> bool:
        """Train all models on all arms with all seeds"""
        try:
            self.logger.info("Starting training phase...")
            
            # Setup trainer
            self.trainer = MultiModelTrainer(
                models=self.config.get('models', ['nnunet', 'unet']),
                seeds=self.config.get('seeds', [0, 1, 2]),
                batch_size=self.config.get('batch_size', 4),
                initial_lr=self.config.get('initial_lr', 6e-4),
                max_epochs=self.config.get('max_epochs', 30),
                input_size=self.config.get('input_size', 256),
                num_classes=self.config.get('num_classes', 3),
                device=self.config.get('device', 'auto')
            )
            
            # Train all models on all arms
            training_results = self.trainer.train_all_models_all_arms(
                arms_dir=str(self.arms_dir),
                val_dir=self.config['val_data_dir'],
                output_dir=str(self.training_dir)
            )
            
            self.study_state['phase'] = 'training_completed'
            self.study_state['completed_phases'].append('training_completed')
            self.study_state['training_results'] = training_results
            
            self.logger.info("Training phase completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed during training phase: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def evaluate_all_results(self) -> bool:
        """Evaluate all trained models"""
        try:
            self.logger.info("Starting evaluation phase...")
            
            # Setup evaluator
            self.evaluator = ComprehensiveEvaluator(
                test_data_dir=self.config['test_data_dir'],
                results_output_dir=str(self.evaluation_dir)
            )
            
            # Evaluate all models
            evaluation_results = self.evaluator.evaluate_all_models_all_arms(
                training_results_dir=str(self.training_dir),
                models=self.config.get('models', ['nnunet', 'unet']),
                seeds=self.config.get('seeds', [0, 1, 2])
            )
            
            self.study_state['phase'] = 'evaluation_completed'
            self.study_state['completed_phases'].append('evaluation_completed')
            self.study_state['evaluation_results'] = evaluation_results
            
            self.logger.info("Evaluation phase completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed during evaluation phase: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def generate_final_report(self) -> bool:
        """Generate final comprehensive report"""
        try:
            self.logger.info("Generating final report...")
            
            # Update study state
            self.study_state['phase'] = 'completed'
            self.study_state['completed_phases'].append('report_generated')
            self.study_state['end_time'] = datetime.now().isoformat()
            
            # Calculate total time
            start_time = datetime.fromisoformat(self.study_state['start_time'])
            end_time = datetime.fromisoformat(self.study_state['end_time'])
            total_time = (end_time - start_time).total_seconds()
            
            self.study_state['total_time_seconds'] = total_time
            self.study_state['total_time_hours'] = total_time / 3600
            
            # Save final study state
            with open(self.output_dir / 'study_state.json', 'w') as f:
                json.dump(self.study_state, f, indent=2, cls=NumpyEncoder)
            
            # Create summary report
            self._create_summary_report()
            
            self.logger.info(f"Study completed successfully in {total_time/3600:.2f} hours")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate final report: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _create_summary_report(self):
        """Create high-level summary report"""
        report_lines = [
            "# Modality Agnostic Controlled Augmentation Study",
            "## Using Cascaded Diffusion for Microscopy Image Synthesis",
            "",
            f"**Study Completion Date:** {self.study_state['end_time']}",
            f"**Total Duration:** {self.study_state['total_time_hours']:.2f} hours",
            f"**Models Evaluated:** {', '.join(self.config.get('models', []))}",
            f"**Seeds Used:** {self.config.get('seeds', [])}",
            "",
            "## Study Overview",
            "",
            "This study evaluates the effectiveness of synthetic data augmentation using",
            "cascaded diffusion models for cell microscopy image segmentation.",
            "",
            "### Dataset Arms Evaluated",
            "- **R (Real-only):** Original training set as baseline",
            "- **RxS@10:** 10% synthetic replacement",
            "- **RxS@25:** 25% synthetic replacement", 
            "- **RxS@50:** 50% synthetic replacement",
            "- **S (Synthetic-only):** 100% synthetic data",
            "- **Rmask+SynthTex@25:** Real masks with 25% synthetic textures",
            "",
            "## Key Findings",
            "",
            "### Statistical Significance",
            "Results are based on paired t-tests with Bonferroni correction for multiple comparisons.",
            "",
            "### Performance Summary",
            "Detailed results are available in:",
            f"- Statistical summary: `{self.evaluation_dir}/statistical_summary.csv`",
            f"- Comprehensive evaluation: `{self.evaluation_dir}/evaluation_report.md`",
            "",
            "## Files Generated",
            "",
            "### Dataset Arms",
            f"- Location: `{self.arms_dir}/`",
            "- Contains all generated dataset arms with metadata",
            "",
            "### Training Results", 
            f"- Location: `{self.training_dir}/`",
            "- Contains trained models and training logs for all arms and seeds",
            "",
            "### Evaluation Results",
            f"- Location: `{self.evaluation_dir}/`",
            "- Contains comprehensive evaluation metrics and statistical analysis",
            "",
            "## Reproducibility",
            "",
            f"This study used fixed seeds ({self.config.get('seeds', [])}) and standardized",
            "hyperparameters across all arms to ensure fair comparison.",
            "",
            "Configuration details are saved in `study_state.json`.",
            "",
            "## Citation",
            "",
            "If you use this work, please cite:",
            "- The cascaded diffusion paper: Yilmaz et al. (2024)",
            "- This benchmarking study: [Your study details]"
        ]
        
        # Save summary report
        with open(self.output_dir / 'STUDY_SUMMARY.md', 'w') as f:
            f.write('\n'.join(report_lines))
    
    def run_complete_study(self) -> bool:
        """Run the complete augmentation study"""
        self.logger.info("Starting complete augmentation study...")
        
        phases = [
            # ('Setup Cascaded Diffusion', self.setup_cascaded_diffusion),  # Skip - already done
            # ('Create Dataset Arms', self.create_dataset_arms),  # Skip - data already generated
            ('Train All Models', self.train_all_models),
            ('Evaluate All Results', self.evaluate_all_results),
            ('Generate Final Report', self.generate_final_report)
        ]
        
        for phase_name, phase_func in phases:
            self.logger.info(f"Starting phase: {phase_name}")
            
            phase_start = time.time()
            success = phase_func()
            phase_time = time.time() - phase_start
            
            if success:
                self.logger.info(f"Completed phase: {phase_name} ({phase_time:.1f}s)")
            else:
                self.logger.error(f"Failed phase: {phase_name}")
                return False
        
        self.logger.info("Complete augmentation study finished successfully!")
        return True
    
    def resume_from_phase(self, phase: str) -> bool:
        """Resume study from a specific phase"""
        # Load previous state if available
        state_file = self.output_dir / 'study_state.json'
        if state_file.exists():
            with open(state_file, 'r') as f:
                self.study_state = json.load(f)
        
        # Map phases to functions
        phase_map = {
            'cascaded_diffusion_setup': self.setup_cascaded_diffusion,
            'dataset_arms_created': self.create_dataset_arms,
            'training_completed': self.train_all_models,
            'evaluation_completed': self.evaluate_all_results,
            'report_generated': self.generate_final_report
        }
        
        if phase not in phase_map:
            self.logger.error(f"Unknown phase: {phase}")
            return False
        
        # Run from specified phase
        phases_to_run = list(phase_map.keys())[list(phase_map.keys()).index(phase):]
        
        for phase_name in phases_to_run:
            if phase_name not in self.study_state.get('completed_phases', []):
                self.logger.info(f"Running phase: {phase_name}")
                success = phase_map[phase_name]()
                if not success:
                    return False
        
        return True


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for the study"""
    return {
        # Data directories
        'train_data_dir': './data/train-preprocessed',
        'val_data_dir': './data/val', 
        'test_data_dir': './data/test',
        'output_dir': './augmentation_study_results',
        
        # Cascaded diffusion model paths (optional)
        'mask_model_path': None,  # Path to trained mask generation model
        'texture_model_path': None,  # Path to trained texture generation model
        
        # Training configuration
        'models': ['nnunet', 'unet'],
        'seeds': [0, 1, 2],
        'batch_size': 4,
        'initial_lr': 6e-4,
        'max_epochs': 30,
        'input_size': 256,
        'num_classes': 3,
        'device': 'auto',
        
        # Study configuration
        'seed': 42,
        'debug': False
    }


def main():
    """Main entry point for the augmentation study"""
    parser = argparse.ArgumentParser(description='Run Modality Agnostic Controlled Augmentation Study')
    
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--train-data', type=str, help='Path to training data directory')
    parser.add_argument('--val-data', type=str, help='Path to validation data directory')
    parser.add_argument('--test-data', type=str, help='Path to test data directory')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--models', nargs='+', default=['nnunet', 'unet'], help='Models to train')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2], help='Random seeds')
    parser.add_argument('--resume-from', type=str, help='Resume from specific phase')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    if args.train_data:
        config['train_data_dir'] = args.train_data
    if args.val_data:
        config['val_data_dir'] = args.val_data
    if args.test_data:
        config['test_data_dir'] = args.test_data
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.models:
        config['models'] = args.models
    if args.seeds:
        config['seeds'] = args.seeds
    if args.debug:
        config['debug'] = True
    
    # Create and run study
    runner = AugmentationStudyRunner(config)
    
    if args.resume_from:
        success = runner.resume_from_phase(args.resume_from)
    else:
        success = runner.run_complete_study()
    
    if success:
        print(f"\nStudy completed successfully!")
        print(f"Results saved to: {config['output_dir']}")
        print(f"See STUDY_SUMMARY.md for overview")
    else:
        print(f"\nStudy failed. Check logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
