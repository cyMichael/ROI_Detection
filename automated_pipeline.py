#!/usr/bin/env python3
"""
Automated ROI Detection Pipeline
This script automates the entire ROI detection workflow from patch extraction to analysis.
"""

import os
import sys
import yaml
import logging
import subprocess
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

class ROIDetectionPipeline:
    """Main pipeline class for automated ROI detection."""
    
    def __init__(self, config_path: str):
        """Initialize the pipeline with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
        
        # Create output directories
        self._create_directories()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        if log_config.get('log_file'):
            file_handler = logging.FileHandler(log_config['log_file'])
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        
        # Setup console handler
        if log_config.get('log_to_console', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)
        
        logging.getLogger().setLevel(log_level)
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_paths = [
            'data.data_dir',
            'data.csv_path',
            'data.wsi_dir',
            'data.xml_annotation_new',
            'data.xml_annotation_other',
            'data.feat_dir',
            'data.results_dir'
        ]
        
        for path_key in required_paths:
            path_value = self._get_nested_value(self.config, path_key)
            if not path_value or path_value.startswith('PATH_TO_'):
                raise ValueError(f"Please configure {path_key} in config.yaml")
    
    def _get_nested_value(self, config: Dict, key: str) -> Any:
        """Get nested value from configuration using dot notation."""
        keys = key.split('.')
        value = config
        for k in keys:
            value = value.get(k, {})
        return value
    
    def _create_directories(self):
        """Create necessary output directories."""
        directories = [
            self.config['data']['feat_dir'],
            self.config['data']['results_dir'],
            self.config['data']['models_save_folder'],
            self.config['data']['classification_save_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def _run_command(self, command: list, step_name: str) -> bool:
        """Run a command and handle errors."""
        self.logger.info(f"Starting {step_name}...")
        self.logger.info(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.info(f"{step_name} completed successfully")
            if result.stdout:
                self.logger.debug(f"Output: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{step_name} failed with return code {e.returncode}")
            self.logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in {step_name}: {e}")
            return False
    
    def _check_output_exists(self, output_path: str, step_name: str) -> bool:
        """Check if output already exists and should be skipped."""
        if not self.config['pipeline'].get('skip_existing', True):
            return False
        
        if os.path.exists(output_path):
            self.logger.info(f"Skipping {step_name} - output already exists: {output_path}")
            return True
        return False
    
    def extract_patches(self) -> bool:
        """Step 1: Extract patches from whole slide images."""
        if not self.config['pipeline'].get('extract_patches', True):
            self.logger.info("Skipping patch extraction (disabled in config)")
            return True
        
        # Check if patches already exist
        patch_count_file = os.path.join(self.config['data']['feat_dir'], 'patch_count.txt')
        if self._check_output_exists(patch_count_file, "patch extraction"):
            return True
        
        command = [
            'python', 'extract_patches_3class.py',
            '--data_dir', self.config['data']['data_dir'],
            '--csv_path', self.config['data']['csv_path'],
            '--xml_annotation_new', self.config['data']['xml_annotation_new'],
            '--xml_annotation_other', self.config['data']['xml_annotation_other'],
            '--feat_dir', self.config['data']['feat_dir'],
            '--patch_size', str(self.config['patches']['patch_size']),
            '--target_patch_size', str(self.config['patches']['target_patch_size'])
        ]
        
        if self.config['pipeline'].get('skip_existing', True):
            command.append('--auto_skip')
        
        return self._run_command(command, "Patch Extraction")
    
    def train_model(self) -> bool:
        """Step 2: Train the patch classification model."""
        if not self.config['pipeline'].get('train_model', True):
            self.logger.info("Skipping model training (disabled in config)")
            return True
        
        # Check if model already exists
        model_pattern = f"{self.config['experiment']['name']}_epoch*_loss*_acc*.pt"
        models_dir = Path(self.config['data']['models_save_folder'])
        existing_models = list(models_dir.glob(model_pattern))
        
        if existing_models and self.config['pipeline'].get('skip_existing', True):
            self.logger.info(f"Skipping model training - found existing models: {existing_models}")
            return True
        
        command = [
            'python', 'method_pcla_3class.py',
            '--exp_name', self.config['experiment']['name'],
            '--data_folder', self.config['data']['feat_dir'],
            '--batch_size', str(self.config['model']['batch_size']),
            '--n_epochs', str(self.config['model']['n_epochs']),
            '--lr_cla', str(self.config['model']['learning_rate']),
            '--weight_decay_cla', str(self.config['model']['weight_decay']),
            '--patience', str(self.config['model']['patience']),
            '--models_save_folder', self.config['data']['models_save_folder'],
            '--num_class', str(self.config['model']['num_classes'])
        ]
        
        return self._run_command(command, "Model Training")
    
    def score_patches(self) -> bool:
        """Step 3: Calculate predicted scores for all patches."""
        if not self.config['pipeline'].get('score_patches', True):
            self.logger.info("Skipping patch scoring (disabled in config)")
            return True
        
        # Find the best model
        model_path = self._find_best_model()
        if not model_path:
            self.logger.error("No trained model found for scoring")
            return False
        
        # Check if scores already exist
        scores_dir = os.path.join(self.config['data']['results_dir'], 
                                 self.config['experiment']['name'], 'score')
        if self._check_output_exists(scores_dir, "patch scoring"):
            return True
        
        command = [
            'python', 'score_pcla_3class.py',
            '--exp_name', self.config['experiment']['name'],
            '--model_load', os.path.basename(model_path),
            '--csv_path', self.config['data']['csv_path'],
            '--patch_path', self.config['data']['data_dir'],
            '--batch_size', str(self.config['model']['batch_size']),
            '--results_dir', self.config['data']['results_dir'],
            '--classification_save_dir', self.config['data']['classification_save_dir'],
            '--models_save_folder', self.config['data']['models_save_folder'],
            '--num_class', str(self.config['model']['num_classes']),
            '--mean1', str(self.config['normalization']['mean'][0]),
            '--mean2', str(self.config['normalization']['mean'][1]),
            '--mean3', str(self.config['normalization']['mean'][2]),
            '--std1', str(self.config['normalization']['std'][0]),
            '--std2', str(self.config['normalization']['std'][1]),
            '--std3', str(self.config['normalization']['std'][2])
        ]
        
        if self.config['pipeline'].get('skip_existing', True):
            command.append('--auto_skip')
        
        return self._run_command(command, "Patch Scoring")
    
    def generate_visualizations(self) -> bool:
        """Step 4: Generate visualization maps."""
        if not self.config['pipeline'].get('generate_visualizations', True):
            self.logger.info("Skipping visualization generation (disabled in config)")
            return True
        
        viz_config = self.config['visualization']
        success = True
        
        # Generate overlay visualization
        if viz_config.get('generate_overlay', True):
            success &= self._generate_visualization('overlay')
        
        # Generate heatmap visualization
        if viz_config.get('generate_heatmap', True):
            success &= self._generate_visualization('heatmap')
        
        # Generate boundary visualization
        if viz_config.get('generate_boundary', True):
            success &= self._generate_visualization('boundary')
        
        return success
    
    def _generate_visualization(self, viz_type: str) -> bool:
        """Generate a specific type of visualization."""
        command = [
            'python', 'visual.py',
            '--exp_name', self.config['experiment']['name'],
            '--csv_path', self.config['data']['csv_path'],
            '--wsi_dir', self.config['data']['wsi_dir'],
            '--results_dir', self.config['data']['results_dir'],
            '--xml_dir', self.config['data']['xml_dir'],
            '--patch_size', str(self.config['patches']['patch_size']),
            '--annotation_ratio', str(self.config['patches']['annotation_ratio']),
            '--percent', str(self.config['visualization']['percent'])
        ]
        
        if viz_type == 'heatmap':
            command.append('--heatmap')
        elif viz_type == 'boundary':
            command.append('--boundary')
        
        if self.config['visualization'].get('auto_skip', True):
            command.append('--auto_skip')
        
        return self._run_command(command, f"{viz_type.title()} Visualization")
    
    def run_analysis(self) -> bool:
        """Step 5: Calculate IoU metrics and generate analysis."""
        if not self.config['pipeline'].get('run_analysis', True):
            self.logger.info("Skipping analysis (disabled in config)")
            return True
        
        # Check if analysis already exists
        analysis_file = os.path.join(self.config['data']['results_dir'], 
                                   self.config['experiment']['name'], 
                                   'summary_iou_final.csv')
        if self._check_output_exists(analysis_file, "analysis"):
            return True
        
        command = [
            'python', 'analysis.py',
            '--results_dir', os.path.join(self.config['data']['results_dir'], 
                                        self.config['experiment']['name']),
            '--csv_dir', os.path.dirname(self.config['data']['csv_path'])
        ]
        
        return self._run_command(command, "Analysis")
    
    def _find_best_model(self) -> Optional[str]:
        """Find the best trained model based on accuracy."""
        models_dir = Path(self.config['data']['models_save_folder'])
        model_pattern = f"{self.config['experiment']['name']}_epoch*_loss*_acc*.pt"
        models = list(models_dir.glob(model_pattern))
        
        if not models:
            return None
        
        # Sort by accuracy (extract from filename)
        def extract_accuracy(model_path):
            try:
                parts = model_path.stem.split('_')
                acc_part = [p for p in parts if p.startswith('acc')][0]
                return float(acc_part.replace('acc', ''))
            except:
                return 0.0
        
        best_model = max(models, key=extract_accuracy)
        self.logger.info(f"Using best model: {best_model}")
        return str(best_model)
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline."""
        self.logger.info("Starting ROI Detection Pipeline")
        self.logger.info(f"Experiment: {self.config['experiment']['name']}")
        
        start_time = time.time()
        success = True
        
        # Run all steps
        steps = [
            ("Patch Extraction", self.extract_patches),
            ("Model Training", self.train_model),
            ("Patch Scoring", self.score_patches),
            ("Visualization Generation", self.generate_visualizations),
            ("Analysis", self.run_analysis)
        ]
        
        for step_name, step_func in steps:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Running: {step_name}")
            self.logger.info(f"{'='*50}")
            
            if not step_func():
                self.logger.error(f"Pipeline failed at step: {step_name}")
                success = False
                break
        
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            self.logger.info(f"\n{'='*50}")
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Total time: {duration:.2f} seconds")
            self.logger.info(f"{'='*50}")
        else:
            self.logger.error(f"\n{'='*50}")
            self.logger.error("Pipeline failed!")
            self.logger.error(f"Time elapsed: {duration:.2f} seconds")
            self.logger.error(f"{'='*50}")
        
        return success

def main():
    """Main function to run the automated pipeline."""
    parser = argparse.ArgumentParser(description='Automated ROI Detection Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate configuration, do not run pipeline')
    
    args = parser.parse_args()
    
    try:
        pipeline = ROIDetectionPipeline(args.config)
        
        if args.validate_only:
            print("Configuration validation successful!")
            return 0
        
        success = pipeline.run_pipeline()
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
