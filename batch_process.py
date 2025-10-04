#!/usr/bin/env python3
"""
Batch Processing Script for ROI Detection Pipeline
This script allows processing multiple datasets or experiments in batch.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from automated_pipeline import ROIDetectionPipeline

class BatchProcessor:
    """Batch processor for multiple experiments."""
    
    def __init__(self, config_dir: str):
        """Initialize batch processor."""
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
        
    def find_config_files(self) -> list:
        """Find all configuration files in the directory."""
        config_files = list(self.config_dir.glob("config_*.yaml"))
        config_files.extend(list(self.config_dir.glob("config_*.yml")))
        return config_files
    
    def process_single_config(self, config_path: str) -> bool:
        """Process a single configuration file."""
        self.logger.info(f"Processing configuration: {config_path}")
        
        try:
            pipeline = ROIDetectionPipeline(str(config_path))
            success = pipeline.run_pipeline()
            
            if success:
                self.logger.info(f"✓ Successfully completed: {config_path}")
            else:
                self.logger.error(f"✗ Failed: {config_path}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"✗ Error processing {config_path}: {e}")
            return False
    
    def process_batch(self, config_files: list = None) -> dict:
        """Process multiple configuration files."""
        if config_files is None:
            config_files = self.find_config_files()
        
        if not config_files:
            self.logger.warning("No configuration files found")
            return {}
        
        results = {}
        
        for config_file in config_files:
            config_name = config_file.stem
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing: {config_name}")
            self.logger.info(f"{'='*60}")
            
            success = self.process_single_config(str(config_file))
            results[config_name] = success
        
        return results
    
    def generate_summary(self, results: dict):
        """Generate summary of batch processing results."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("BATCH PROCESSING SUMMARY")
        self.logger.info(f"{'='*60}")
        
        total = len(results)
        successful = sum(results.values())
        failed = total - successful
        
        self.logger.info(f"Total configurations: {total}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Success rate: {successful/total*100:.1f}%")
        
        self.logger.info("\nDetailed Results:")
        for config_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            self.logger.info(f"  {config_name}: {status}")

def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description='Batch Process ROI Detection Pipeline')
    parser.add_argument('--config-dir', type=str, default='.',
                       help='Directory containing configuration files')
    parser.add_argument('--config-files', nargs='+', type=str,
                       help='Specific configuration files to process')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('batch_process.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        processor = BatchProcessor(args.config_dir)
        
        if args.config_files:
            # Process specific files
            config_files = [Path(f) for f in args.config_files]
            results = processor.process_batch(config_files)
        else:
            # Process all config files in directory
            results = processor.process_batch()
        
        processor.generate_summary(results)
        
        # Return appropriate exit code
        failed_count = sum(1 for success in results.values() if not success)
        return 1 if failed_count > 0 else 0
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
