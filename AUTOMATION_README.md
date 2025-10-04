# Automated ROI Detection Pipeline

This document describes the automated pipeline for ROI detection in melanocytic skin tumor whole slide images. The automation system eliminates the need for manual command-line execution and provides a streamlined workflow.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
python setup.py

# Or manually install requirements
pip install -r requirements.txt
```

### 2. Configure Pipeline
Edit `config.yaml` with your data paths:
```yaml
data:
  data_dir: "/path/to/your/h5/files"
  csv_path: "/path/to/your/metadata.csv"
  wsi_dir: "/path/to/your/svs/files"
  # ... other paths
```

### 3. Run Automated Pipeline
```bash
# Run complete pipeline
python automated_pipeline.py --config config.yaml

# Validate configuration only
python automated_pipeline.py --config config.yaml --validate-only
```

## 📁 File Structure

```
ROI_Detection/
├── automated_pipeline.py      # Main automation script
├── config.yaml               # Configuration file
├── setup.py                  # Setup and validation script
├── requirements.txt          # Python dependencies
├── AUTOMATION_README.md      # This file
│
├── extract_patches_3class.py # Original patch extraction
├── method_pcla_3class.py     # Original model training
├── score_pcla_3class.py      # Original patch scoring
├── visual.py                 # Original visualization
├── analysis.py               # Original analysis
│
└── utils/                    # Utility functions
    ├── core_utils.py
    ├── eval_utils.py
    ├── file_utils.py
    └── utils.py
```

## ⚙️ Configuration

The `config.yaml` file contains all pipeline parameters:

### Data Paths
```yaml
data:
  data_dir: "PATH_TO_SAVE_MEL/PATCHES"      # .h5 files directory
  csv_path: "PATH_TO_CSV"                   # Metadata CSV file
  wsi_dir: "PATH_TO_WSI"                    # .svs files directory
  xml_annotation_new: "PATH_TO_ANNOTATIONS" # XML annotations
  xml_annotation_other: "PATH_TO_OTHER"     # Other annotations
  xml_dir: "PATH_TO_GROUND_TRUTH"           # Ground truth labels
  feat_dir: "PATH_TO_SAVE_FEATURES"         # Extracted patches
  results_dir: "PATH_TO_SAVE_RESULTS"       # All results
  models_save_folder: "./saved_models/"     # Trained models
  classification_save_dir: "PATH_TO_CLASS"  # Classification results
```

### Model Configuration
```yaml
model:
  architecture: "vgg16"        # Model architecture
  num_classes: 3               # Number of classes
  batch_size: 100              # Batch size
  n_epochs: 20                 # Number of epochs
  learning_rate: 5e-4          # Learning rate
  weight_decay: 1e-4           # Weight decay
  patience: 20                 # Early stopping patience
```

### Pipeline Control
```yaml
pipeline:
  extract_patches: true        # Run patch extraction
  train_model: true           # Run model training
  score_patches: true         # Run patch scoring
  generate_visualizations: true # Generate visualizations
  run_analysis: true          # Run analysis
  skip_existing: true         # Skip if outputs exist
```

## 🔄 Pipeline Workflow

The automated pipeline runs the following steps:

1. **Patch Extraction** (`extract_patches_3class.py`)
   - Extracts patches from whole slide images
   - Uses XML annotations for region detection
   - Creates train/val/test splits

2. **Model Training** (`method_pcla_3class.py`)
   - Trains PCLA-3C patch classification model
   - Uses VGG16 architecture
   - Implements early stopping

3. **Patch Scoring** (`score_pcla_3class.py`)
   - Computes prediction scores for all patches
   - Generates slide-level classifications
   - Saves results in HDF5 format

4. **Visualization Generation** (`visual.py`)
   - Creates heatmap visualizations
   - Generates boundary detection
   - Produces overlay images

5. **Analysis** (`analysis.py`)
   - Calculates IoU metrics
   - Generates summary statistics
   - Creates final reports

## 🎛️ Advanced Usage

### Running Specific Steps
You can disable specific steps in the configuration:

```yaml
pipeline:
  extract_patches: false      # Skip patch extraction
  train_model: true          # Only train model
  score_patches: false       # Skip scoring
  generate_visualizations: false # Skip visualization
  run_analysis: false        # Skip analysis
```

### Custom Visualization Types
```yaml
visualization:
  generate_heatmap: true     # Generate heatmaps
  generate_boundary: true    # Generate boundaries
  generate_overlay: true     # Generate overlays
  percent: 0.2              # Top percentage for visualization
```

### Hardware Configuration
```yaml
hardware:
  use_gpu: true             # Use GPU if available
  num_workers: 4            # Data loading workers
  pin_memory: true          # Pin memory for faster loading
```

## 📊 Output Structure

The pipeline creates the following output structure:

```
results/
├── pcla_3class/
│   ├── score/              # Patch scores (.h5 files)
│   ├── overlay/            # Visualization images
│   ├── pcla_3class.txt     # Processing logs
│   └── summary_iou_final.csv # Final analysis
│
saved_models/
├── pcla_3class_epoch*_loss*_acc*.pt # Trained models
│
classification_results/
└── classification_pcla_3class.csv   # Classification results
```

## 🔧 Troubleshooting

### Common Issues

1. **Configuration Errors**
   ```bash
   # Validate configuration
   python automated_pipeline.py --config config.yaml --validate-only
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt
   ```

3. **CUDA Issues**
   - Check CUDA installation: `nvidia-smi`
   - Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

4. **Memory Issues**
   - Reduce batch size in config
   - Use CPU instead of GPU
   - Process smaller datasets

### Logging

The pipeline creates detailed logs:
- Console output for real-time monitoring
- Log file: `pipeline.log` (configurable)
- Step-specific error messages

## 🚀 Performance Tips

1. **Use GPU**: Ensure CUDA is properly configured
2. **Batch Size**: Adjust based on available memory
3. **Skip Existing**: Enable `skip_existing` to resume interrupted runs
4. **Parallel Processing**: Increase `num_workers` for faster data loading

## 📈 Monitoring Progress

The pipeline provides real-time progress updates:
- Step-by-step execution status
- Time estimates and completion times
- Error handling with detailed messages
- Automatic resumption from failures

## 🔄 Resuming Interrupted Runs

The pipeline automatically handles interruptions:
- Skips completed steps when `skip_existing: true`
- Resumes from the last successful step
- Preserves intermediate results

## 📝 Example Commands

```bash
# Full pipeline run
python automated_pipeline.py --config config.yaml

# Validate configuration
python automated_pipeline.py --config config.yaml --validate-only

# Setup environment
python setup.py

# Check specific step (modify config.yaml)
python automated_pipeline.py --config config.yaml
```

## 🆘 Support

For issues and questions:
1. Check the logs for detailed error messages
2. Validate your configuration file
3. Ensure all dependencies are installed
4. Verify data paths and file permissions

The automated pipeline significantly reduces the complexity of running the ROI detection workflow while maintaining all the functionality of the original scripts.
