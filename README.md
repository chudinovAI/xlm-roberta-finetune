# XLM-RoBERTa Fine-tuning for Text Classification

A comprehensive project for fine-tuning XLM-RoBERTa large model using LoRA (Low-Rank Adaptation) for text classification tasks. This project is designed to work with the AINL-Eval-2025 competition dataset and provides both training and inference capabilities.

## Project Description

This project implements a fine-tuning pipeline for the XLM-RoBERTa large model using Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA adapters. The model is trained on Russian text classification data and can be used to generate predictions for test datasets.

### Key Features

- **LoRA Fine-tuning**: Efficient fine-tuning using Low-Rank Adaptation
- **Distributed Training**: Support for multi-GPU training with Accelerate
- **Slurm Integration**: Ready-to-use scripts for cluster environments
- **Comprehensive Pipeline**: From data loading to inference and submission generation
- **Russian Text Support**: Optimized for Russian language text classification

## Project Structure

```
xlm-roberta-finetune/
├── data/           # Competition dataset
│   ├── train.csv         # Training data
│   ├── dev.csv           # Development data
│   ├── dev_full.csv      # Full development data
│   └── test.csv          # Test data for inference
|   debug/          # Debug files
│   ├── create_debug_dataset.py        # Debug dataset
│   ├── debug.sh          # Debug shell script
├── src/                      # Source code
│   ├── config.py            # Configuration settings
│   ├── data_utils.py        # Data loading and preprocessing
│   ├── model_utils.py       # Model creation and LoRA setup
│   ├── train.py             # Training script
│   └── inference.py         # Inference and submission generation
├── outputs/                  # Output directory (created during training)
│   ├── saved_models/
│   │   ├── checkpoints/     # Training checkpoints
│   │   └── best_model_lora/ # Best model for inference
│   └── submission.csv       # Generated predictions
├── run.sh                   # Slurm training script
├── pyproject.toml          # Project dependencies
├── uv.lock                 # Locked dependencies
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.11-3.13
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-compatible GPU (recommended for training)

### Setup Environment

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/chudinovAI/xlm-roberta-finetune.git
   cd xlm-roberta-finetune
   ```

2. **Create virtual environment and install dependencies**:
   ```bash
   # Create virtual environment with uv
   uv venv
   
   # Activate the environment
   source .venv/bin/activate  # On Linux/macOS
   # or
   .venv\Scripts\activate     # On Windows
   
   # Install dependencies
   uv pip install -e .
   # or
   uv sync
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
   ```

### Dependencies

The project uses the following key dependencies:
- `torch>=2.2.0` - PyTorch for deep learning
- `transformers>=4.40.0` - Hugging Face transformers
- `peft>=0.17.1` - Parameter-Efficient Fine-Tuning
- `datasets>=2.21.0` - Dataset handling
- `pandas>=2.3.2` - Data manipulation
- `scikit-learn>=1.7.1` - Evaluation metrics

## Training

### Configuration

Before training, review the configuration in `src/config.py`:

```python
MODEL_NAME = "FacebookAI/xlm-roberta-large"
TRAIN_DATA_PATH = "data/train.csv"
DEV_DATA_PATH = "data/dev_full.csv"
TEST_DATA_PATH = "data/test.csv"

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Training Configuration
LEARNING_RATE = 2e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 3
```

### Running on Slurm Cluster

The project includes a Slurm-ready script (`run.sh`) for cluster environments:

1. **Submit the job**:
   ```bash
   sbatch run.sh
   ```

2. **Monitor the job**:
   ```bash
   squeue -u $USER
   ```

3. **Check logs**:
   ```bash
   # Job logs are saved to job_logs/ directory
   tail -f job_logs/finetune_<job_id>.out
   tail -f job_logs/finetune_<job_id>.err
   ```

### Local Training (Alternative)

If you want to run training locally without Slurm:

```bash
# Activate environment
source .venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Run training
python src/train.py
```

### Training Process

The training script (`src/train.py`) performs the following steps:

1. **Data Loading**: Loads train and dev datasets using `load_datasets()`
2. **Tokenization**: Tokenizes text data with XLM-RoBERTa tokenizer
3. **Model Setup**: Creates base model and applies LoRA adapters
4. **Training**: Runs distributed training with evaluation
5. **Model Saving**: Saves the best model to `outputs/saved_models/best_model_lora/`

### Expected Outputs

After successful training, you should see:
- Model checkpoints in `outputs/saved_models/checkpoints/`
- Best model in `outputs/saved_models/best_model_lora/`
- Training logs with accuracy and F1 scores

## Inference

### Generating Predictions

After training is complete, use the inference script to generate predictions:

```bash
# Activate environment
source .venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Run inference
python src/inference.py
```

### Inference Process

The inference script (`src/inference.py`) performs the following steps:

1. **Model Loading**: Loads the best trained model from `BEST_MODEL_DIR`
2. **Data Loading**: Loads test data from `TEST_DATA_PATH`
3. **Prediction**: Generates predictions for all test samples with progress bar
4. **Submission Creation**: Creates submission DataFrame with 'id' and 'label' columns
5. **File Saving**: Saves predictions to `SUBMISSION_PATH`

### Output

The inference script generates:
- `outputs/submission.csv` - Final predictions file ready for submission
- Console output showing sample predictions and progress

### Submission Format

The generated `submission.csv` file contains:
```csv
id,label
0,predicted_label_1
1,predicted_label_2
...
```

## Configuration Options

### Model Configuration
- `MODEL_NAME`: Base model identifier (default: "FacebookAI/xlm-roberta-large")
- `MAX_LENGTH`: Maximum sequence length for tokenization (default: 512)

### LoRA Configuration
- `LORA_R`: LoRA rank (default: 16)
- `LORA_ALPHA`: LoRA alpha scaling factor (default: 32)
- `LORA_DROPOUT`: LoRA dropout rate (default: 0.1)

### Training Configuration
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `PER_DEVICE_TRAIN_BATCH_SIZE`: Training batch size per device (default: 8)
- `PER_DEVICE_EVAL_BATCH_SIZE`: Evaluation batch size per device (default: 8)
- `NUM_TRAIN_EPOCHS`: Number of training epochs (default: 3)
- `WEIGHT_DECAY`: Weight decay for regularization (default: 0.01)


### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `PER_DEVICE_TRAIN_BATCH_SIZE` and `PER_DEVICE_EVAL_BATCH_SIZE`
   - Reduce `MAX_LENGTH` if possible

2. **Model Loading Errors**:
   - Ensure the training completed successfully
   - Check that `BEST_MODEL_DIR` contains the saved model files

3. **Data Loading Issues**:
   - Verify that data files exist in the correct paths
   - Check that CSV files have the required columns ('text', 'label' for train/dev, 'id', 'text' for test)
