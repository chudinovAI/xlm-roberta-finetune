#!/bin/bash
#================================================================================
# Slurm SBATCH Directives
#================================================================================
set -e
#SBATCH --job-name=xlm-roberta-finetune
#SBATCH --output=job_logs/finetune_%j.out
#SBATCH --error=job_logs/finetune_%j.err
#SBATCH --time=08:00:00                 ??????????
#SBATCH --nodes=2                       ??????????
#SBATCH --ntasks-per-node=1             ??????????
#SBATCH --gpus-per-task=4               ??????????
#SBATCH --cpus-per-task=16              ??????????
#SBATCH --partition=gpu                 ??????????

#================================================================================

echo "================================================================================"
echo "Job starting on $(hostname) at $(date)"
echo "Slurm Job ID: ${SLURM_JOB_ID}"
echo "Requesting ${SLURM_NNODES} nodes, ${SLURM_GPUS_PER_TASK} GPUs per task."
echo "================================================================================"

export PROJECT_DIR=$(pwd)
echo "Project directory set to: ${PROJECT_DIR}"

echo "Activating Python virtual environment..."
source "${PROJECT_DIR}"/.venv/bin/activate
echo "Virtual environment activated successfully."

echo "Setting up PYTHONPATH..."
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"
echo "PYTHONPATH set to: ${PYTHONPATH}"

echo "Configuring environment variables..."
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export ACCELERATE_MAIN_PROCESS_PORT=29501
echo "Environment variables configured."

echo "Changing directory to ${PROJECT_DIR}"
cd "${PROJECT_DIR}"

echo "Starting distributed training with accelerate..."

accelerate launch -m finetuner.train

status=$?
if [ $status -ne 0 ]; then
    echo "================================================================================"
    echo "ERROR: Training failed with exit code $status."
    echo "Check the logs above for detailed error information."
    echo "================================================================================"
    exit $status
fi

echo "================================================================================"
echo "SUCCESS: Training completed successfully!"
echo "Job finished successfully at $(date)."
echo "================================================================================"
