echo "================================================================================"
echo "Job starting on $(hostname) at $(date)"
echo "Slurm Job ID: ${SLURM_JOB_ID}"
echo "Requesting ${SLURM_NNODES} nodes, ${SLURM_GPUS_PER_TASK} GPUs per task."
echo "================================================================================"

export PROJECT_DIR=$(pwd)
echo "Project directory set to: ${PROJECT_DIR}"

echo "Activating Python virtual environment..."
source ${PROJECT_DIR}/.venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export ACCELERATE_MAIN_PROCESS_PORT=29501

echo "Changing directory to ${PROJECT_DIR}"
cd ${PROJECT_DIR}

echo "Starting distributed training with accelerate..."

accelerate launch train.py

status=$?
if [ $status -ne 0 ]; then
    echo "Training failed with exit code $status."
    exit $status
fi

echo "Job finished successfully at $(date)."
#================================================================================
