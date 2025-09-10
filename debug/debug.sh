#!/bin/bash
set -e

echo "--- Starting LOCAL Training Sanity Check ---"

export PROJECT_DIR=$(pwd)
export PYTHONPATH="${PROJECT_DIR}/src"
echo "Project directory: ${PROJECT_DIR}"
echo "PYTHONPATH set to: ${PYTHONPATH}"

echo "Activating Python virtual environment..."
source "${PROJECT_DIR}/.venv/bin/activate"
echo "Activation complete."

echo "Starting local training with accelerate..."
accelerate launch -m finetuner.train

echo "--- LOCAL Training Sanity Check Finished Successfully! ---"
