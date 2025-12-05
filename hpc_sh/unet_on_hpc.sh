#!/bin/bash

# ------------------------------
# BSUB directives (LSF scheduler)
# ------------------------------

# Queue with GPU nodes (change if needed)
#BSUB -q c02516

# Request 1 GPU in exclusive mode (if supported by your cluster)
#BSUB -gpu "num=1:mode=exclusive_process"

# Job name
#BSUB -J UnetViTTrain

# Number of CPU cores
#BSUB -n 4

# Keep job on a single node
#BSUB -R "span[hosts=1]"

# Request sufficient RAM (increase if needed)
#BSUB -R "rusage[mem=20GB]"

# Wall time (hh:mm)
#BSUB -W 08:00

# Absolute paths for log files
#BSUB -o /zhome/0c/9/212141/DL/Deep-learning-for-segmentation-of-micro-tomography-images/logs/unetvit_train_%J.out
#BSUB -e /zhome/0c/9/212141/DL/Deep-learning-for-segmentation-of-micro-tomography-images/logs/unetvit_train_%J.err


# ------------------------------
# Script starts
# ------------------------------

set -e

# ------------------------------
# Configuration
# ------------------------------
IMG_DATA_PATH="/zhome/0c/9/212141/DL/Deep-learning-for-segmentation-of-micro-tomography-images/datas/Original Images"
MASK_DATA_PATH="/zhome/0c/9/212141/DL/Deep-learning-for-segmentation-of-micro-tomography-images/datas/Original Masks"
SAVE_DIR="/zhome/0c/9/212141/DL/Deep-learning-for-segmentation-of-micro-tomography-images/models/predicted_models"
LOG_DIR="/zhome/0c/9/212141/DL/Deep-learning-for-segmentation-of-micro-tomography-images/logs"

EPOCHS=100
LR=1e-4
BATCH_SIZE=1  # can reduce to 2 or 1 if OOM persists

# Set PyTorch environment variable to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
export PYTHONPATH="/zhome/0c/9/212141/DL/Deep-learning-for-segmentation-of-micro-tomography-images/src:$PYTHONPATH"

echo "Starting training script..."
echo "IMG_DATA_PATH: $IMG_DATA_PATH"
echo "MASK_DATA_PATH: $MASK_DATA_PATH"
echo "SAVE_DIR: $SAVE_DIR"
echo "EPOCHS: $EPOCHS"
echo "LR: $LR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LOG_DIR: $LOG_DIR"

# ------------------------------
# Activate Python virtual environment
# ------------------------------
if [ -f ~/DL/Deep-learning-for-segmentation-of-micro-tomography-images/venv/bin/activate ]; then
    echo "Activating virtual environment..."
    source ~/DL/Deep-learning-for-segmentation-of-micro-tomography-images/venv/bin/activate
else
    echo "ERROR: Virtual environment not found!"
    exit 1
fi

# ------------------------------
# Prepare save directory
# ------------------------------
mkdir -p "$SAVE_DIR"
echo "Save directory created/checked: $SAVE_DIR"

mkdir -p "$LOG_DIR"
echo "Log directory created/checked: $LOG_DIR"


# ------------------------------
# Clean any leftover GPU memory from previous jobs
# ------------------------------
echo "Clearing CUDA cache..."
python3 - <<EOF
import torch
torch.cuda.empty_cache()
EOF

# ------------------------------
# Run training
# ------------------------------
echo "Executing U-Net training script..."
python3 src/scripts/train/train_unet.py \
    --img_data_path "$IMG_DATA_PATH" \
    --mask_data_path "$MASK_DATA_PATH" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --save_dir "$SAVE_DIR"

echo "Training script finished successfully."

