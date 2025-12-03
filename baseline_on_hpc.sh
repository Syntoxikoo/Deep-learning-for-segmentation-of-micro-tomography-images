#!/bin/bash
#BSUB -J TOMO_trans
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err
#BSUB -N

echo "=================================="
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=================================="

rm -rf .venv
rm -rf .uv_cache
# Load module
module purge
module load python3/3.11.9
module load cuda/12.1
module load ffmpeg

mkdir -p logs


# Set environment variables
SYSTEM_PYTHON=$(which python3)
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_NUM_THREADS=$LSB_DJOB_NUMPROC


echo "Syncing environment..."
uv venv .venv --python "$SYSTEM_PYTHON"
uv sync --python .venv/bin/python

source .venv/bin/activate
echo ""
echo "=========== training ================="

python -m src.scripts.train.train_with_baseline_trans \
    --on_cluster True \
    --epochs 100 \
    --learning_rate 1e-4 \
    --batch_size 6 \
    --loss both

EXIT_CODE=$?



echo "Job finished with exit code: $EXIT_CODE"
exit $EXIT_CODE
