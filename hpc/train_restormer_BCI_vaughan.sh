#!/bin/bash
#SBATCH --job-name=restormer_BCI_vau
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=23:00:00
#SBATCH -A ap_invilab
#SBATCH -p ampere_gpu
#SBATCH --gpus-per-node=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/restormer_BCI_vau_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/restormer_BCI_vau_%j.err
set -euo pipefail
CONTAINER="$VSC_SCRATCH/containers/basicsr_nvidia.sif"
CODE_DIR="$VSC_DATA/projects/code"
DATA_SQSH="/scratch/antwerpen/grp/ap_invilab_td_thesis/BCI.sqsh"
OUTPUT_DIR="$VSC_DATA/projects/outputs/restormer_BCI_vaughan"
mkdir -p "$OUTPUT_DIR"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 > "$OUTPUT_DIR/gpu_usage.csv" &
GPU_LOG_PID=$!
srun apptainer exec \
    --nv \
    -B "$CODE_DIR":/code \
    -B "$DATA_SQSH":/data:image-src=/ \
    -B "$OUTPUT_DIR":/output \
    "$CONTAINER" \
    bash -c "
    mkdir -p /tmp/bci
    ln -s /data/train/HE  /tmp/bci/train_HE
    ln -s /data/train/IHC /tmp/bci/train_IHC
    ln -s /data/val/HE    /tmp/bci/val_HE
    ln -s /data/val/IHC   /tmp/bci/val_IHC
    bash /code/Restormer/train_restormer.sh
    "
kill $GPU_LOG_PID || true
