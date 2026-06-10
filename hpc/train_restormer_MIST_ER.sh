#!/bin/bash
#SBATCH --job-name=restormer_MIST_ER
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=23:00:00
#SBATCH -A ap_invilab
#SBATCH -p ampere_gpu
#SBATCH --gpus-per-node=1
#SBATCH -o /data/antwerpen/212/vsc21216/projects/logs/restormer_MIST_ER_%j.out
#SBATCH -e /data/antwerpen/212/vsc21216/projects/logs/restormer_MIST_ER_%j.err
set -euo pipefail
CONTAINER="$VSC_SCRATCH/containers/basicsr_nvidia.sif"
CODE_DIR="$VSC_DATA/projects/code"
DATA_SQSH="/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_ER_neutral.sqsh"
OUTPUT_DIR="$VSC_DATA/projects/outputs/restormer_MIST_ER_vaughan"
mkdir -p "$OUTPUT_DIR"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total            --format=csv -l 5 > "$OUTPUT_DIR/gpu_usage.csv" &
GPU_LOG_PID=$!
srun apptainer exec     --nv     -B "$CODE_DIR":/code     -B "$DATA_SQSH":/data:image-src=/     -B "$OUTPUT_DIR":/output     "$CONTAINER"     bash -c "
    mkdir -p /tmp/mist
    ln -s /data/train/HE  /tmp/mist/train_HE
    ln -s /data/train/IHC /tmp/mist/train_IHC
    ln -s /data/val/HE    /tmp/mist/val_HE
    ln -s /data/val/IHC   /tmp/mist/val_IHC
    bash /code/Restormer/train_restormer.sh     /code/Restormer/experiments/MIST_ER_Restormer/train_MIST_ER.yml
    "
kill $GPU_LOG_PID || true
