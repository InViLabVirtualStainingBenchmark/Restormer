--- 
# Restormer: HPC Virtual Staining Benchmark
H&E → IHC Translation · BCI & MIST Datasets · CalcUA HPC (Vaughan A100)

> This documents HPC training, inference, and evaluation for Restormer as part of the InViLab Virtual Staining Benchmark. For local setup and initial experiments, see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## Table of Contents
- [Overview](#overview)
- [Environment](#environment)
- [Cluster Structure](#cluster-structure)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Notes](#notes)

---

## Overview

Restormer uses a hierarchical transformer architecture with multi-DConv head transposed attention (MDTA) and gated-dconv feed-forward networks (GDFN), applied here to paired H&E → IHC virtual staining with L1 loss.

**Datasets:**

| Dataset | Task | Train | Val | Test |
|---------|------|-------|-----|------|
| BCI | H&E → IHC | 3896 | 488 | 489 |
| MIST ER | H&E → ER IHC | 4153 | 500 | 500 |
| MIST HER2 | H&E → HER2 IHC | 4642 | 500 | 500 |
| MIST Ki67 | H&E → Ki67 IHC | 4361 | 500 | 500 |
| MIST PR | H&E → PR IHC | 4139 | 500 | 500 |

**Key training settings:**

| Parameter | Value |
|-----------|-------|
| Architecture | Restormer (dim=48, blocks=[4,6,6,8]) |
| Input crop size | 256 × 256 |
| Batch size | 1 (BCI) / 2 (MIST) |
| Total iterations | 100,000 (BCI) / 50,000 (MIST) |
| Loss function | L1Loss |
| Optimizer | Adam (lr=2e-4) |
| LR scheduler | CosineAnnealingRestartCyclicLR |

---

## Environment

Training runs inside an Apptainer container on the **CalcUA Vaughan cluster** (NVIDIA A100 40GB, `ampere_gpu` partition).

**Container:** `basicsr_nvidia.sif`
- Base image: `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`
- PyTorch: 2.1.2+cu121
- BasicSR: 1.4.2 (modified — see [DOCUMENTATION.md](DOCUMENTATION.md))
- Python: 3.9

**Container location:**
```
$VSC_SCRATCH/containers/basicsr_nvidia.sif
```

> ⚠️ The container uses a modified BasicSR 1.4.2 with Restormer-specific patches applied. Do not replace with a standard BasicSR container.

---

## Cluster Structure

**Compute nodes used:**

| Partition | Node | GPU | Used for |
|-----------|------|-----|----------|
| ampere_gpu | nvam1.vaughan | 4× A100 40GB | BCI + MIST training |

**Key paths:**
```
$VSC_DATA/projects/code/Restormer/          ← repository
$VSC_DATA/projects/jobs/                    ← SLURM job scripts
$VSC_DATA/projects/logs/                    ← job logs
$VSC_DATA/projects/outputs/                 ← training checkpoints
$VSC_SCRATCH/containers/                    ← Apptainer containers
/scratch/antwerpen/grp/ap_invilab_td_thesis/  ← shared group storage
```

---

## Dataset Preparation

All datasets are stored as **SquashFS images** (`.sqsh`) for fast HPC I/O using a neutral folder structure:

```
dataset.sqsh (mounted at /data)
├── train/
│   ├── HE/        ← H&E input images
│   └── IHC/       ← IHC ground truth images
├── val/
│   ├── HE/
│   └── IHC/
└── test/
    ├── HE/
    └── IHC/
```

**Squashfs locations (shared group storage):**
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/BCI.sqsh
/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_ER_neutral.sqsh
/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_HER2_neutral.sqsh
/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_Ki67_neutral.sqsh
/scratch/antwerpen/grp/ap_invilab_td_thesis/MIST_PR_neutral.sqsh
```

**Runtime symlinks:**

The training configs expect `train_HE/`, `train_IHC/` etc. Job scripts create symlinks at runtime inside the container:

```bash
# BCI
mkdir -p /tmp/bci
ln -s /data/train/HE  /tmp/bci/train_HE
ln -s /data/train/IHC /tmp/bci/train_IHC
ln -s /data/val/HE    /tmp/bci/val_HE
ln -s /data/val/IHC   /tmp/bci/val_IHC

# MIST
mkdir -p /tmp/mist
ln -s /data/train/HE  /tmp/mist/train_HE
ln -s /data/train/IHC /tmp/mist/train_IHC
ln -s /data/val/HE    /tmp/mist/val_HE
ln -s /data/val/IHC   /tmp/mist/val_IHC
```

---

## Training

### How Training Works

Unlike NAFNet and Uformer, Restormer training fits within a single 23-hour SLURM job:
- **BCI**: 100k iterations at ~0.71s/iter ≈ ~20h
- **MIST**: 50k iterations at ~0.71s/iter ≈ ~10h

Training uses `--auto_resume` so if a job is interrupted it automatically resumes from the latest `.state` file.

### Job Scripts

```
$VSC_DATA/projects/jobs/
├── train_restormer_BCI_vaughan.sh      ← BCI training (ampere_gpu)
├── train_restormer_MIST_ER.sh
├── train_restormer_MIST_HER2.sh
├── train_restormer_MIST_Ki67.sh
└── train_restormer_MIST_PR.sh
```

### Submitting Training

**BCI:**
```bash
sbatch $VSC_DATA/projects/jobs/train_restormer_BCI_vaughan.sh
```

**MIST (all 4 biomarkers):**
```bash
for marker in ER HER2 Ki67 PR; do
    sbatch $VSC_DATA/projects/jobs/train_restormer_MIST_${marker}.sh
done
```

### Output Structure

```
$VSC_DATA/projects/outputs/restormer_BCI_vaughan/
├── restormer_BCI/
│   ├── models/
│   │   ├── net_g_latest.pth     ← latest checkpoint
│   │   └── net_g_90000.pth      ← checkpoint at 90k iters
│   └── training_states/         ← optimizer states for resuming
├── gpu_usage.csv
└── train_log.txt

$VSC_DATA/projects/outputs/restormer_MIST_ER_vaughan/
├── gpu_usage.csv
└── train_log.txt
```

> MIST checkpoints are saved inside the container output path. Verify locations after training completes.

### Monitoring

```bash
# Check running jobs
squeue -u vsc21216 --format="%.18i %.35j %.8T %.10M %R"

# Watch training progress
tail -f $VSC_DATA/projects/logs/restormer_MIST_ER_<JOBID>.out

# Check quota
myquota
```

---

## Inference

Benchmark inference uses the unified `benchmark_inference.py` script:

```bash
sbatch $VSC_DATA/projects/jobs/run_benchmark_BCI_restormer.sh
```

Results are saved to:
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/benchmark_inference/restormer_BCI/
├── comparison/      ← side-by-side PNGs (HE | predicted | GT)
├── predicted/       ← predicted IHC only
├── metrics.csv      ← per-image PSNR and SSIM
└── summary.txt      ← average PSNR and SSIM
```

---

## Evaluation

Evaluation uses the shared `evaluate.py` script from the InViLab benchmark repository, run inside the `evaluate_nvidia.sif` container on the `broadwell` (CPU) partition of Leibniz.

**Metrics computed:** PSNR, SSIM, MS-SSIM, LPIPS (AlexNet + VGG), MAE, FID

Results are appended to:
```
/scratch/antwerpen/grp/ap_invilab_td_thesis/benchmark_results.csv
```

---

## Results

### BCI Dataset

| Model | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | LPIPS-Alex ↓ | LPIPS-VGG ↓ | MAE ↓ | FID ↓ |
|-------|--------|--------|-----------|--------------|-------------|-------|-------|
| Restormer (256 crop, 100k iters) | **19.43 dB** | **0.4750** | — | — | — | — | — |

*MS-SSIM, LPIPS, MAE, FID will be updated after evaluation completes.*

### MIST Dataset

| Model | Marker | PSNR ↑ | SSIM ↑ | LPIPS-Alex ↓ | FID ↓ |
|-------|--------|--------|--------|--------------|-------|
| Restormer | ER | — | — | — | — |
| Restormer | HER2 | — | — | — | — |
| Restormer | Ki67 | — | — | — | — |
| Restormer | PR | — | — | — | — |

*MIST training in progress. Results will be updated after inference and evaluation complete.*

---

## Notes

- **Single-job training** — unlike NAFNet and Uformer, Restormer training fits within one 23-hour job. No chaining needed.
- **`--auto_resume`** is passed via `train_restormer.sh` — training automatically resumes from the latest state file if interrupted.
- **BCI checkpoint** — `net_g_latest.pth` is used for inference since Restormer saves by iteration, not by best validation PSNR. Use `net_g_90000.pth` or `net_g_latest.pth`.
- **Always use neutral squashfs** (`BCI.sqsh`, `MIST_*_neutral.sqsh`). Old format squashfs files have been deleted.
- **MIST uses 50k iters** vs 100k for BCI — chosen to balance training time with the smaller MIST dataset size relative to BCI.