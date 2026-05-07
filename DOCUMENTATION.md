# Restormer — Virtual Staining Setup Guide
### Adapted for BasicSR 1.4.2 | BCI & MIST Datasets


---


## Overview




**Datasets supported:**
- BCI (Breast Cancer Immunohistochemistry) — H&E → IHC translation
- MIST — H&E → IHC translation across 4 biomarkers: HER2, ER, Ki67, PR


---


## Environment


```
Python:      3.9
BasicSR:     1.4.2
PyTorch:     2.1.2+cu121
TorchVision: 0.16.2+cu121
Conda env:   vs_ua
```


Activate the environment before running anything:
```bash
conda activate vs_ua
```


---


## Repository Structure


```
virtual_stain/repos/Restormer/
├── basicsr/                        ← Local Restormer-specific basicsr code (NOT used for training)
├── experiments/
│   ├── BCI_Restormer/
│   │   ├── train_BCI.yml           ← Training config for BCI
│   │   ├── test_BCI.yml            ← Test/inference config for BCI
│   │   ├── models/                 ← Saved checkpoints
│   │   ├── training_states/        ← Optimizer/scheduler states for resuming
│   │   ├── logs/                   ← Training logs
│   │   └── visualization/          ← Validation images (if save_img: true)
│   ├── MIST_HER2_Restormer/
│   │   ├── train_MIST_HER2.yml
│   │   └── test_MIST_HER2.yml
│   ├── MIST_ER_Restormer/
│   │   ├── train_MIST_ER.yml
│   │   └── test_MIST_ER.yml
│   ├── MIST_Ki67_Restormer/
│   │   ├── train_MIST_Ki67.yml
│   │   └── test_MIST_Ki67.yml
│   └── MIST_PR_Restormer/
│       ├── train_MIST_PR.yml
│       └── test_MIST_PR.yml
├── results/
│   └── restormer_BCI/
│       └── visualization/
│           └── BCI_Test/           ← Output inference images saved here
└── data -> ../../data/             ← Symlink or relative path to datasets
```


---


## Dataset Structure


### BCI Dataset
```
virtual_stain/data/BCI/
├── HE/
│   ├── train/      ← Input images (low quality)
│   └── test/
└── IHC/
   ├── train/      ← Ground truth images
   └── test/
```


### MIST Dataset
```
virtual_stain/data/MIST/
├── HER2/
│   └── TrainValAB/
│       ├── trainA/   ← HE input
│       ├── trainB/   ← IHC ground truth
│       ├── valA/
│       └── valB/
├── ER/
│   └── TrainValAB/
│       ├── trainA/  trainB/  valA/  valB/
├── Ki67/
│   └── TrainValAB/
│       ├── trainA/  trainB/  valA/  valB/
└── PR/
   └── TrainValAB/
       ├── trainA/  trainB/  valA/  valB/
```


---


## Modifications Made to BasicSR 1.4.2


All changes were made to the **installed** basicsr package at:
```
~/miniconda3/envs/vs_ua/lib/python3.9/site-packages/basicsr/
```


### 1. Added Restormer model and architecture files


Copied from local Restormer repo into installed basicsr:


```bash
# Create archs folder (does not exist in 1.4.2)
mkdir -p .../basicsr/models/archs/


# Copy architecture files
cp basicsr/models/archs/restormer_arch.py    .../basicsr/models/archs/
cp basicsr/models/archs/arch_util.py         .../basicsr/models/archs/
cp basicsr/models/archs/__init__.py          .../basicsr/models/archs/


# Copy model class
cp basicsr/models/image_restoration_model.py .../basicsr/models/


# Copy losses and scheduler
cp -r basicsr/models/losses/                 .../basicsr/models/
cp basicsr/models/lr_scheduler.py            .../basicsr/models/
```


### 2. Registered ImageCleanModel with MODEL_REGISTRY


Added decorator to `image_restoration_model.py`:
```python
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ImageCleanModel(BaseModel):
   ...
```


### 3. Added CosineAnnealingRestartCyclicLR scheduler support


In `base_model.py` → `setup_schedulers()`:
```python
elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
   from basicsr.models.lr_scheduler import CosineAnnealingRestartCyclicLR
   for optimizer in self.optimizers:
       self.schedulers.append(
           CosineAnnealingRestartCyclicLR(optimizer, **train_opt['scheduler']))
```


### 4. Fixed test mode crash (missing `train` key)


In `image_restoration_model.py`:
```python
# BEFORE (crashes during test — no 'train' section in test yml)
self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)


# AFTER (safe for both train and test mode)
self.mixing_flag = self.opt.get('train', {}).get('mixing_augs', {}).get('mixup', False)
```


### 5. Added `validation()` wrapper method


In `image_restoration_model.py`, added method to bridge 1.4.2 signature:
```python
def validation(self, dataloader, current_iter, tb_logger, save_img=False):
   self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr=True, use_image=True)
```


### 6. Fixed experiment folder archiving


In `train.py`, prevented basicsr from renaming the experiment folder when `--auto_resume` is used:
```python
# BEFORE
if resume_state is None:
   make_exp_dirs(opt)


# AFTER
if resume_state is None and not opt.get('auto_resume', False):
   make_exp_dirs(opt)
```


### 7. Fixed SameFileError when copying yml


In `options.py` → `copy_opt_file()`:
```python
# BEFORE
copyfile(opt_file, filename)


# AFTER
if osp.abspath(opt_file) != osp.abspath(filename):
   copyfile(opt_file, filename)
```


### 8. Fixed path override for experiments and results


In `options.py` → `parse_options()`, changed hard-coded path generation to respect yml-defined paths:


**For training:**
```python
experiments_root = opt['path'].get('experiments_root', osp.join(root_path, 'experiments', opt['name']))
opt['path']['models'] = opt['path'].get('models', osp.join(experiments_root, 'models'))
opt['path']['training_states'] = opt['path'].get('training_states', osp.join(experiments_root, 'training_states'))
opt['path']['log'] = opt['path'].get('log', experiments_root)
opt['path']['visualization'] = opt['path'].get('visualization', osp.join(experiments_root, 'visualization'))
```


**For testing:**
```python
results_root = opt['path'].get('results_root', osp.join(root_path, 'results', opt['name']))
opt['path']['results_root'] = results_root
opt['path']['log'] = opt['path'].get('log', results_root)
opt['path']['visualization'] = opt['path'].get('visualization', osp.join(results_root, 'visualization'))
```


---


## Training


### BCI Dataset


```bash
conda activate vs_ua
cd ~/virtual_stain/repos/Restormer


python -m basicsr.train \
 -opt /home/thomas/virtual_stain/repos/Restormer/experiments/BCI_Restormer/train_BCI.yml \
 --auto_resume
```


### MIST Subsets


```bash
# HER2
python -m basicsr.train \
 -opt /home/thomas/virtual_stain/repos/Restormer/experiments/MIST_HER2_Restormer/train_MIST_HER2.yml \
 --auto_resume


# ER
python -m basicsr.train \
 -opt /home/thomas/virtual_stain/repos/Restormer/experiments/MIST_ER_Restormer/train_MIST_ER.yml \
 --auto_resume


# Ki67
python -m basicsr.train \
 -opt /home/thomas/virtual_stain/repos/Restormer/experiments/MIST_Ki67_Restormer/train_MIST_Ki67.yml \
 --auto_resume


# PR
python -m basicsr.train \
 -opt /home/thomas/virtual_stain/repos/Restormer/experiments/MIST_PR_Restormer/train_MIST_PR.yml \
 --auto_resume
```


> **Important:** Always use `--auto_resume` to prevent basicsr from archiving your experiment folder on rerun. Always use the **absolute path** to the yml file, not a relative path.


### Key training parameters (in yml)


| Parameter | Value | Notes |
|---|---|---|
| `total_iter` | 300000 (cluster) / 1000 (local test) | Full training = 300k |
| `gt_size` / `lq_size` | 128 (cluster) / 64 (local) | Reduce if OOM |
| `batch_size_per_gpu` | 4 | Reduce if OOM |
| `save_checkpoint_freq` | 5000 | Saves every 5k iters |
| `val_freq` | 5000 | Validates every 5k iters |


---


## Resuming Training


If training is interrupted, resume from the last checkpoint:


```bash
python -m basicsr.train \
 -opt /home/thomas/virtual_stain/repos/Restormer/experiments/BCI_Restormer/train_BCI.yml \
 --auto_resume
```


The `--auto_resume` flag automatically finds the latest `.state` file in `training_states/` and resumes from there.


---


## Inference / Testing


### Step 1 — Create results folder (first time only)


```bash
mkdir -p ~/virtual_stain/repos/Restormer/results/restormer_BCI/visualization
```


### Step 2 — Run test


```bash
conda activate vs_ua
cd ~/virtual_stain/repos/Restormer


python -m basicsr.test \
 -opt /home/thomas/virtual_stain/repos/Restormer/experiments/BCI_Restormer/test_BCI.yml
```


### Output images location


```
virtual_stain/repos/Restormer/results/restormer_BCI/visualization/BCI_Test/
```


Each output image is named after its input filename. The test script also prints the final **PSNR** score on the full test set.


---


## Common Issues & Fixes


| Error | Cause | Fix |
|---|---|---|
| `KeyError: 'ImageCleanModel'` | Model not registered | Add `@MODEL_REGISTRY.register()` decorator |
| `KeyError: 'scheduler'` | 1.4.2 expects `scheduler` key | Add `scheduler:` block to yml |
| `KeyError: 'mixing_augs'` | Missing key in yml | Add `mixing_augs: {mixup: false}` under `train:` |
| `KeyError: 'use_hflip'` | 1.4.2 dataset requires explicit flip flags | Add `use_hflip: true` and `use_rot: true` to dataset |
| `KeyError: 'train'` during test | Model accesses train config in test mode | Use `.get('train', {})` pattern |
| `SameFileError` | basicsr copies yml to same location | Add same-file check in `copy_opt_file` |
| `CUDA out of memory` | Crop size too large | Reduce `gt_size` and `lq_size` to 64 |
| Folder getting archived | basicsr renames existing experiment folders | Always use `--auto_resume` flag |
| Checkpoints saving to conda env | `options.py` overrides yml paths | Fixed with `.get()` pattern in `parse_options` |


---


## Notes for Future Users


- **Always use absolute paths** in yml files and when passing `-opt` to avoid path resolution issues with basicsr 1.4.2.
- **Always use `--auto_resume`** when rerunning training to prevent the experiment folder from being archived.
- The `total_iter: 1000` in the yml files is for **local smoke testing only**. Change to `300000` for real training on the cluster.
- The `gt_size: 64` is for **local testing only** due to GPU memory constraints. Change to `128` on the cluster.
- All 5 datasets (BCI + HER2/ER/Ki67/PR) use identical model architecture and training hyperparameters for fair comparison.
- Results (inference images) go to `results/` folder. Training checkpoints go to `experiments/` folder. These are intentionally separate.

