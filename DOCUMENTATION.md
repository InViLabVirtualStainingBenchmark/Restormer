--- 
# Restormer: BCI Virtual Staining
Local setup and initial training for H&E → IHC translation · BCI & MIST Datasets · Thomas's PC

> **Note:** This documents the local development setup used on Thomas's PC before HPC training.
> For HPC training, inference, and evaluation on the CalcUA cluster, see [HPC-INSTRUCTION.md](HPC-INSTRUCTION.md).
> See the [official Restormer repository](https://github.com/swz30/Restormer) for the original codebase.

---

## 1. Project Goal

This adapts Restormer (originally designed for image restoration tasks including denoising, deraining, and deblurring) to perform **virtual histological staining**:

- **Input:** H&E stained histology tiles
- **Output:** IHC stained equivalents

Trained and evaluated on the **BCI** and **MIST** datasets as part of the InViLab Virtual Staining Benchmark.

---

## 2. Environment Setup

A unified conda environment is used across all models in the benchmark:

```bash
conda activate vs_ua
```

| Component | Version |
|-----------|---------|
| Python | 3.9 |
| PyTorch | 2.1.2+cu121 |
| TorchVision | 0.16.2+cu121 |
| BasicSR | 1.4.2 |

> **Note:** Restormer uses BasicSR 1.4.2 (pip-installed into `vs_ua`) rather than a local copy. This differs from NAFNet which installs BasicSR via `setup.py develop`. See Section 4 for the modifications applied to the installed package.

---

## 3. Dataset Preparation

### BCI Dataset

```
data/BCI/
├── HE/
│   ├── train/      ← H&E input tiles
│   └── test/
└── IHC/
    ├── train/      ← IHC ground truth tiles
    └── test/
```

### MIST Dataset

```
data/MIST/
├── HER2/
│   └── TrainValAB/
│       ├── trainA/   ← H&E input
│       ├── trainB/   ← IHC ground truth
│       ├── valA/
│       └── valB/
├── ER/    (same structure)
├── Ki67/  (same structure)
└── PR/    (same structure)
```

> For HPC training, datasets are stored as SquashFS images with a neutral `HE/` / `IHC/` structure. See [HPC-INSTRUCTION.md](HPC-INSTRUCTION.md).

---

## 4. Modifications to BasicSR 1.4.2

Restormer requires several changes to the pip-installed BasicSR 1.4.2 package. All modifications were applied to:
```
~/miniconda3/envs/vs_ua/lib/python3.9/site-packages/basicsr/
```

The original unmodified BasicSR source is archived in `basicsr.zip` for reference.

### Mod 1 — Added Restormer architecture files

BasicSR 1.4.2 does not include Restormer's model or architecture. Copied from the local repo:

```bash
mkdir -p .../basicsr/models/archs/
cp basicsr/models/archs/restormer_arch.py    .../basicsr/models/archs/
cp basicsr/models/archs/arch_util.py         .../basicsr/models/archs/
cp basicsr/models/archs/__init__.py          .../basicsr/models/archs/
cp basicsr/models/image_restoration_model.py .../basicsr/models/
cp -r basicsr/models/losses/                 .../basicsr/models/
cp basicsr/models/lr_scheduler.py            .../basicsr/models/
```

### Mod 2 — Registered ImageCleanModel

Added `@MODEL_REGISTRY.register()` decorator to `image_restoration_model.py`:

```python
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class ImageCleanModel(BaseModel):
    ...
```

### Mod 3 — Added CosineAnnealingRestartCyclicLR scheduler

In `base_model.py` → `setup_schedulers()`:

```python
elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
    from basicsr.models.lr_scheduler import CosineAnnealingRestartCyclicLR
    for optimizer in self.optimizers:
        self.schedulers.append(
            CosineAnnealingRestartCyclicLR(optimizer, **train_opt['scheduler']))
```

### Mod 4 — Fixed test mode crash

In `image_restoration_model.py` — safe access when `train` key is absent during test:

```python
# FROM
self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
# TO
self.mixing_flag = self.opt.get('train', {}).get('mixing_augs', {}).get('mixup', False)
```

### Mod 5 — Added `validation()` wrapper

In `image_restoration_model.py` — bridges BasicSR 1.4.2 API signature:

```python
def validation(self, dataloader, current_iter, tb_logger, save_img=False):
    self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr=True, use_image=True)
```

### Mod 6 — Fixed experiment folder archiving

In `train.py` — prevents BasicSR from renaming the experiment folder when `--auto_resume` is used:

```python
# FROM
if resume_state is None:
    make_exp_dirs(opt)
# TO
if resume_state is None and not opt.get('auto_resume', False):
    make_exp_dirs(opt)
```

### Mod 7 — Fixed SameFileError

In `options.py` → `copy_opt_file()`:

```python
# FROM
copyfile(opt_file, filename)
# TO
if osp.abspath(opt_file) != osp.abspath(filename):
    copyfile(opt_file, filename)
```

### Mod 8 — Fixed path override

In `options.py` → `parse_options()` — changed hard-coded path generation to respect yml-defined paths:

```python
# Training
experiments_root = opt['path'].get('experiments_root', osp.join(root_path, 'experiments', opt['name']))
opt['path']['models'] = opt['path'].get('models', osp.join(experiments_root, 'models'))
opt['path']['training_states'] = opt['path'].get('training_states', osp.join(experiments_root, 'training_states'))
opt['path']['log'] = opt['path'].get('log', experiments_root)
opt['path']['visualization'] = opt['path'].get('visualization', osp.join(experiments_root, 'visualization'))

# Testing
results_root = opt['path'].get('results_root', osp.join(root_path, 'results', opt['name']))
opt['path']['results_root'] = results_root
opt['path']['log'] = opt['path'].get('log', results_root)
opt['path']['visualization'] = opt['path'].get('visualization', osp.join(results_root, 'visualization'))
```

> Without this fix, BasicSR ignores yml-defined paths and saves checkpoints to the conda environment directory.

---

## 5. Training

Training configs live in `experiments/`:

```
experiments/
├── BCI_Restormer/
│   ├── train_BCI.yml
│   └── test_BCI.yml
├── MIST_ER_Restormer/
│   ├── train_MIST_ER.yml
│   └── test_MIST_ER.yml
├── MIST_HER2_Restormer/  (same structure)
├── MIST_Ki67_Restormer/  (same structure)
└── MIST_PR_Restormer/    (same structure)
```

```bash
conda activate vs_ua
cd ~/virtual_stain/repos/Restormer

# BCI
python -m basicsr.train \
    -opt /home/thomas/virtual_stain/repos/Restormer/experiments/BCI_Restormer/train_BCI.yml \
    --auto_resume

# MIST
python -m basicsr.train \
    -opt /home/thomas/virtual_stain/repos/Restormer/experiments/MIST_ER_Restormer/train_MIST_ER.yml \
    --auto_resume
```

> Always use `--auto_resume` and **absolute paths** to the yml. Relative paths cause path resolution issues with BasicSR 1.4.2. `--auto_resume` prevents BasicSR from archiving the experiment folder on rerun.

Full benchmark training (100k iterations) runs on HPC — see [HPC-INSTRUCTION.md](HPC-INSTRUCTION.md).

---

## 6. Inference

```bash
conda activate vs_ua
cd ~/virtual_stain/repos/Restormer

python -m basicsr.test \
    -opt /home/thomas/virtual_stain/repos/Restormer/experiments/BCI_Restormer/test_BCI.yml
```

Results are saved to `results/restormer_BCI/visualization/BCI_Test/`. Each output image is named after its input filename.

---

## 7. Known Issues & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'ImageCleanModel'` | Model not registered | Add `@MODEL_REGISTRY.register()` — Mod 2 |
| `KeyError: 'scheduler'` | 1.4.2 expects `scheduler` key | Add `scheduler:` block to yml |
| `KeyError: 'mixing_augs'` | Missing key in yml | Add `mixing_augs: {mixup: false}` under `train:` |
| `KeyError: 'use_hflip'` | 1.4.2 dataset requires explicit flags | Add `use_hflip: true` and `use_rot: true` |
| `KeyError: 'train'` during test | Model accesses train config in test mode | Mod 4 |
| `SameFileError` | BasicSR copies yml to same location | Mod 7 |
| `CUDA out of memory` | Crop size too large | Reduce `gt_size` and `lq_size` to 64 |
| Folder getting archived | BasicSR renames existing experiment folders | Always use `--auto_resume` — Mod 6 |
| Checkpoints saving to conda env | `options.py` overrides yml paths | Mod 8 |

---

## 8. Notes

- **Always use absolute paths** in yml files and when passing `-opt` — BasicSR 1.4.2 has path resolution issues with relative paths.
- **Always use `--auto_resume`** when rerunning training.
- `total_iter: 1000` in the yml files is for local smoke testing only. Full training uses 100k on HPC.
- `gt_size: 64` is for local testing only. HPC uses 128.
- Results go to `results/`, checkpoints go to `experiments/` — intentionally separate.
- `basicsr.zip` in the repo root is the original unmodified BasicSR 1.4.2 source, kept as a diff reference.