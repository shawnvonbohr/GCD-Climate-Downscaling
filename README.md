# GCD: GCM-consistent Diffusion for Zero-shot Downscaling across Heterogeneous GCMs

[Chinese README](README_CN.md)

## Overview

Climate downscaling across heterogeneous General Circulation Models (GCMs) is an ill-posed inverse problem: coarse GCM outputs and high-resolution reanalysis fields arise from different dynamical systems and lack fine-scale temporal pairing. **GCD** performs zero-shot downscaling by combining a high-resolution climate diffusion prior with coarse GCM constraints, without model-specific retraining.

GCD integrates geographic conditioning, spectrally filtered measurement guidance, and distillation-accelerated Conjugate Gradient (CG) guidance. It reduces inference from 1,000 diffusion steps to 6 while preserving input-GCM consistency. Across five heterogeneous GCMs, GCD improves extreme-event fidelity and recovers high-frequency tropical-cyclone structures.

## Installation

Python 3.8+, a CUDA GPU, and a CUDA-compatible PyTorch build are required.

```bash
pip install -r requirements.txt
```

## Configuration

- `configs/model_config_5var.yaml`: UNet architecture configuration (five variables as an example).
- `configs/distillation_config_4to1.yaml`: 4-to-1 distillation training configuration.
- `configs/5var_miroc6_coarsen_nearest_cg.yaml`: downscaling inference task configuration (MIROC6 as an example).
- `configs/diffusion_config.yaml`: diffusion-process configuration.

## Data 

Data and model weights are not included. Obtain the data and set the paths under `data` in both `configs/distillation_config_4to1.yaml` and `configs/5var_miroc6_coarsen_nearest_cg.yaml`.

## Model weights
Progressive distillation requires a pretrained teacher checkpoint, namely the diffusion-prior model trained on ERA5. once it is obtained , we can use the distillation training to produce the student model.

## Distillation training

Run 4-to-1 distillation: DDIM-25 teacher → DDIM-6 student.

```bash
python distillation/train_progressive_distillation.py \
  --config configs/distillation_config_4to1.yaml \
  --gpus 1
```

Training uses DDP for multiple GPUs, for example, to use four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python distillation/train_progressive_distillation.py \
  --config configs/distillation_config_4to1.yaml \
  --gpus 4
```



## Downscaling inference

The command below uses MIROC6 as an example. Other GCM data can also be used for zero-shot downscaling.

```bash
python distillation/distilled_dps_inference_ddp.py \
  --model_config configs/model_config_5var.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/5var_miroc6_coarsen_nearest_cg.yaml \
  --distilled_ckpt checkpoints/progressive_4to1/student_round0_final.pt
```

## Citation

This work has been accepted at NeurIPS 2026. Author names and publication links will be added when available.

```bibtex
@inproceedings{gcd2026,
  title     = {GCM-consistent Diffusion for Zero-shot Downscaling across Heterogeneous GCMs},
  author    = {xxx},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```