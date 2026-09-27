# GCD: GCM-consistent Diffusion for Zero-shot Downscaling across Heterogeneous GCMs

[English README](README.md)

## 简介

异质的通用环流模型（GCM）的气候降尺度是一个病态逆问题：粗分辨率 GCM 输出与高分辨率再分析场来自不同的动力系统，且缺少细尺度时间配对数据。**GCD** 在无需针对特定 GCM 重新训练的前提下，将高分辨率气候扩散先验与粗分辨率 GCM 约束结合，实现零样本降尺度。

GCD 融合地理条件、频谱滤波测量引导和蒸馏加速的共轭梯度（CG）引导，在保持输入 GCM 一致性的同时，将推理步数从 1,000 步降低到 6 步。在五个异质 GCM 上，GCD 提升了极端事件的刻画能力，并恢复了热带气旋的高频结构。


## 安装

需要 Python 3.8+、CUDA GPU 和匹配 CUDA 的 PyTorch：

```bash
pip install -r requirements.txt
```

## 配置

- `configs/model_config_5var.yaml`：UNet 模型结构（5变量为例）
- `configs/distillation_config_4to1.yaml`：蒸馏训练配置（4→1）
- `configs/5var_miroc6_coarsen_nearest_cg.yaml`：降尺度推理任务（miroc6数据为例）
- `configs/diffusion_config.yaml`：扩散过程配置

## 数据

仓库不包含数据与模型权重。请自行获取数据，并在 `configs/distillation_config_4to1.yaml` 和 `configs/5var_miroc6_coarsen_nearest_cg.yaml` 的 `data` 部分填写路径。

## 模型权重
蒸馏训练需要预训练的 teacher 权重（也就是使用era5训练的扩散先验模型）。提供 teacher 权重后，可使用蒸馏训练生成 student 权重。




## 蒸馏训练

进行 4→1 蒸馏：DDIM-25 teacher → DDIM-6 student。

```bash
python distillation/train_progressive_distillation.py \
  --config configs/distillation_config_4to1.yaml \
  --gpus 1
```

多卡训练采用 DDP，例如使用四张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python distillation/train_progressive_distillation.py \
  --config configs/distillation_config_4to1.yaml \
  --gpus 4
```




## 降尺度推理
此处以miroc6数据降尺度推理为例，也可使用其他GCM数据进行zero-shot降尺度
```bash
python distillation/distilled_dps_inference_ddp.py \
  --model_config configs/model_config_5var.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/5var_miroc6_coarsen_nearest_cg.yaml \
  --distilled_ckpt checkpoints/progressive_4to1/student_round0_final.pt
```


## 引用

本文已被 NeurIPS 2026 接收。作者信息与论文链接确认后会补充。

```bibtex
@inproceedings{gcd2026,
  title     = {GCM-consistent Diffusion for Zero-shot Downscaling across Heterogeneous GCMs},
  author    = {xxx},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```