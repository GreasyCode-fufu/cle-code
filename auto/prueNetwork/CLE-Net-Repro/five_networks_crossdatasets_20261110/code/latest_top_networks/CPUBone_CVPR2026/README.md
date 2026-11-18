# CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities

**[Paper (CVPR)](https://openaccess.thecvf.com/content/CVPR2026F/papers/Nottebaum_CPUBone_Efficient_Vision_Backbone_Design_for_Devices_with_Low_Parallelization_CVPRF_2026_paper.pdf)** | **[Paper (arXiv)](https://arxiv.org/abs/2603.26425)** | **CVPR Findings 2026**

Official repository for CPUBone, a family of vision backbones optimized for CPU-based inference.

Authors:  [Moritz Nottebaum](https://scholar.google.de/citations?user=y7paI7EAAAAJ&hl=de), [Matteo Dunnhofer](https://scholar.google.de/citations?user=GIhkF8UAAAAJ&hl=de&oi=ao) and [Christian Micheloni](https://scholar.google.de/citations?user=Gbnq0F8AAAAJ&hl=de&oi=ao)

**Model checkpoints are available now!!** [here](https://www.dropbox.com/scl/fo/42fb1n952cgrlrb0xokio/ANrEBwPz3R_9KIPhxH_PLzQ?rlkey=r3g2oa6sklx4n7kwh8jiomtqr&st=d6aojlxv&dl=0) (see Quick Start for more info). 

**Models B0,B1,B2 were retrained**: making them bf16/fp16 robust, and yielding improved accuracy. It is recommended to use the updated ones.
(*architecturally we added batchnorms to the mbconv block*)

**CPUBone models are soon in [TIMM library](https://github.com/huggingface/pytorch-image-models).**

---

## Overview

CPUs, unlike GPUs and other high-parallelization platforms, require models that balance the number of operations (MACs) and hardware-efficient execution measured by MACs per second (MACpS). CPUBone investigates two modifications to standard convolutions — **grouped convolutions** (groups=2) and **reduced kernel sizes** (2×2 instead of 3×3) — that reduce MACs while preserving MACpS on CPUs, achieving state-of-the-art speed–accuracy trade-offs across a wide range of devices.

CPUBone also transfers its efficiency to downstream tasks (object detection, semantic segmentation).

> If you are looking for a more general-purpose backbone (not CPU-specific), our [LowFormer](https://github.com/altair199797/LowFormer/tree/main) may be more suitable.

Latency benchmarks are run on exported models: desktop and embedded CPU numbers (RPi 5, Xeon) use ONNX Runtime, while mobile numbers (Pixel 7 Pro) use TFLite.

The full model definition is self-contained in [`cpubone_model.py`](cpubone_model.py) and requires only PyTorch.

---

## Model Zoo

### ImageNet-1K Classification

| Model | Params (M) | MACs (M) | Top-1 (%) | RPi 5 (ms) | Pixel 7 Pro (ms) | Xeon W-2125 (ms) |
|-------|-----------|----------|-----------|-----------|-----------------|-----------------|
| CPUBone-B0 | 10.4 | 519 | 77.6 | 24.2 | 13.8 | 6.9 |
| CPUBone-B1 | 12.4 | 746 | 78.7 | 33.5 | 18.6 | 9.4 |
| CPUBone-B2 | 23.9 | 1354 | 80.3 | 60.3 | 32.4 | 16.1 |
| CPUBone-B3 | 40.7 | 4054 | 83.1 | 199.8 | 83.1 | 34.1 |

Models and ablations can be found [here](https://www.dropbox.com/scl/fo/42fb1n952cgrlrb0xokio/ANrEBwPz3R_9KIPhxH_PLzQ?rlkey=r3g2oa6sklx4n7kwh8jiomtqr&st=d6aojlxv&dl=0) (just put the folder in the root directory). 
Latency measured at batch size 1. RPi 5 and Xeon numbers use ONNX Runtime; Pixel 7 Pro numbers use TFLite.

### Tiny Backbones

| Model | Params (M) | MACs (M) | Top-1 (%) | RPi 5 (ms) |
|-------|-----------|----------|-----------|-----------|
| CPUBone-Nano | 6.52 | 190 | 72.67 | — |
| CPUBone-T0 | 7.54 | 269 | 74.85 | 12.47 |
| CPUBone-S0 | 8.73 | 359 | 75.89 | 15.53 |

### BF16/FP16 Robust 

In contrast to the other model variants, CPUBone-B0/B1/B2 were not robust to bf16 and fp16 inference — see [`bf16_fp16_accuracies.txt`](bf16_fp16_accuracies.txt) for full accuracy numbers across all precisions. We recommend to use the cpubone_{b0|b1|b2}_bftraining version, where we also added batchnorm to the MBConv block, leading to higher stability and improved accuracies. The additional batchnorms do not change latency, as the batchnorm is fused into the convolution in execution frameworks like ONNXRuntime.


| Model | Params (M) | Top-1 fp32 (%) | Top-1 bf16 (%) | Top-1 fp16 (%) |
|-------|-----------|----------------|----------------|----------------|
| CPUBone-B0-bftraining | 10.4 | 77.79 | 77.72 | 77.78 |
| CPUBone-B1-bftraining | 12.4 | 78.95 | 78.90 | 78.91 |
| CPUBone-B2-bftraining | 23.9 | 80.78 | 80.72 | 80.76 |

Full Top-1 / Top-5 results for all models and precisions are in [`bf16_fp16_accuracies.txt`](bf16_fp16_accuracies.txt).

### Downstream Tasks

**Object Detection on COCO 2017** (RetinaNet framework):

| Model | AP (%) |
|-------|--------|
| CPUBone-B0 | 37.5 |
| CPUBone-B1 | 39.0 |
| CPUBone-B2 | 40.4 |
| CPUBone-B3 | 42.9 |

**Semantic Segmentation on ADE20K** (Semantic FPN framework):

| Model | mIoU (%) |
|-------|----------|
| CPUBone-B0 | 37.9 |
| CPUBone-B1 | 39.2 |
| CPUBone-B2 | 42.1 |
| CPUBone-B3 | 44.1 |

---

## Quick Start: Using a Pretrained Model

`cpubone_model.py` is a standalone file — the only dependency is PyTorch.

Model loading requires the YAML config files from the `configs/` directory of this repository. Each config defines the architecture hyperparameters (width, depth, etc.) for a given model variant.

```python
from cpubone_model import get_cpubone_b1
import torch

# Requires configs/cls/imagenet/cpubone_b1.yaml to be present
model = get_cpubone_b1(pretrained=True)
model.eval()

x = torch.randn(1, 3, 224, 224)
out = model(x)  # (1, 1000)
print(out.shape)
```

Available convenience functions: `get_cpubone_b0`, `get_cpubone_b1`, `get_cpubone_b2`, `get_cpubone_b3`.

Pretrained weights are expected at `.exp/cls/imagenet/cpubone_<name>/checkpoint/evalmodel.pt`. You can simply download  [checkpoint folder](https://www.dropbox.com/scl/fo/42fb1n952cgrlrb0xokio/ANrEBwPz3R_9KIPhxH_PLzQ?rlkey=r3g2oa6sklx4n7kwh8jiomtqr&st=d6aojlxv&dl=0) and put it into this project directory (might be neccessary to append a period (".") at the beginning to the folder after download). It contains all the models and ablations.
Both the config path and checkpoint path can be passed explicitly:

```python
from cpubone_model import get_cpubone

model = get_cpubone(
    config_path="configs/cls/imagenet/cpubone_b1.yaml",
    checkpoint_path="path/to/evalmodel.pt",
    pretrained=True,
)
```

---

## Installation

```bash
conda create -n cpubone python=3.11
conda activate cpubone
pip install -r requirements.txt
```

For ImageNet-1K training and evaluation, set the dataset path in the config file (`configs\cls\imagenet\default.yml`).
Checkpoints can be downloaded from [here](https://www.dropbox.com/scl/fo/42fb1n952cgrlrb0xokio/ANrEBwPz3R_9KIPhxH_PLzQ?rlkey=r3g2oa6sklx4n7kwh8jiomtqr&st=d6aojlxv&dl=0).

---

## Evaluation

```bash
python eval_cls_model.py --model cpubone_b1 --path /path/to/imagenet/val
```

`--path` must point to the ImageNet validation directory (the folder containing the class subfolders).

Additional flags:
- `--testrun` — skip weight loading and dataset, useful for architecture checks
- `--latency` — measure CPU/GPU latency with TorchScript
- `--bench` — full throughput benchmark

---

## Training

**Single GPU:**
```bash
torchrun --nproc_per_node=1 train_cls_model.py \
    configs/cls/imagenet/cpubone_b1.yaml \
    --path .exp/cls/imagenet/cpubone_b1
```

**Multi-GPU (8 GPUs):**
```bash
torchrun --nproc_per_node=8 train_cls_model.py \
    configs/cls/imagenet/cpubone_b1.yaml \
    --path .exp/cls/imagenet/cpubone_b1
```

To resume training, add `--resume`.

Gradient accumulation is controlled via the `bsizemult` parameter in the run config. Learning rate is scaled automatically with world size when `total_lr` and `total_batch_size` are set in the config.

---

## Custom Models

The easiest way to define a custom CPUBone variant is via the `custom` model name in a config file:

```yaml
net_config:
  name: custom
  width_list: [16, 32, 64, 128, 256]
  depth_list: [0, 1, 1, 4, 4]
  fastit: true
```

Or directly in Python by passing kwargs to `cpubone_backbone_b1`:

```python
from cpubone_model import cpubone_backbone_b1

backbone, width_list = cpubone_backbone_b1(
    name="custom",
    width_list=[16, 32, 64, 128, 256],
    depth_list=[0, 1, 1, 4, 4],
    fastit=True,
)
```

---

## Acknowledgements

The training infrastructure in the `cpubone/` package is built on top of [EfficientViT](https://github.com/mit-han-lab/efficientvit). The code in this repository also builds on our prior work [LowFormer](https://github.com/altair199797/LowFormer).

---

## Citation

```bibtex
@inproceedings{nottebaum2026cpubone,
  title     = {CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities},
  author    = {Nottebaum, Moritz and Dunnhofer, Matteo and Micheloni, Christian},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```
