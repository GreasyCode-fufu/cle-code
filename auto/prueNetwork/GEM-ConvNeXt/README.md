# GEM-ConvNeXt

**梯度先验注入与跨尺度证据聚合的 CLE 图像三分类网络**（ConvNeXt-T 基座 + 3 个结构化创新点）。

从 `/home/login/data/cle/auto/ZCode/Classification` 提取的独立发布版（2026-11-08）。

## 网络结构

| 创新点 | 位置 | 结构 |
|---|---|---|
| **MSP 多尺度证据投影** | 特征端 | stage-2/3 经无偏置 1×1 投影到 768 维语义空间 + 各尺度 LayerNorm |
| **DCH 稠密跨尺度头** | 分类端 | `[LN₄(GAP(f₄)); LN₃(GAP(P₃f₃)); LN₂(GAP(P₂f₂))] ∈ R²³⁰⁴` 拼接 → Linear(2304,K)，不可旁路 |
| **FGS 梯度增广茎干** | 输入端 | 第 4 通道 = Sobel 梯度幅值（z-score）；预训练权重膨胀 + 新通道零初始化（初始逐位等价，实测 0.00e+00） |

主类 `ConvNeXtV5`，工厂 `convnext_v6f`（最终网络）、`convnext_v2m`（MSP-only 消融档）
（`models/optimize/convnext_cle_v5.py`）；`convnext_v2c`（+MSP+DCH 档）在
`models/optimize/convnext_cle_v2.py`（未随包，消融数据见 experiment_data）。

## 性能（cle_ac_v1 test，576 张，3 seeds 同批）

| 指标 | ConvNeXt-T 基线 | **GEM-ConvNeXt** | Δ |
|---|---|---|---|
| Macro-F1 | 0.9489±0.0099 | **0.9629±0.0023** | **+1.40pt** |
| Cohen's Kappa | 0.9234 | **0.9443** | **+2.09pt** |
| MCC | 0.9236 | **0.9446** | **+2.10pt** |
| 恶性敏感度 | 0.9337 | **0.9583** | **+2.46pt** |
| Params / MACs / FPS | 27.82M / 4.45G / 848 | 28.27M / 4.63G / 820 | +1.6% / +4% / -3% |

跨域（vs 同批 baseline，3 seeds）：CUB-200 **+2.55**、TCGA **+1.33**、heelBone +0.79、
covid +0.21、BUSI -0.48。消融链与全部数据见 `experiment_data/`。

## 文件清单

```
GEM-ConvNeXt/
├── README.md / 创新说明.md / 运行与测试说明.md
├── demo.py                       # 独立构建+前向演示(仅依赖 PyTorch)
├── requirements.txt / class_indices.json
├── models/
│   ├── Convnext.py               # ConvNeXt-T/S 基座(facebookresearch 官方实现)
│   └── optimize/convnext_cle_v5.py  # ★ GEM 本体(ConvNeXtV5 + MSP/DCH/FGS/CSAP/XRM 等工厂)
├── train_cle_fair.py             # 训练入口(含 FGS 膨胀变换/白名单审计/focal/EMA/margin)
├── evaluate.py / cle_fair_common.py / train.py / my_dataset.py / utils.py / consensus_manifest.py
└── experiment_data/
    ├── cle_main_results.csv      # 主域 4 配置 × 3 seeds × 9 指标 + MEAN 行
    ├── crossdomain_ablation.csv  # 5 域 × 4 配置逐 seed F1
    ├── model_complexity.csv      # params/MACs/FPS(独占 GPU 口径)
    ├── negative_screening_summary.csv  # 25 个证伪候选汇总
    ├── run_meta/                 # 最终网络 3 seeds 训练元数据(预训练加载审计)
    └── _build_data.py            # 上述 CSV 的生成脚本(数据溯源)
```

依赖：`torch >= 2.0`（训练/评测另需 torchvision、scikit-learn、Pillow、numpy、tqdm、thop）。

## 快速开始

```bash
python demo.py            # 构建 -> 随机权重前向(不依赖数据集/权重文件)
python demo.py --cpu
```

推理预处理（与训练一致）：Resize(256) → CenterCrop(224) → ImageNet 归一化。
类别顺序：`["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]`。

## 引用说明

- ConvNeXt: Liu et al., "A ConvNet for the 2020s", CVPR 2022
- MSP/DCH/FGS 组合、零初始等价茎干膨胀、跨域消融链为本文设计
