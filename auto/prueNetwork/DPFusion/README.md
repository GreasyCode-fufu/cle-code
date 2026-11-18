# DPFusion — CLE 三分类：成像先验网络（独立可运行包）

本目录是从 `~/Classification` 中抽出的**最小可运行子集**，只保留本文设计的网络结构与它真正依赖的文件。

## 1. 设计文件在哪

**核心文件：`models/cle_convnext.py`**（904 行）

| 行号 | 组件 | 说明 |
|---|---|---|
| 58 | `FreqExcite` | FADES：固定 Sobel/Laplacian 高通组 + 可学习门控（含 Haar 小波变体） |
| 131 | `AnisoDilatedBlock` | ADB：保留 7×7 depthwise，并联条带 + 空洞支路，零初始化 `alpha` |
| 184 / 206 | `DualPathHead` / `ResidualDetailHead` | 双路头 v1（已淘汰）/ 残差细节注入头 v2 |
| 236 | `FOVMask` | 圆形视野掩膜 + masked average pooling |
| 266 / 289 | `_sqrt_newton_schulz` / `CovPoolHead` | 协方差二阶池化头（iSQRT-COV） |
| 327 | `MultiExitHead` | 多出口聚合头（隐式深监督） |
| **352** | **`LocalContrastNorm`** | **LCN：9×9 局部对比度归一化（零参数）** |
| 386 | `CLEConvNeXt` | **主网络类**，复用 `models/Convnext.py` 的 `Block/LayerNorm/DropPath` |
| **738** | **`SpectralNotch`** | **SN：FFT 域 48 个可学习径向增益（最终推荐）** |
| 780 | `AdaptiveLCN` | 尺度自适应 LCN（从径向频谱预测 5/9/17 窗口权重） |
| 847 | `AngularSpectralNotch` | ASN：12×8（半径, 角度）掩膜 |
| **827** | **`cle_convnext_sn`** | **推荐模型的构造函数** |
| 675 | `cle_convnext_lcn_only` | 零参数方案构造函数 |
| 894 / 899 | `cle_convnext_dpf` / `_inflate_stem_dpf` | 双先验并行前端（特征级融合） |

网络整体结构（推荐模型 `cle_convnext_sn`）：

```
x (3×224×224)
  └─ SpectralNotch        # 可学习径向频谱掩膜，48 参数，恒等初始化
       └─ ConvNeXt-T stem (4×4/s4) → Stage1–4（原版 block，未改动）
            └─ LayerNorm → GAP → Linear(768→3)
```

参数量 **27.82 M**、**4.45 GMACs**，与基线 ConvNeXt-T 完全相同（新增仅 `log_gain` 的 48 个参数）。

## 2. test 结果（best-of-5，seed 42/46）

| 方案 | Macro-F1 | Accuracy | AUROC | Kappa | 错误数/576 |
|---|---|---|---|---|---|
| **+SN（推荐）** | **0.9740** | 0.9740 | 0.9971 | 0.9608 | 15 |
| DPFusion（LCN+SN 决策级融合） | 0.9721 | 0.9722 | 0.9954 | 0.9582 | 16 |
| +LCN | 0.9688 | 0.9688 | 0.9967 | 0.9530 | 18 |
| ConvNeXt-T（基线） | 0.9633 | 0.9635 | 0.9967 | 0.9452 | 21 |

## 3. 目录说明

```
DPFusion/
├── models/
│   ├── cle_convnext.py      ← 网络结构设计（唯一需要阅读的模型文件）
│   └── Convnext.py          ← 基线 ConvNeXt（被 cle_convnext.py import）
├── cle_fair_common.py       ← build_model 注册表 + 数据/指标工具
├── train_cle_fair.py        ← 训练入口（严格权重审计 + val Macro-F1 选 checkpoint）
├── train.py                 ← 训练脚本依赖（提供数据集 mean/std 计算）
├── utils.py / my_dataset.py / consensus_manifest.py
├── evaluate.py              ← 评估入口（11 项指标 + 复杂度 + FPS）
├── run.sh                   ← 启动器（自动选择 /opt/conda 解释器）
├── class_indices.json / requirements.txt
├── scripts/
│   ├── predict_fusion.py    ← 双先验决策级融合推理
│   ├── eval_all.py          ← 批量评估并聚合 11 指标
│   └── table_backbones.py   ← 生成主干对比表
├── weights/
│   ├── sn_best/             ← SN 最优权重（seed 42，test 0.9740）
│   └── lcn_best/            ← LCN 最优权重（seed 42，test 0.9688）
├── datasets -> ~/Classification/datasets   （软链接，避免复制 226 MB）
└── self_test.py             ← 一键自检（构建模型 + 复现两个 test 指标）
```

**注意**：`evaluate.py` 的注册表里保留了原项目的其他主干名（resnet34 等），但本包只带了
`models/Convnext.py` 与 `models/cle_convnext.py`，因此**只有以 `convnext_*` 与 `cle_convnext_*`
开头的模型可用**；需要其他主干时请从原项目补对应 `models/*.py`。

## 4. 快速使用

```bash
cd ~/DPFusion

# 0) 自检（应输出 SN 0.9740 / Fusion 0.9721）
./run.sh python self_test.py

# 1) 评估推荐模型
./run.sh eval --model cle_convnext_sn \
    --weights ./weights/sn_best/model_best.pth \
    --data-path ./datasets/cle_ac_v1/test --out-dir ./eval_out/sn

# 2) 双先验决策级融合
./run.sh python scripts/predict_fusion.py \
    --lcn-weights weights/lcn_best/model_best.pth \
    --sn-weights  weights/sn_best/model_best.pth \
    --data-path   ./datasets/cle_ac_v1/test --out-dir ./eval_out/fusion

# 3) 重新训练（协议与论文一致：224/bs32/AdamW 1e-4/wd 5e-2/Cosine 300ep）
./run.sh train --model cle_convnext_sn --num_classes 3 --img_size 224 --seed 42 \
    --data-path-train ./datasets/cle_ac_v1/train \
    --data-path-val   ./datasets/cle_ac_v1/val \
    --weights <ImageNet 预训练权重路径> \
    --save-dir ./weights_new --epochs 300
```

`--weights` 需要 ConvNeXt-T 的 ImageNet 权重
（`convnext_tiny_1k_224_ema.pth`，官方 `dl.fbaipublicfiles.com/convnext/`）。
原项目中位于 `~/Classification/pretrained/convnext_tiny_1k_224_ema.pth`，本包未复制（109 MB）。

## 5. 环境

conda 环境 `sunfuxiang`：Python 3.12.0 / PyTorch 2.12.0+cu130 / torchvision 0.27.0 /
timm 1.0.27 / scikit-learn 1.9.0 / thop / tqdm / rich。`run.sh` 会自动使用
`/opt/conda/envs/sunfuxiang/bin/python`。
