# SF-ConvNeXt

**尺度–频率感知的共聚焦激光显微内镜（CLE）图像三分类网络**（ConvNeXst-T 基座 + 2 个结构化创新点）。

从 `/home/login/data/cle/auto/ZCode/Classification` 项目提取的独立发布版（2026-10-26，源仓库 commit `f73097d` 一致）。

## 网络结构

在 ConvNeXt-T（`models/Convnext.py`）上做两处**结构化替换**（不可旁路，区别于并联门控式即插即用）：

| 创新点 | 位置 | 结构 |
|---|---|---|
| **WR 频域多尺度卷积块** | stage-3/4 全部 12 个 block | 7×7 深度卷积整体替换为 WTConv：基卷积分支（7×7，**直接继承 ImageNet 预训练权重**）+ Haar 两级级联小波路径（各子带 3×3 分组深度卷积，尺度初始 0.1，逆变换重构）。感受野随分解级数指数扩展，参数线性增加 |
| **CH 三尺度拼接分类头** | 分类端 | 头输入 = [LN₄(GAP(f₄)); LN₃(GAP(P₃f₃)); LN₂(GAP(P₂f₂))] ∈ R^2304（fᵢ 为 stage-i 特征，P 为 1×1 投影），分类器稠密读取全部三尺度证据 |

主类 `ConvNeXtV2`，工厂函数 `convnext_v2bc`（`models/optimize/convnext_cle_v2.py`）。
同文件还提供消融变体工厂：`convnext_v2a/v2b/v2c/v2ab/v2abc/v2ac/v2bc4`（含 WS 小波茎干替换与 WR 位置消融）。

## 文件清单

```
SF-ConvNeXt/
├── demo.py                          # 独立构建+加载+推理演示(仅依赖 PyTorch)
├── README.md
├── requirements.txt
├── models/
│   ├── Convnext.py                  # ConvNeXt-T/S 基座(facebookresearch 官方实现)
│   └── optimize/
│       ├── convnext_cle.py          # Haar 小波算子 + WTConv2d(ECCV 2024 移植, 内联 db1 免 pywt)
│       └── convnext_cle_v2.py       # ★ SF-ConvNeXt 本体(ConvNeXtV2 + 全部工厂)
└── weights/
    ├── sf_convnext_best.pth         # 最优训练权重(seed42)
    ├── run_meta.json                # 训练元数据(数据/协议/预训练审计)
    └── final_eval_report/           # 最终 test 评估报告(含 1000 次 bootstrap 95% CI)
```

依赖：`torch >= 2.0`（无 timm / pywt / 其它第三方依赖）。

## 快速开始

```bash
python demo.py            # 构建 -> 加载最优权重 -> 前向
python demo.py --cpu
```

推理预处理（与训练一致）：

```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet 统计
                         std=[0.229, 0.224, 0.225]),
])
```

类别顺序：`["Benign_lesions", "Malignant_lesions", "body_of_stomach_normal"]`。

## 性能（cle_ac_v1 测试集，576 张，三分类）

| 指标 | ConvNeXt-T 基线(5seed均值) | **SF-ConvNeXt(3seed均值)** | 最优 run(95% bootstrap CI) |
|---|---|---|---|
| Macro-F1 | 0.9476±0.0084 | **0.9600±0.0065** | **0.9672** [0.9518, 0.9806] |
| Accuracy | 0.9479 | **0.9601** | 0.9670 [0.9514, 0.9809] |
| Cohen's Kappa | 0.9216 | **0.9399 (+1.83)** | 0.9504 |
| MCC | 0.9222 | **0.9401 (+1.79)** | 0.9505 |
| Macro-AUROC | 0.9940 | **0.9964** | 0.9967 [0.9939, 0.9987] |
| Params / MACs | 27.80M / 4.45G | 28.67M / 4.64G | — |

在 18 个对比主干（ResNet/ResNeXt/MobileNetV3/ShuffleNetV2/GhostNet/FasterNet/DeiT-T/S/B/ViT-B/Swin-T/PVT-T/PoolFormer/ConvNeXt-S 等）中 Macro-F1 排名第 1，优于 3 倍参数的 DeiT-B（+1.25pt）与 2 倍参数的 ConvNeXt-S（+0.62pt）。逐类（最优 run）：恶性 sens/spec/prec = 0.960/0.985/0.966。

## 训练协议（复现用）

AdamW lr=1e-4 wd=5e-2，余弦退火 T_max=300，bs=32，224²，AMP；增广 = 256² resize + 水平/垂直翻转 + ±15° 旋转 + 随机裁剪；ImageNet 归一化；按验证集 Macro-F1 选 checkpoint（早停 30）。ImageNet-1k 官方 ConvNeXt-T ema 权重经键重映射严格加载（`train_cle_fair.py` 的 `MODEL_CHECKPOINT_TRANSFORMS["convnext_v2bc"]`，在源仓库中）。

## 引用说明

- ConvNeXt: Liu et al., "A ConvNet for the 2020s", CVPR 2022
- WTConv 算子: Finder et al., "Wavelet Convolutions for Large Receptive Fields", ECCV 2024（本实现为内联 Haar(db1) 的忠实移植；WR 的替换式集成/预训练继承/深度选择为本文设计）
