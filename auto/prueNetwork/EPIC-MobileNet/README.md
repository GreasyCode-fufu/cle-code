# EPIC-MobileNet（暂定名）

**证据保持 · FOV 感知 · 凸分类头的轻量 CLE 图像三分类网络**（MobileNetV3-Large 基座 + 3 个创新点）。

从 `/home/login/data/cle/auto/ZCode/Classification`（R6 轮）提取的独立发布版（2026-11-09）。
EPIC = **E**vidence-**P**reserving, FOV-**I**ntegrated, **C**onvex-head。

## 网络结构

| 创新点 | 环节 | 做法 | 主域单步增益（5 seeds） |
|---|---|---|---|
| **① 证据保持微调** | 阶段 1 训练 | 冻结 /2–/16（参数 + BN 推理态），只微调 /32；可迁移性门控决定是否启用 | +1.46 F1 |
| **② 解耦凸证据头（DCE）** | 阶段 2 拟合 | 冻结主干，强 L2 多项逻辑回归拟合分类头，C 仅在 val 上选；部署为单个 Linear | +0.43 F1，恶性敏感度 +3.07，跨 seed std 1.07→0.30 |
| **③ FOV 感知多尺度证据** | 证据池化 | /4、/8、/16、/32 四抽头，仅在圆形视野组织区掩码平均（1/4 分辨率计算） | +0.46 F1 |

部署结构 = MobileNetV3-L `features` + 四抽头 FOV 掩码平均池化 + `(z−μ)/σ` + `Linear(1136, K)`。
工厂：`mbv3_r6_fz16`（阶段 1）、`mbv3_dce4m` / `mbv3_fz16_dce4m`（部署/阶段 2 输出），见 `models/optimize/mbv3_cle_r6.py`。

## 性能（cle_ac_v1 test，576 张，5 seeds 42–46 同批同协议）

| 指标 | MobileNetV3-L 基线 | **EPIC-MobileNet** | Δ |
|---|---|---|---|
| Macro-F1 | 94.16±0.56 | **96.51±0.40** | **+2.35** |
| Cohen's Kappa | 91.22 | **94.73** | **+3.51** |
| MCC | 91.29 | **94.73** | **+3.44** |
| Balanced Acc | 94.02 | **96.54** | +2.52 |
| Macro-AUROC | 99.10 | **99.57** | +0.46 |
| **恶性敏感度** | 90.00 | **96.70** | **+6.70** |
| 参数 | 4.206M | **2.975M** | **−29.3%** |
| MACs | 0.233G | 0.232G | −0.5% |
| CPU 单线程 / 4 线程延迟 | 6.12 / 4.05 ms | 6.23 / 4.20 ms | +1.8% / +3.7% |
| GPU bs=1 延迟 / bs64 吞吐 | 1.193 ms / 19166 张/s | 1.405 ms / 17611 张/s | +18% / −8% |

- 官方 3 seeds（42/43/44）：96.43 vs 94.08（+2.35）。5/5 seeds 的 ΔF1 配对 bootstrap 95% CI > 0；跨 seed 配对 t 检验 p = 3.4×10⁻⁵。
- 漏诊恶性（癌/异型增生 → 良性）5 seeds 合计 63 → 18（−71%）。
- 对比轻量主干（同协议 5 seeds，官方代码+官方 ImageNet 权重）：
  最新网络 iFormer-T（ICLR'25）94.26、EfficientViM-M1（CVPR'25）94.08、SHViT-S1（CVPR'24）93.41、LSNet-T（CVPR'25）91.23
  （LSNet 的 Triton 动态核无 CPU 部署路径）；经典 StarNet-S2 93.47、GhostNet 93.08、FasterNet-T0 93.06、
  ShuffleNetV2 92.49、MobileNetV4-Conv-S 89.71 —— EPIC 对以上全部 5/5 seeds 配对胜出（vs 最新网络 +2.25 ~ +5.27），
  且参数 2.98M / MACs 0.23G / CPU 8.0ms / GPU 1.39ms 同时占优（仅 iFormer 参数略少，但 MACs 2.2×、CPU 1.3× 慢）。
- **跨域（5 域 × 3 seeds，同批 baseline，门控选档）**：BUSI +0.51、heelBone +4.79、covid +0.33、CUB-200 +4.90、TCGA +1.45 ——
  连同主域 **6/6 域为正**；可迁移性门控（只用 train/val）在 6/6 域选中最优档，TCGA 为预注册样本外检验（预测成立）。

## 文件清单

```
EPIC-MobileNet/
├── README.md / 创新说明.md / 运行与测试说明.md
├── demo.py                         # 独立构建 + 前向演示(可选加载权重对单图分类)
├── epic_gate.py                    # 可迁移性门控: 新数据集上是否启用 ①(只用 train/val)
├── epic_refit.py                   # 阶段 2: 解耦凸证据头重拟合(②+③), 输出 evaluate.py 可直接评测的权重
├── requirements.txt / class_indices.json
├── models/
│   ├── MobileNetV3.py              # MobileNetV3-L 基座(torchvision 同构)
│   ├── timm_light.py               # 对比主干(StarNet-S2 / MobileNetV4-Conv-S, 需 timm)
│   └── optimize/mbv3_cle_r6.py     # ★ EPIC 本体(MBV3R6 冻结/MBV3Convex 凸证据头/fov_weights) + R6 全部候选工厂
├── train_cle_fair.py               # 训练入口(原协议; 严格加载审计)
├── evaluate.py / cle_fair_common.py / train.py / my_dataset.py / utils.py / consensus_manifest.py
└── experiment_data/                # 全部结果 CSV + run_meta 审计 + _build_data.py(数据溯源)
```

依赖：`torch >= 2.0`、torchvision、scikit-learn、Pillow、numpy、scipy、tqdm、thop（对比主干另需 timm）。

## 快速开始

```bash
python demo.py            # 构建 -> 合成 CLE 样图前向(不依赖数据集/权重)
python demo.py --cpu
```
完整训练（阶段 1 → 门控 → 阶段 2 → 评测）见 `运行与测试说明.md`。
