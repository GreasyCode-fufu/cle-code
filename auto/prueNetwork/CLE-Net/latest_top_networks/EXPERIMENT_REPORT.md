# CLE_ac 最新网络对比实验报告

日期：2026-11-09（Asia/Shanghai）

## 结论

在完全相同的 CLE_ac 划分和训练协议下，CLE-Net（`convnext_egb_scg`）的三种子测试 Macro-F1 为 **0.9630±0.0053**，仍高于本次新增的 5 个公开网络。最有论文价值的新对照是 **MambaOut-Tiny：0.9572±0.0117**，只比 CLE-Net 低 0.58 个百分点，同时参数量从 31.28M 降到 24.25M。其次是 TransNeXt-Tiny（0.9523±0.0118）。

本次包含两个 2026 方法：ThinkingViT 是 **CVPR 2026 主会**；CPUBone 是 **CVPR Findings 2026**，报告和论文中不可写成主会论文。ConvNeur 虽是 CVPR 2026 主会且代码已下载，但作者仓库审计时没有可用的公开预训练权重链接，因此没有用“从头训练”冒充公平的预训练对照。

## 论文表格（独立测试集 576 张，seed 42/43/44）

| 方法 | 会议 | Params | MACs | Macro-F1 | Accuracy | Macro-AUROC | 相对 CLE-Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CLE-Net (convnext_egb_scg)** | 已发表网络 | 31.28M | 4.85G | **0.9630±0.0053** | — | — | — |
| MambaOut-Tiny | CVPR 2025 | 24.25M | 4.48G | 0.9572±0.0117 | 0.9572±0.0116 | 0.9953±0.0019 | −0.58 pp |
| TransNeXt-Tiny | CVPR 2024 | 27.63M | 5.53G | 0.9523±0.0118 | 0.9525±0.0116 | 0.9946±0.0016 | −1.07 pp |
| MambaVision-T | CVPR 2025 | 31.14M | 4.47G | 0.9449±0.0156 | 0.9456±0.0156 | 0.9936±0.0017 | −1.81 pp |
| ThinkingViT-DeiT 3H→6H | CVPR 2026 | 21.74M | 5.85G¹ | 0.9395±0.0194 | 0.9398±0.0201 | 0.9924±0.0022 | −2.35 pp |
| CPUBone-B2 | CVPR Findings 2026 | 21.31M | **1.36G** | 0.9189±0.0137 | 0.9207±0.0123 | 0.9894±0.0028 | −4.41 pp |

¹ ThinkingViT 为动态嵌套网络，普通 THOP 会只统计当前执行子网并错误报告 0.369M 参数。这里参数量用完整 checkpoint 静态计数 21.74M；MACs 使用作者对 `threshold=0.0`（全部进入第二阶段）的官方 5.85 GMACs。

## 逐种子 Test Macro-F1

| 方法 | seed 42 | seed 43 | seed 44 |
|---|---:|---:|---:|
| CLE-Net | 0.9617 | 0.9585 | 0.9688 |
| ThinkingViT | 0.9173 | 0.9530 | 0.9484 |
| CPUBone-B2 | 0.9338 | 0.9068 | 0.9162 |
| MambaVision-T | 0.9359 | 0.9629 | 0.9359 |
| MambaOut-Tiny | 0.9497 | 0.9706 | 0.9512 |
| TransNeXt-Tiny | 0.9495 | 0.9653 | 0.9422 |

## 公平实验协议

- 数据：train 2077、val 427、test 576，三类；使用项目既有患者级划分。
- 输入：224×224；加载 ImageNet-1K 权重时统一 ImageNet mean/std。
- 优化：AdamW，lr=1e-4，weight decay=0.05，batch size=32，CosineAnnealingLR。
- 最多 300 epochs，patience=30；只按验证集 Macro-F1 选择最佳 checkpoint。
- 正式重复：seed 42、43、44；损失为未加权 CrossEntropy；AMP 开启。
- 所有预训练权重均做严格键审计：只允许最终 1000→3 分类层缺失，其他可训练参数全部匹配，unexpected keys=0。
- CPUBone 和 MambaOut 的预训练头内还含特征投影层；适配器保留这些层，只替换最终分类线性层。

## 方法选择与代码

- ThinkingViT（CVPR 2026 主会）：https://github.com/ds-kiel/ThinkingViT
- CPUBone（CVPR Findings 2026）：https://github.com/altair199797/CPUBone
- MambaVision（CVPR 2025）：https://github.com/NVlabs/MambaVision
- MambaOut（CVPR 2025）：https://github.com/yuweihao/MambaOut 。名称中虽有 Mamba，但本文选择的网络是无 SSM 的 gated CNN。
- TransNeXt（CVPR 2024）：https://github.com/DaiShiResearch/TransNeXt

官方代码分别保存在 `latest_top_networks/` 下独立子目录。另保留 ConvNeur、VMamba、UniRepLKNet 作为候选，不混入当前结果。

## 与 ZCode 任务隔离

ZCode 的轻量化网络任务、`latest_lightweight_nets`、其 checkpoint 和进程均未修改，也未纳入本表。本任务仅写入 `CLE-Net/latest_top_networks/`。评估期间观察到 ZCode 同时运行 4 个 `convnext_tiny` 进程，因此 GPU FPS 会受并发负载影响；本报告明确不使用该次 FPS 做论文结论，分类指标、Params 与 MACs不受此限制。

## 文件索引

- `comparison_per_seed.csv`：15 次测试的逐种子完整指标。
- `comparison_summary.csv/json`：均值与样本标准差。
- `paper_comparison_table.csv`：含 CLE-Net 的论文表格。
- `results/<model>/seed<seed>/<model>/`：checkpoint、训练日志、run_meta 和测试评估。
- `adapters.py`：不修改上游源码的统一三分类适配器。
- `run_train.py` / `run_eval.py`：公平训练与评估入口。
- `METHODS_MANIFEST.json`：论文、会议、代码、权重和候选状态。
- `environment.yml` / `pip_freeze.txt`：专用环境快照。
