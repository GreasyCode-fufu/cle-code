# L1-96 任务感知 / 类别条件谱 + Fisher 判别谱分析报告
- 运行耗时 33.32s | Ultralytics 8.4.84 / torch 2.4.1+cu121 / CUDA 12.1 / NVIDIA GeForce RTX 4090
- 类别映射: {0: 'spalling', 1: 'crack', 2: 'moisture'}; Fisher 类别 K=4: ['background', 'spalling', 'crack', 'moisture'] (Sb 秩 ≤ K-1 = 3)

## A. 配置
- 权重 `/data/cle/sunfuxiang-2/ultralytics-main/runs/detect/runs/detect/hongkong_L1-96/weights/best.pt`; 数据 yaml `/data/cle/sunfuxiang-2/ultralytics-main/runs/analysis/l1_96_task_aware_spectrum/_auto_data.yaml` (auto-generated, check_det_dataset=ok)
- split=val, imgsz=640, batch=8, device=`cuda:0`, FP32, seed=42
- 分析图像: **1105** / 1105
- 预处理: 复用 `LetterBox(auto=False, stride=32)`, GT 框经 get_params(ratio/left/top) 精确映射到 640 → 特征图.
- 采样: natural (自然频率) + balanced (每类≤512, 背景≤512, 每框≤256).
- Fisher 正则 eps=0.0001 (Sw+eps·trace(Sw)/C·I), 白化经 eigh (不直接求 inverse).

## B. 跨类别重叠统计 (natural, 前景位置中落入≥2类的比例)
| feature | overlap | foreground | overlap/fg |
|---|---|---|---|
| layer12_output | 2544 | 346017 | 0.007 |
| layer15_output | 10096 | 1380428 | 0.007 |
| layer18_output | 2544 | 346017 | 0.007 |
| layer21_output | 679 | 86064 | 0.008 |
| layer12_cv1 | 2544 | 346017 | 0.007 |
| layer12_bottleneck_last | 2544 | 346017 | 0.007 |
| layer12_prefuse | 2544 | 346017 | 0.007 |
| layer18_cv1 | 2544 | 346017 | 0.007 |
| layer18_bottleneck_last | 2544 | 346017 | 0.007 |
| layer18_prefuse | 2544 | 346017 | 0.007 |
| layer21_cv1 | 679 | 86064 | 0.008 |
| layer21_bottleneck_last | 679 | 86064 | 0.008 |
| layer21_prefuse | 679 | 86064 | 0.008 |

## C. 类别条件中心化协方差谱 (r95/C, balanced)
| feature | background | spalling | crack | moisture | global |
|---|---|---|---|---|---|
| layer12_bottleneck_last | 0.79 | 0.79 | 0.81 | 0.79 | 0.79 |
| layer12_cv1 | 0.76 | 0.77 | 0.79 | 0.74 | 0.78 |
| layer12_output | 0.71 | 0.72 | 0.72 | 0.71 | 0.73 |
| layer12_prefuse | 0.74 | 0.75 | 0.77 | 0.73 | 0.76 |
| layer15_output | 0.69 | 0.75 | 0.75 | 0.72 | 0.72 |
| layer18_bottleneck_last | 0.79 | 0.79 | 0.79 | 0.79 | 0.79 |
| layer18_cv1 | 0.67 | 0.70 | 0.70 | 0.68 | 0.69 |
| layer18_output | 0.62 | 0.68 | 0.68 | 0.65 | 0.67 |
| layer18_prefuse | 0.67 | 0.67 | 0.67 | 0.67 | 0.69 |
| layer21_bottleneck_last | 0.72 | 0.70 | 0.71 | 0.70 | 0.73 |
| layer21_cv1 | 0.66 | 0.64 | 0.70 | 0.58 | 0.69 |
| layer21_output | 0.61 | 0.62 | 0.66 | 0.59 | 0.65 |
| layer21_prefuse | 0.65 | 0.60 | 0.65 | 0.57 | 0.67 |

### C2. 类别条件 r95 / 有效秩 明细 (balanced)

| feature | group | C | r95 | r95/C | stable | eff | part |
|---|---|---|---|---|---|---|---|
| layer12_bottleneck_last | background | 48 | 38 | 0.792 | 7.8 | 29.6 | 20.0 |
| layer12_bottleneck_last | spalling_union | 48 | 38 | 0.792 | 8.1 | 28.3 | 18.5 |
| layer12_bottleneck_last | crack_union | 48 | 39 | 0.812 | 8.1 | 31.3 | 21.7 |
| layer12_bottleneck_last | moisture_union | 48 | 38 | 0.792 | 8.4 | 30.5 | 21.0 |
| layer12_cv1 | background | 96 | 73 | 0.760 | 10.3 | 53.4 | 31.8 |
| layer12_cv1 | spalling_union | 96 | 74 | 0.771 | 10.4 | 54.6 | 32.7 |
| layer12_cv1 | crack_union | 96 | 76 | 0.792 | 11.7 | 55.4 | 33.8 |
| layer12_cv1 | moisture_union | 96 | 71 | 0.740 | 9.2 | 51.3 | 30.2 |
| layer12_output | background | 96 | 68 | 0.708 | 9.7 | 48.6 | 29.3 |
| layer12_output | spalling_union | 96 | 69 | 0.719 | 9.5 | 47.5 | 27.5 |
| layer12_output | crack_union | 96 | 69 | 0.719 | 9.7 | 47.9 | 27.8 |
| layer12_output | moisture_union | 96 | 68 | 0.708 | 10.4 | 49.0 | 29.6 |
| layer12_prefuse | background | 144 | 106 | 0.736 | 10.9 | 71.4 | 38.4 |
| layer12_prefuse | spalling_union | 144 | 108 | 0.750 | 10.7 | 71.4 | 36.8 |
| layer12_prefuse | crack_union | 144 | 111 | 0.771 | 11.4 | 75.3 | 41.1 |
| layer12_prefuse | moisture_union | 144 | 105 | 0.729 | 10.1 | 71.1 | 37.7 |
| layer15_output | background | 64 | 44 | 0.688 | 9.7 | 34.7 | 23.1 |
| layer15_output | spalling_union | 64 | 48 | 0.750 | 11.7 | 39.5 | 27.8 |
| layer15_output | crack_union | 64 | 48 | 0.750 | 9.6 | 38.0 | 25.6 |
| layer15_output | moisture_union | 64 | 46 | 0.719 | 8.3 | 36.1 | 23.2 |
| layer18_bottleneck_last | background | 48 | 38 | 0.792 | 7.6 | 31.7 | 22.2 |
| layer18_bottleneck_last | spalling_union | 48 | 38 | 0.792 | 9.3 | 31.8 | 23.3 |
| layer18_bottleneck_last | crack_union | 48 | 38 | 0.792 | 8.1 | 31.4 | 22.3 |
| layer18_bottleneck_last | moisture_union | 48 | 38 | 0.792 | 9.7 | 32.1 | 23.7 |
| layer18_cv1 | background | 96 | 64 | 0.667 | 9.0 | 41.6 | 23.5 |
| layer18_cv1 | spalling_union | 96 | 67 | 0.698 | 9.6 | 43.7 | 24.9 |
| layer18_cv1 | crack_union | 96 | 67 | 0.698 | 7.6 | 42.9 | 23.2 |
| layer18_cv1 | moisture_union | 96 | 65 | 0.677 | 6.6 | 39.8 | 20.3 |
| layer18_output | background | 96 | 60 | 0.625 | 8.5 | 40.4 | 23.6 |
| layer18_output | spalling_union | 96 | 65 | 0.677 | 10.0 | 42.6 | 24.9 |
| layer18_output | crack_union | 96 | 65 | 0.677 | 8.4 | 42.9 | 24.6 |
| layer18_output | moisture_union | 96 | 62 | 0.646 | 6.9 | 38.6 | 20.6 |
| layer18_prefuse | background | 144 | 96 | 0.667 | 10.1 | 62.4 | 33.1 |
| layer18_prefuse | spalling_union | 144 | 97 | 0.674 | 11.7 | 65.0 | 36.1 |
| layer18_prefuse | crack_union | 144 | 97 | 0.674 | 9.7 | 63.3 | 33.7 |
| layer18_prefuse | moisture_union | 144 | 96 | 0.667 | 10.3 | 62.3 | 32.8 |
| layer21_bottleneck_last | background | 128 | 92 | 0.719 | 11.9 | 64.8 | 38.8 |
| layer21_bottleneck_last | spalling_union | 128 | 89 | 0.695 | 11.4 | 55.2 | 30.1 |
| layer21_bottleneck_last | crack_union | 128 | 91 | 0.711 | 11.7 | 58.8 | 33.1 |
| layer21_bottleneck_last | moisture_union | 128 | 90 | 0.703 | 12.0 | 60.4 | 34.8 |
| layer21_cv1 | background | 256 | 168 | 0.656 | 10.5 | 95.7 | 42.9 |
| layer21_cv1 | spalling_union | 256 | 163 | 0.637 | 11.4 | 89.8 | 42.2 |
| layer21_cv1 | crack_union | 256 | 178 | 0.695 | 12.1 | 103.7 | 50.1 |
| layer21_cv1 | moisture_union | 256 | 149 | 0.582 | 8.6 | 78.9 | 33.5 |
| layer21_output | background | 256 | 155 | 0.605 | 9.9 | 83.4 | 37.3 |
| layer21_output | spalling_union | 256 | 159 | 0.621 | 13.3 | 80.2 | 37.4 |
| layer21_output | crack_union | 256 | 168 | 0.656 | 13.5 | 92.4 | 46.0 |
| layer21_output | moisture_union | 256 | 151 | 0.590 | 11.3 | 80.3 | 38.1 |
| layer21_prefuse | background | 384 | 250 | 0.651 | 12.6 | 135.7 | 57.6 |
| layer21_prefuse | spalling_union | 384 | 231 | 0.602 | 16.1 | 116.9 | 50.2 |
| layer21_prefuse | crack_union | 384 | 251 | 0.654 | 14.3 | 132.4 | 58.4 |
| layer21_prefuse | moisture_union | 384 | 219 | 0.570 | 12.3 | 115.4 | 49.6 |

## D. Fisher 判别谱
| feature | mode | nonzero(>1e-6) | eff_rank | d90 | d95 | d99 | top1 | top2 | top3 | E(top3) |
|---|---|---|---|---|---|---|---|---|---|---|
| layer12_bottleneck_last | natural | 3 | 1.50 | 2 | 2 | 2 | 0.292 | 0.038 | 0.003 | 1.000 |
| layer12_bottleneck_last | balanced | 3 | 2.13 | 2 | 3 | 3 | 0.414 | 0.114 | 0.045 | 1.000 |
| layer12_cv1 | natural | 3 | 2.24 | 2 | 2 | 3 | 0.741 | 0.422 | 0.053 | 1.000 |
| layer12_cv1 | balanced | 3 | 2.66 | 3 | 3 | 3 | 1.880 | 1.006 | 0.535 | 1.000 |
| layer12_output | natural | 3 | 2.13 | 2 | 2 | 3 | 0.690 | 0.321 | 0.037 | 1.000 |
| layer12_output | balanced | 3 | 2.67 | 3 | 3 | 3 | 1.479 | 0.800 | 0.430 | 1.000 |
| layer12_prefuse | natural | 3 | 2.25 | 2 | 2 | 3 | 0.795 | 0.466 | 0.057 | 1.000 |
| layer12_prefuse | balanced | 3 | 2.68 | 3 | 3 | 3 | 1.961 | 1.069 | 0.581 | 1.000 |
| layer15_output | natural | 3 | 2.06 | 2 | 3 | 3 | 0.355 | 0.107 | 0.027 | 1.000 |
| layer15_output | balanced | 3 | 2.70 | 3 | 3 | 3 | 0.727 | 0.379 | 0.234 | 1.000 |
| layer18_bottleneck_last | natural | 3 | 1.17 | 1 | 1 | 2 | 0.388 | 0.010 | 0.002 | 1.000 |
| layer18_bottleneck_last | balanced | 3 | 1.88 | 2 | 3 | 3 | 0.280 | 0.045 | 0.025 | 1.000 |
| layer18_cv1 | natural | 3 | 2.30 | 2 | 3 | 3 | 0.574 | 0.269 | 0.062 | 1.000 |
| layer18_cv1 | balanced | 3 | 2.75 | 3 | 3 | 3 | 1.128 | 0.745 | 0.373 | 1.000 |
| layer18_output | natural | 3 | 2.19 | 2 | 3 | 3 | 0.557 | 0.190 | 0.059 | 1.000 |
| layer18_output | balanced | 3 | 2.85 | 3 | 3 | 3 | 0.748 | 0.555 | 0.334 | 1.000 |
| layer18_prefuse | natural | 3 | 2.23 | 2 | 3 | 3 | 0.690 | 0.283 | 0.066 | 1.000 |
| layer18_prefuse | balanced | 3 | 2.79 | 3 | 3 | 3 | 1.186 | 0.772 | 0.447 | 1.000 |
| layer21_bottleneck_last | natural | 3 | 1.39 | 1 | 2 | 3 | 0.840 | 0.068 | 0.011 | 1.000 |
| layer21_bottleneck_last | balanced | 3 | 2.38 | 3 | 3 | 3 | 0.767 | 0.220 | 0.169 | 1.000 |
| layer21_cv1 | natural | 3 | 2.43 | 2 | 3 | 3 | 1.018 | 0.539 | 0.154 | 1.000 |
| layer21_cv1 | balanced | 3 | 2.73 | 3 | 3 | 3 | 2.714 | 1.737 | 0.871 | 1.000 |
| layer21_output | natural | 3 | 2.27 | 2 | 3 | 3 | 1.069 | 0.443 | 0.119 | 1.000 |
| layer21_output | balanced | 3 | 2.84 | 3 | 3 | 3 | 1.865 | 1.359 | 0.793 | 1.000 |
| layer21_prefuse | natural | 3 | 2.35 | 2 | 3 | 3 | 1.263 | 0.596 | 0.162 | 1.000 |
| layer21_prefuse | balanced | 3 | 2.76 | 3 | 3 | 3 | 2.888 | 1.933 | 0.991 | 1.000 |

> **关键**: Sb 秩 ≤ K−1 = 3, 故 Fisher 有效秩必然很低 (≤3). 这只表示“类别均值差异位于 ≤3 维线性子空间”, **不等于** 原特征只需 3 个通道, 更不能等同于 C2f hidden 宽度.

## E. 逐通道任务重要性 (方差 vs Fisher)
| feature | mode | Spearman(var,F) | topk重合 | 低方差高Fisher数 |
|---|---|---|---|---|
| layer12_bottleneck_last | natural | -0.098 | 0.25 | 8 |
| layer12_bottleneck_last | balanced | -0.159 | 0.25 | 9 |
| layer12_cv1 | natural | +0.306 | 0.25 | 9 |
| layer12_cv1 | balanced | +0.309 | 0.17 | 10 |
| layer12_output | natural | -0.097 | 0.08 | 14 |
| layer12_output | balanced | -0.176 | 0.08 | 16 |
| layer12_prefuse | natural | +0.210 | 0.28 | 15 |
| layer12_prefuse | balanced | +0.200 | 0.17 | 16 |
| layer15_output | natural | -0.079 | 0.12 | 9 |
| layer15_output | balanced | -0.324 | 0.12 | 11 |
| layer18_bottleneck_last | natural | +0.096 | 0.38 | 6 |
| layer18_bottleneck_last | balanced | +0.046 | 0.38 | 6 |
| layer18_cv1 | natural | -0.100 | 0.08 | 12 |
| layer18_cv1 | balanced | -0.169 | 0.08 | 14 |
| layer18_output | natural | -0.366 | 0.08 | 18 |
| layer18_output | balanced | -0.283 | 0.08 | 19 |
| layer18_prefuse | natural | -0.053 | 0.22 | 19 |
| layer18_prefuse | balanced | -0.139 | 0.22 | 21 |
| layer21_bottleneck_last | natural | +0.073 | 0.00 | 11 |
| layer21_bottleneck_last | balanced | +0.070 | 0.00 | 11 |
| layer21_cv1 | natural | -0.007 | 0.12 | 31 |
| layer21_cv1 | balanced | -0.007 | 0.12 | 31 |
| layer21_output | natural | +0.054 | 0.19 | 30 |
| layer21_output | balanced | +0.055 | 0.22 | 30 |
| layer21_prefuse | natural | -0.039 | 0.12 | 58 |
| layer21_prefuse | balanced | -0.041 | 0.12 | 58 |

## F. 报告必答

### Q1. 全局高方差通道是否也是任务判别性最高的通道?
- 10 个特征 (balanced) 方差与 Fisher 的平均 Spearman = **-0.048**. 该值越接近 +1 表示“高方差≈高判别”. 实测见上表 (事实). 若 |ρ| 偏低, 说明方差排序与判别排序显著错位 → 高方差不等于高任务价值 (相关性).

### Q2. 是否存在低方差但高 Fisher 的通道?
- layer12_bottleneck_last: 9 个, 通道索引示例 [5, 11, 12, 19, 23, 41, 45, 46, 47] (事实).
- layer12_cv1: 10 个, 通道索引示例 [0, 27, 30, 38, 43, 62, 64, 67, 85, 95] (事实).
- layer12_output: 16 个, 通道索引示例 [10, 15, 21, 23, 24, 29, 31, 48, 49, 50, 60, 64]... (事实).
- layer12_prefuse: 16 个, 通道索引示例 [0, 26, 27, 40, 43, 62, 64, 101, 108, 115, 127, 131]... (事实).
- layer15_output: 11 个, 通道索引示例 [2, 8, 10, 14, 19, 23, 24, 27, 40, 53, 57] (事实).
- layer18_bottleneck_last: 6 个, 通道索引示例 [6, 8, 11, 18, 26, 38] (事实).
- layer18_cv1: 14 个, 通道索引示例 [2, 12, 16, 18, 26, 30, 32, 44, 48, 51, 65, 66]... (事实).
- layer18_output: 19 个, 通道索引示例 [1, 3, 5, 7, 9, 12, 17, 23, 28, 36, 39, 45]... (事实).
- layer18_prefuse: 21 个, 通道索引示例 [2, 16, 18, 30, 48, 51, 82, 98, 100, 102, 104, 107]... (事实).
- layer21_bottleneck_last: 11 个, 通道索引示例 [11, 18, 25, 26, 28, 31, 57, 67, 76, 95, 100] (事实).
- layer21_cv1: 31 个, 通道索引示例 [21, 24, 46, 49, 53, 83, 89, 103, 110, 112, 113, 126]... (事实).
- layer21_output: 30 个, 通道索引示例 [22, 25, 47, 48, 61, 66, 67, 70, 72, 74, 88, 89]... (事实).
- layer21_prefuse: 58 个, 通道索引示例 [24, 53, 83, 89, 103, 110, 112, 159, 164, 165, 172, 176]... (事实).

### Q3. Layer 12/18/21 中哪个位置的任务判别信息分布最广?
- 以 balanced 各类 effective_rank 之和衡量 (越高=判别信息越分散):

| feature | Σ_k eff_rank(bal) | Σ_k r95/C(bal) |
|---|---|---|
| layer12_bottleneck_last | 90.0 | 2.40 |
| layer12_prefuse | 217.8 | 2.25 |
| layer18_bottleneck_last | 95.3 | 2.38 |
| layer18_prefuse | 190.6 | 2.01 |
| layer21_bottleneck_last | 174.5 | 2.11 |
| layer21_prefuse | 364.7 | 1.83 |

### Q4. 哪个位置更适合结构化通道裁剪?
- 结构化裁剪适合“方差与 Fisher 高度一致且低方差高Fisher少”的特征 (排序一致→按方差裁剪不会误删判别通道).

| feature | Spearman(var,F) | 低方差高Fisher | 判别 |
|---|---|---|---|
| layer12_bottleneck_last | -0.159 | 9 | 风险高 (推测) |
| layer12_cv1 | +0.309 | 10 | 需谨慎 (推测) |
| layer12_output | -0.176 | 16 | 风险高 (推测) |
| layer12_prefuse | +0.200 | 16 | 风险高 (推测) |
| layer15_output | -0.324 | 11 | 风险高 (推测) |
| layer18_bottleneck_last | +0.046 | 6 | 风险高 (推测) |
| layer18_cv1 | -0.169 | 14 | 风险高 (推测) |
| layer18_output | -0.283 | 19 | 风险高 (推测) |
| layer18_prefuse | -0.139 | 21 | 风险高 (推测) |
| layer21_bottleneck_last | +0.070 | 11 | 风险高 (推测) |
| layer21_cv1 | -0.007 | 31 | 风险高 (推测) |
| layer21_output | +0.055 | 30 | 风险高 (推测) |
| layer21_prefuse | -0.041 | 58 | 风险高 (推测) |

### Q5. moisture 判别信息是否集中在少数通道?
- layer12_bottleneck_last: |moisture−bg| 均值差前 23 个通道即覆盖 80% 的类间差能量 (累计 0.82) (事实).
- layer12_cv1: |moisture−bg| 均值差前 48 个通道即覆盖 80% 的类间差能量 (累计 0.80) (事实).
- layer12_output: |moisture−bg| 均值差前 52 个通道即覆盖 80% 的类间差能量 (累计 0.81) (事实).
- layer12_prefuse: |moisture−bg| 均值差前 69 个通道即覆盖 80% 的类间差能量 (累计 0.80) (事实).
- layer15_output: |moisture−bg| 均值差前 33 个通道即覆盖 80% 的类间差能量 (累计 0.81) (事实).
- layer18_bottleneck_last: |moisture−bg| 均值差前 26 个通道即覆盖 80% 的类间差能量 (累计 0.81) (事实).
- layer18_cv1: |moisture−bg| 均值差前 47 个通道即覆盖 80% 的类间差能量 (累计 0.81) (事实).
- layer18_output: |moisture−bg| 均值差前 45 个通道即覆盖 80% 的类间差能量 (累计 0.80) (事实).
- layer18_prefuse: |moisture−bg| 均值差前 72 个通道即覆盖 80% 的类间差能量 (累计 0.80) (事实).
- layer21_bottleneck_last: |moisture−bg| 均值差前 64 个通道即覆盖 80% 的类间差能量 (累计 0.81) (事实).
- layer21_cv1: |moisture−bg| 均值差前 129 个通道即覆盖 80% 的类间差能量 (累计 0.80) (事实).
- layer21_output: |moisture−bg| 均值差前 123 个通道即覆盖 80% 的类间差能量 (累计 0.80) (事实).
- layer21_prefuse: |moisture−bg| 均值差前 187 个通道即覆盖 80% 的类间差能量 (累计 0.80) (事实).

### Q6. natural 与 balanced 差异多大?
- natural vs balanced 的 top-1 Fisher 特征值平均相对差异 = **0.900**. 差异越大 → 自然分布被背景/某类数量支配越严重 (相关性).

### Q7. 全局 PCA 可压缩性判断是否会损害少数类别?
- 对每个特征比较 “三类中最低的 r99/C” 与 “global r95/C”: 在 **13/13** 个特征上, 即便是能量最集中的类别, 其 r99/C 仍高于 global r95/C (差距最大: layer18_output, min_class_r99/C=0.90 vs global_r95/C=0.67). 这意味着按全局 PCA 裁到 r95 会落在“任一类 99% 能量所需通道数”之下 → 可能切掉少数类 r95→r99 的判别方向, **存在损害少数类的风险** (推测, 需训练验证).

### Q8. 谱结果能否解释 P5 压缩比 P4 更敏感?
- bottleneck_last 三类平均 r99/C: P4(L12/18)=0.96, P5(L21)=0.92 —— 两者**几乎持平**, 说明“逐类谱平坦度”本身并不能区分 P4/P5 (事实).
- 区分点在别处 (事实+推测): (1) SR1 裁剪深度不同 — P5 hidden 128→96 削掉 25% 通道, P4 48→40 仅削 17%; 全局谱分析测得该裁剪 P5 保留 96.1% / P4 保留 96.7% 能量, P5 略丢更多. (2) 判别通道脆弱性 — P5 bottleneck 方差-Fisher Spearman=+0.07 (P4=-0.06), 低方差高Fisher 通道 P5=11 vs P4=8 → P5 有更多“低方差却具判别力”的通道, 方差/PCA 式裁剪更易误删.
- 综上: 谱结果**能**解释 “P5 压缩更敏感”, 但主要源于 **裁剪更深 + 判别通道更脆弱**, 而非逐类谱更平坦 (与既有实验 L1-96 0.429 / SR1 0.412 / SR1-P4Only 0.427 方向一致; 推测, 需训练验证; 与全局谱分析 [[l196-spectrum-analysis]] 互证).

## G. 结论分级 (实测事实 / 统计相关性 / 合理推测 / 尚未验证)
- **实测事实**: 类条件 r95/C、Fisher top-3 特征值、Spearman 系数、低方差高Fisher通道数 (均来自 1105 val 图实测 Gram).
- **统计相关性**: “高方差↔高判别” 由 Spearman 量化; “natural/balanced 差异” 由特征值相对差异量化.
- **合理推测**: Q4 的“较安全/需谨慎/风险高”判别、Q7 的“会损害少数类”、Q8 的“P5 更敏感” — 均为基于谱数据的推断.
- **尚未验证的压缩建议**: 具体通道裁剪/hidden 调整方案需经训练验证; 本脚本不修改 YAML, 不自动训练.
- **约束**: Fisher 有效秩 (≤3) 不等于 C2f hidden 宽度, 也不等于所需通道数.
