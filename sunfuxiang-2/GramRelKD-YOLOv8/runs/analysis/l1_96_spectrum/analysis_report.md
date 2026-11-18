# L1-96 多层特征谱分析报告
- 生成时间(脚本运行): 运行耗时 4.49s
- Ultralytics 8.4.84 / PyTorch 2.4.1+cu121 / CUDA 12.1 / GPU NVIDIA GeForce RTX 4090

## A. 实际分析的模型与数据配置
- 权重: `/data/cle/sunfuxiang-2/ultralytics-main/runs/detect/runs/detect/hongkong_L1-96/weights/best.pt`
- 数据 YAML: `/data/cle/sunfuxiang-2/ultralytics-main/runs/analysis/l1_96_spectrum/_auto_data.yaml` (auto-generated (no yaml found in data-root))
- check_det_dataset 校验: `ok`
- split: `val`, imgsz=640, batch=8, workers=4, device=`cuda:0`
- 预处理: 复用 Ultralytics `LetterBox(new_shape=(640,640), auto=False, stride=32)`, BGR→RGB, /255 归一化, 无增强 (与 val 一致).
- nc=3, stride=32
- max_spatial_samples=0 (0=使用全部空间位置)

## B. 实际分析图像数量
- 该 split 全部图像数: **1105**
- 本次分析图像数: **500** (seed=42, 路径排序后固定抽样)

## C. 每个层的通道数与内部 hidden 值
| Layer | 类型 | 输出通道 | hidden_c | bottleneck数 | 期望输出 | 是否匹配 |
|---|---|---|---|---|---|---|
| 4 | C2f | 64 | 32 | 2 | 64 | ✓ |
| 6 | C2f | 96 | 48 | 2 | 96 | ✓ |
| 8 | C2f | 256 | 128 | 1 | 256 | ✓ |
| 12 | C2f | 96 | 48 | 1 | 96 | ✓ |
| 15 | C2f | 64 | 32 | 1 | 64 | ✓ |
| 18 | C2f | 96 | 48 | 1 | 96 | ✓ |
| 21 | C2f | 256 | 128 | 1 | 256 | ✓ |

重点 C2f self.c 校验:

| Layer | 实际 self.c | 预期 self.c | 是否匹配 |
|---|---|---|---|
| 12 | 48 | 48 | ✓ |
| 18 | 48 | 48 | ✓ |
| 21 | 128 | 128 | ✓ |

## D. 每层 r90 / r95 / r99 与有效秩指标 (raw 谱)
| Feature | C | count | r90 | r95 | r99 | r95/C | StableRank | EffRank | PartRatio |
|---|---|---|---|---|---|---|---|---|---|
| layer4_output | 64 | 3200000 | 35 | 45 | 59 | 0.703 | 6.3 | 30.8 | 17.7 |
| layer6_output | 96 | 800000 | 50 | 67 | 88 | 0.698 | 8.3 | 43.6 | 24.6 |
| layer8_output | 256 | 200000 | 112 | 163 | 231 | 0.637 | 8.6 | 77.0 | 31.7 |
| layer12_output | 96 | 800000 | 52 | 68 | 88 | 0.708 | 6.9 | 44.2 | 23.0 |
| layer15_output | 64 | 3200000 | 34 | 44 | 58 | 0.688 | 5.8 | 31.4 | 17.8 |
| layer18_output | 96 | 800000 | 45 | 61 | 86 | 0.635 | 7.3 | 40.0 | 22.0 |
| layer21_output | 256 | 200000 | 106 | 157 | 228 | 0.613 | 6.2 | 72.7 | 25.2 |
| layer12_cv1 | 96 | 800000 | 58 | 73 | 90 | 0.760 | 6.5 | 47.3 | 23.2 |
| layer12_bottleneck_last | 48 | 800000 | 30 | 37 | 46 | 0.771 | 6.0 | 27.3 | 16.8 |
| layer12_prefuse | 144 | 800000 | 81 | 105 | 134 | 0.729 | 6.8 | 61.9 | 26.3 |
| layer18_cv1 | 96 | 800000 | 48 | 65 | 87 | 0.677 | 7.0 | 41.1 | 22.0 |
| layer18_bottleneck_last | 48 | 800000 | 30 | 37 | 45 | 0.771 | 6.0 | 28.4 | 17.4 |
| layer18_prefuse | 144 | 800000 | 71 | 95 | 129 | 0.660 | 7.6 | 57.8 | 27.0 |
| layer21_cv1 | 256 | 200000 | 120 | 168 | 232 | 0.656 | 7.3 | 84.3 | 30.5 |
| layer21_bottleneck_last | 128 | 200000 | 67 | 90 | 118 | 0.703 | 6.6 | 54.0 | 24.7 |
| layer21_prefuse | 384 | 200000 | 173 | 247 | 345 | 0.643 | 7.8 | 114.0 | 36.5 |

### Centered 谱指标 (供对照)

| Feature | C | r95_cen | r99_cen | r95/C_cen | StableRank_cen | EffRank_cen |
|---|---|---|---|---|---|---|
| layer4_output | 64 | 47 | 60 | 0.734 | 7.3 | 33.8 |
| layer6_output | 96 | 69 | 89 | 0.719 | 10.6 | 47.6 |
| layer8_output | 256 | 170 | 233 | 0.664 | 11.8 | 90.8 |
| layer12_output | 96 | 70 | 89 | 0.729 | 11.5 | 51.9 |
| layer15_output | 64 | 46 | 59 | 0.719 | 10.9 | 37.3 |
| layer18_output | 96 | 64 | 87 | 0.667 | 9.7 | 44.5 |
| layer21_output | 256 | 166 | 231 | 0.648 | 12.5 | 93.5 |
| layer12_cv1 | 96 | 75 | 91 | 0.781 | 12.2 | 57.6 |
| layer12_bottleneck_last | 48 | 38 | 46 | 0.792 | 9.0 | 31.4 |
| layer12_prefuse | 144 | 109 | 135 | 0.757 | 12.7 | 77.8 |
| layer18_cv1 | 96 | 67 | 88 | 0.698 | 10.1 | 45.4 |
| layer18_bottleneck_last | 48 | 38 | 46 | 0.792 | 8.3 | 32.7 |
| layer18_prefuse | 144 | 99 | 131 | 0.688 | 11.2 | 67.7 |
| layer21_cv1 | 256 | 176 | 235 | 0.688 | 12.5 | 105.1 |
| layer21_bottleneck_last | 128 | 93 | 119 | 0.727 | 12.6 | 65.0 |
| layer21_prefuse | 384 | 258 | 349 | 0.672 | 15.1 | 145.6 |

## E. Backbone P4 (L6) vs Neck P4 (L12) 谱结构比较
- L6 (Backbone P4): C=96, r95_raw=67, r95/C=0.70, StableRank=8.28, EffRank=43.59
- L12 (Top-down Neck P4): C=96, r95_raw=68, r95/C=0.71, StableRank=6.86, EffRank=44.25
- 解读见末尾 I 节.

## F. Detect P3 / P4 / P5 谱结构比较
- layer15_output: C=64, r95_raw=44, r95/C=0.69, StableRank=5.84, EffRank=31.44
- layer18_output: C=96, r95_raw=61, r95/C=0.64, StableRank=7.34, EffRank=39.95
- layer21_output: C=256, r95_raw=157, r95/C=0.61, StableRank=6.23, EffRank=72.73

## G. Layer 12 / 18 / 21 内部特征谱比较

### Layer 12 (C2f, self.c=48)

| 子特征 | C | r95_raw | r99_raw | r95/C | StableRank | EffRank |
|---|---|---|---|---|---|---|
| cv1 | 96 | 73 | 90 | 0.760 | 6.55 | 47.30 |
| bottleneck_last | 48 | 37 | 46 | 0.771 | 5.99 | 27.34 |
| prefuse | 144 | 105 | 134 | 0.729 | 6.78 | 61.92 |
| output | 96 | 68 | 88 | 0.708 | 6.86 | 44.25 |

### Layer 18 (C2f, self.c=48)

| 子特征 | C | r95_raw | r99_raw | r95/C | StableRank | EffRank |
|---|---|---|---|---|---|---|
| cv1 | 96 | 65 | 87 | 0.677 | 7.04 | 41.09 |
| bottleneck_last | 48 | 37 | 45 | 0.771 | 5.98 | 28.37 |
| prefuse | 144 | 95 | 129 | 0.660 | 7.56 | 57.84 |
| output | 96 | 61 | 86 | 0.635 | 7.34 | 39.95 |

### Layer 21 (C2f, self.c=128)

| 子特征 | C | r95_raw | r99_raw | r95/C | StableRank | EffRank |
|---|---|---|---|---|---|---|
| cv1 | 256 | 168 | 232 | 0.656 | 7.31 | 84.29 |
| bottleneck_last | 128 | 90 | 118 | 0.703 | 6.60 | 54.03 |
| prefuse | 384 | 247 | 345 | 0.643 | 7.83 | 113.97 |
| output | 256 | 157 | 228 | 0.613 | 6.23 | 72.73 |

## H. 谱衰减快慢排序 (raw 谱, 按 r95/C 升序)
说明: r95/C 越小 ⇒ 能量越集中 ⇒ 谱衰减越快/越可压缩; 越大 ⇒ 维度利用越充分/冗余越低. 本模型所有 r95/C 均落在 0.61–0.77, 总体偏平坦, 故用三分位 (相对本模型内部) 标注.

| 排名 | Feature | C | r95_raw | r95/C | StableRank | 相对分类 |
|---|---|---|---|---|---|---|
| 1 | layer21_output | 256 | 157 | 0.613 | 6.2 | 相对衰减快/较可压缩 |
| 2 | layer18_output | 96 | 61 | 0.635 | 7.3 | 相对衰减快/较可压缩 |
| 3 | layer8_output | 256 | 163 | 0.637 | 8.6 | 相对衰减快/较可压缩 |
| 4 | layer21_prefuse | 384 | 247 | 0.643 | 7.8 | 相对衰减快/较可压缩 |
| 5 | layer21_cv1 | 256 | 168 | 0.656 | 7.3 | 相对衰减快/较可压缩 |
| 6 | layer18_prefuse | 144 | 95 | 0.660 | 7.6 | 中等 |
| 7 | layer18_cv1 | 96 | 65 | 0.677 | 7.0 | 中等 |
| 8 | layer15_output | 64 | 44 | 0.688 | 5.8 | 中等 |
| 9 | layer6_output | 96 | 67 | 0.698 | 8.3 | 中等 |
| 10 | layer4_output | 64 | 45 | 0.703 | 6.3 | 中等 |
| 11 | layer21_bottleneck_last | 128 | 90 | 0.703 | 6.6 | 中等 |
| 12 | layer12_output | 96 | 68 | 0.708 | 6.9 | 中等 |
| 13 | layer12_prefuse | 144 | 105 | 0.729 | 6.8 | 相对衰减慢/维度利用高 |
| 14 | layer12_cv1 | 96 | 73 | 0.760 | 6.5 | 相对衰减慢/维度利用高 |
| 15 | layer12_bottleneck_last | 48 | 37 | 0.771 | 6.0 | 相对衰减慢/维度利用高 |
| 16 | layer18_bottleneck_last | 48 | 37 | 0.771 | 6.0 | 相对衰减慢/维度利用高 |

关键观察: 没有任何层的 r95/C ≤ 0.5, 即不存在“急剧低秩/高度冗余”的层; 最可压缩的是 P5 系列 (layer8/21_output, layer21_prefuse, r95/C≈0.61–0.64), 最不可压缩的是 Layer12/18 的 bottleneck_last (r95/C≈0.77, hidden 几乎被用满).

## I. 与既有实验结论的一致性判断
### 0) 总体结论: 谱普遍平坦, 无急剧冗余层
- 全部 16 个特征的 raw r95/C 范围 = [0.613, 0.771], 均值 0.691. 没有任何层 r95/C≤0.5, 即不存在“维度严重过剩”的层; 信息在通道间分布较广 → 任何层的激进均匀裁剪都有风险.

### 1) P4 hidden 压缩敏感 (Layer 12 / 18, self.c=48)
- Layer 12 bottleneck_last: r95=37, r99=46 (hidden=48). SR1 把 hidden 48→40: 40 落在 r95(37) 与 r99(46) 之间, 裁剪到 40 维仅保留 bottleneck 谱能量的 96.7% (即丢掉 3.3% 的尾部方差, 主要是 95%→99% 这段方向) → 与 SR1 (48→40) test mAP 0.429→0.412 的下降一致: **P4 hidden 压缩敏感**.
- Layer 18 bottleneck_last: r95=37, r99=45 (hidden=48). SR1 把 hidden 48→40: 40 落在 r95(37) 与 r99(45) 之间, 裁剪到 40 维仅保留 bottleneck 谱能量的 96.9% (即丢掉 3.1% 的尾部方差, 主要是 95%→99% 这段方向) → 与 SR1 (48→40) test mAP 0.429→0.412 的下降一致: **P4 hidden 压缩敏感**.

### 2) Neck P4 (L12) 是否存在轻度冗余?
- L12 prefuse (concat 后, cv2 前): C=144, r95/C=0.729; cv2 输出: C=96, r95/C=0.708.
- 1×1 cv2 把 144→96 压缩 (48 维), 两端 r95/C 相近 (≈0.71–0.73), 说明 cv2 已能无损吸收 concat 带来的维度. 这与 SR1-P4Only (只压 hidden、不动 L21 输出 128) 时 mAP 仅 0.429→0.427 微降一致: **Neck P4 仅存在轻度/可被 cv2 吸收的冗余, 但 hidden 本身不可激进压缩**.

### 3) Layer 21 / P5 (hidden) 压缩敏感 (self.c=128)
- Layer 21 bottleneck_last: r95=90, r99=118 (hidden=128). SR1 把 hidden 128→96: 96 仅略高于 r95(90), 裁剪到 96 维仅保留 bottleneck 谱能量的 96.1% (即丢掉 3.9%, 覆盖 r95→r99(90→118) 及之后 32 维的大部分方差) → 与 SR1 (128→96) test mAP 0.429→0.412 的“明显下降”一致: **L21/P5 hidden 压缩敏感**, 且比 P4 丢失能量更多.

### 4) Backbone P4 (L6) 压缩敏感?
- L6 (Backbone P4) r95/C=0.698, StableRank=8.3, EffRank=43.6. 其谱同样偏平坦 (信息铺满 96 通道), 与 Neck P4 (L12) 谱形相近, 说明 P4 语义信息在通道间分布广 → 对通道裁剪整体敏感, 与上述 hidden 压缩敏感的结论方向一致.

### 重要约束说明
- **不能把 output 的 r95 直接当作 C2f 应设的 hidden 通道数.** hidden 维度与最终输出特征的有效秩不是同一个量. 应综合参考同一 C2f 的 cv1 / bottleneck_last / prefuse 三个内部特征谱的一致性.
- 下面给出基于内部谱一致性的 **保守压缩候选区间 (实验建议, 非数学证明)**:
  - Layer 12: 当前 hidden=48; bottleneck_last r95=37, r99=46. 弱参考区间 hidden ≈ [37, 46] (仅供实验探索, 需以训练验证为准).
  - Layer 18: 当前 hidden=48; bottleneck_last r95=37, r99=45. 弱参考区间 hidden ≈ [37, 45] (仅供实验探索, 需以训练验证为准).
  - Layer 21: 当前 hidden=128; bottleneck_last r95=90, r99=118. 弱参考区间 hidden ≈ [90, 118] (仅供实验探索, 需以训练验证为准).
- 本脚本不会自动修改任何 YAML, 不会自动开始训练.
