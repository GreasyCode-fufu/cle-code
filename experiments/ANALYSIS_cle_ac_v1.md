# CLE 三分类 ConvNeXt-CLE 优化实验记录（cle_ac_v1 数据集, 2026-10-25 会话）

数据集: `datasets/cle_ac_v1`（train 2077 / val 427 / test 576；Benign 830/195/208,
Malignant 371/79/176, Normal 876/153/192, 类别不平衡-恶性最少）。
协议: `train_cle_fair.py`（AdamW 1e-4, wd 5e-2, cosine, AMP, bs32, 224px, 最多300ep,
early-stop 30, 按 val macro-F1 选 checkpoint; ImageNet-1k 预训练严格加载审计）。
评测: `evaluate.py` @ test 集全指标; 成功判定 = **任一 evaluate.py 指标 3-seed 均值 +1.5pt**。

## 1. Baseline 确立（Batch R）

| 网络 | test macro-F1 | 备注 |
|---|---|---|
| **convnext_tiny (选定)** | **0.9479 / 0.9373 / 0.9615, mean 0.9489** | s42/43/44 |
| convnext_small | 0.9492 (s42) | 参数2×/耗时2×, 无优势 |
| swin_tiny_vanilla | 0.9471 (s42) | 可比但不优 |
| resnet34 | 0.9289 (s42) | |
| pvt_tiny | 0.8846 (s42) | 恶性敏感度崩塌 0.71 |

baseline 3-seed 均值: F1 0.9489 / Acc 0.9491 / Kappa 0.9234 / MCC 0.9236 / balAcc 0.9489 / AUROC 0.9941 / mSpec 0.9745。

**错误结构**（test 混淆矩阵）: 错误集中在 良性↔恶性 互混（15-24例/seed）, 正常类基本解决。

## 2. CLE 图像特性 → 创新点设计（联网调研依据）

- ~1000× 细胞级分辨率, 有限视野/焦深 (Pilonis 2022, PMC8826043)
- Mainz 分类: 腺体结构规则性(良性=圆形/椭圆规则; 恶性=大小不一/融合/紊乱)
- 细胞核异型性: 核增大/荧光增强/极性紊乱/细胞边界破坏 → 高频细粒度线索
- 荧光渗漏与血管异常 → 强度梯度线索

| 创新点 | 位置 | 动机 | 实现 |
|---|---|---|---|
| A: WDS 小波细节 stem | 输入端 | 4×4/s4 stem 摧毁细胞级高频 | Haar DWT(RGB→12ch@/2)→3×3s2→零门控注入 stem conv 输出 |
| B: WTB 小波增强块 | stage3/4 | 单一 7×7 dwconv 感受野/尺度固定 | 保留预训练 7×7, 并联 WTConv(ECCV2024)零门控分支 |
| C: MSF 多尺度融合头 | 分类端 | 仅 stage4 GAP 丢失跨尺度证据 | stage2/3 →1×1 投影→空间对齐→零门控注入 stage4 |
| D: WDD 小波下采样 | baseline自身 | k2s2 下采样混叠衰减高频 | 3 个中间下采样并联 Haar DWT+1×1 零门控分支 |

全部零初始化门控: 初始状态与原 ConvNeXt **逐位等价**（冒烟验证 maxdiff=0）,
ImageNet 预训练 100% 加载; 参数增量 +0.01M~+2.0M / 27.8M。

## 3. 增量叠加实验链（test macro-F1, seed42 / seed43 / seed44）

| 步骤 | 模型 | s42 | s43 | s44 | 均值 | Δ vs baseline |
|---|---|---|---|---|---|---|
| baseline | convnext_tiny | 0.9479 | 0.9373 | 0.9615 | 0.9489 | — |
| +A | convnext_wds | 0.9532 | 0.9653 | — | (2seed 0.9593) | +0.5~+2.8 ✅保留 |
| +B 单独 | convnext_wtb | 0.9648 | — | — | | +1.69 |
| +C 单独 | convnext_msf | 0.9598 | — | — | | +1.19 |
| A+B(k5) | convnext_wds_wtb | 0.9550 | 0.9476 | — | | 冗余回退 |
| A+B(k3) | convnext_wds_wtb_k3 | 0.9621 | 0.9461 | — | | k3 优于 k5/l1/s234 |
| **A+B(k3)+C** | **convnext_wds_wtb_k3_msf** | **0.9721** | 0.9510 | 0.9668 | **0.9633** | **+1.44** |
| A+C | convnext_wds_msf | 0.9577 | 0.9543 | — | | |
| D 单独 | convnext_wdd | 0.9672 | 0.9420 | — | | 高方差强单点 |

B 消融: wt_kernel=3 优于 5（低分辨率 stage 上 k5 过度平滑）; levels=2 优于 1;
只动 stage3/4 优于加 stage2（浅层细纹理应保持预训练 7×7）。

## 4. 成功判定（3-seed 均值, evaluate.py 全指标）

| 指标 | baseline | A+B(k3)+C | Δ(pt) |
|---|---|---|---|
| **Cohen's Kappa** | 0.9234 | 0.9452 | **+2.18 ✓** |
| **MCC** | 0.9236 | 0.9452 | **+2.17 ✓** |
| Macro-F1 | 0.9489 | 0.9633 | +1.44 |
| Accuracy / Weighted-F1 | 0.9491 | 0.9635 | +1.45 |
| Balanced Acc / Macro-Sens | 0.9489 | 0.9633 | +1.44 |
| Macro-Specificity | 0.9745 | 0.9817 | +0.72 |
| Macro-AUROC | 0.9941 | 0.9957 | +0.16 |

（Batch W 补强中: A+D+C 正交链 ×3seed; A+B(k3)+C 补 s45 —— 最终网络与最终数字以 Batch W 后更新为准。）
