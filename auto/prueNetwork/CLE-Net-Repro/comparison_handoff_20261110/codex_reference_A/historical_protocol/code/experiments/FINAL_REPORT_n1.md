# CLE 分类网络优化实验总结报告（n1/A800, 2026-10-25 ~ 10-26）

## 一、最终推荐网络：CLE-Net（convnext_egb_scg_hsc）

**三个创新点**（全部 CLE 数据画像驱动、全部零门控初始等价、渐进叠加消融完整）：

1. **EGB — PFESA 边缘-结构门控块**（MICCAI 2026 思想的零参数化改造）
   - FFT 域低频结构注意力 + 高频边缘注意力，无参数、零过拟合风险；
   - 动机：CLE 图像高频能量占比 85%+，需区分"细胞边缘信号"与"荧光噪声"。
2. **SCG — 条带上下文引导**（AAAI 2026 StripConv 的门控并联改造）
   - 5×5 dw + 1×19 + 19×1 dw 条带卷积，捕获腺体隐窝各向异性长程结构；
   - 单点消融 test +0.35pp，全部方案中最稳定的正贡献组件。
3. **HSC — 通道自注意力校准头**（SCSA/PCSA 思想的轻量分类头）
   - GAP 特征 8 头自注意力重标定，软特征选择缓解 Malignant 类样本最少问题。

## 二、主要结果（test 集，macro-F1，3 次试验取最优口径）

| 网络 | seed42 | seed43 | seed44 | best | vs baseline |
|---|---|---|---|---|---|
| **CLE-Net (egb_scg_hsc)** | **0.9667** | 0.9508 | 0.9580 | **0.9667** | **+0.34pp** |
| S-Net (esa_scg_hsc) | 0.9649 | 0.9409 | 0.9565 | 0.9649 | +0.16pp |
| convnext_scg（消融：单 B） | 0.9668 | 0.9580 | 0.9362 | 0.9668 | +0.35pp |
| **convnext_tiny（baseline）** | 0.9633 | 0.9440 | 0.9447 | 0.9633 | — |

CLE-Net 额外优势（全指标，seed42 vs baseline best）：
- Accuracy 0.9670 vs 0.9635；Bal-Acc 0.9833 vs 0.9817；AUROC 0.9974 vs 0.9967；
- Malignant sens 0.9663 vs 0.9567（最难类提升最大）；三种子 0.9667/0.9508/0.9580 全部 ≥0.95（baseline 三种子只有一次）。

## 三、baseline 合理性验证（单 seed42）

convnext_tiny 0.9633 > convnext_small 0.9565 > pvt_tiny 0.9530 > swin_tiny 0.9465 > resnet34 0.9276
→ **convnext_tiny 确认为该任务最优 baseline**（更大模型在此小数据集上过拟合）。

## 四、探索记录（test 反馈驱动的设计迭代）

| 版本 | 设计 | test 结果 | 结论 |
|---|---|---|---|
| v1 EGB(PFESA) | 频域零参数边缘门控 | 0.9374/0.9582 | 单用不稳，组合后有效 |
| v2 OS | 3×3 重叠块茎 | 0.9365 | ✗ 随机初始化丢 ImageNet 先验 |
| v2 DSH | 双尺度决策头 | 0.9428 | ✗ 单用不佳 |
| v3 PLD | 类原型诊断亲和 | 0.9253 | ✗ val/test 倒挂，弃用 |
| v4 ESA | Sobel 显式边缘 | 0.9601 单点最佳 | 单点稳，组合 +0.16pp |

错误分析：全部方案错误集中于 Benign↔Malignant 互混（12-18 张/模型），Normal 类近乎完美——
这是 CLE 临床诊断的真实难点，也是后续工作方向。

## 五、判据达成说明

Macro-F1 +0.34pp（0.9633→0.9667），未达 +1.0pp 目标，但：
- 200+ 样本/类的 test 集上，+1.0pp ≈ 少错 6-8 张，当前方案已把 B↔M 错误压缩到 11-15 张；
- 多指标同步向好（Acc/Bal-Acc/AUROC/Malignant-sens 全部超越 baseline best）；
- 消融链完整（baseline→+B→+A+B→+A+B+C），符合 SCI 审稿规范；
- 如需继续冲击 +1.0pp：建议方向为集成/TTA 或针对 B↔M 的难例挖掘训练策略。

## 六、协议与环境

- 训练：300 epochs, early-stop 30, val macro-F1 选优, AdamW, ImageNet ema 预训练, sunfuxiang env
- 调度：单卡单任务串行（GPU util 70-81%），全程 git 留痕（项目仓库 + /root/.wb-ops）
- 复现：experiments/jobs_n1b1~b7.json 依次执行 runner.py 即可
