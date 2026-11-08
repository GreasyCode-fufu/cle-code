# R5 终版报告 — CLE-Net v2 论文方案（2026-11-08）

## 1. 最终网络: convnext_v6f = ConvNeXt-T + MSP + DCH + FGS（三创新点）

**主域成绩（cle_ac_v1 test, 3 seeds 42/43/44, 同批同协议）**

| 指标 | baseline | v6f | Δ |
|---|---|---|---|
| Macro-F1 | 94.89±0.99 | **96.29±0.23** | **+1.40** |
| Cohen's Kappa | 92.34±1.50 | **94.43±0.32** | **+2.09** |
| MCC | 92.36±1.49 | **94.46±0.30** | **+2.10** |
| Balanced Acc | 94.89±1.03 | 96.33±0.18 | +1.44 |
| Macro-Specificity | 97.45±0.51 | 98.15±0.10 | +0.70 |
| Macro-AUROC | 99.41±0.17 | 99.59±0.08 | +0.18 |
| **恶性敏感度** | 93.37±2.14 | **95.83±1.07** | **+2.46** |

## 2. 三个创新点与消融链（每步 3 seeds 同批）

| 配置 | F1 | ΔF1 | 恶性sens | κ |
|---|---|---|---|---|
| ConvNeXt-T baseline | 94.89±0.99 | — | 93.37 | 92.34 |
| **+创新1 MSP**（v2m: 多尺度证据投影, 投影后求和头） | 95.69±0.09 | **+0.80** | 95.08 | 93.57 |
| **+创新2 DCH**（v2c: 稠密跨尺度拼接头, 不可旁路） | 96.38±0.32 | **+0.69** | 94.51 | 94.60 |
| **+创新3 FGS**（v6f 最终: 梯度增广茎干, 零init等价） | 96.29±0.23 | -0.10 | **95.83** | 94.43 |

- **MSP (Multi-Scale evidence Projection)**: stage-2/3 特征经无偏置 1×1 投影到 768 维
  语义空间 + 各尺度 LN——解决"仅 stage-4 特征被分类器消费"。单独 +0.80。
- **DCH (Dense Cross-scale Head)**: 三尺度池化向量**拼接** + 稠密 Linear(2304,K)——
  结构上强制每个 logit 消费全部三尺度证据（不可旁路）。设计动机有判决性实验支撑:
  前作 V1 门控并联注入训练后门控范数 ~10⁻³、置零后 F1 仅 ±0.35pt（可旁路, 表观
  提升不可归因, reports/06-08）。+0.69。
- **FGS (Foreground/Gradient-augmented Stem)**: 输入增广第 4 通道 = Sobel 梯度幅值
  （z-score 归一）, 预训练 stem 权重膨胀、新通道零初始化——**初始与预训练逐位等价
  （max|Δstem|=0.00e+00 实测）**, 严格加载审计通过。动机: 暗背景 25-31% + 腺体边缘
  规则性是 B/M 判别核心。F1 让 0.10 换恶性敏感度 +1.32（漏诊大降, 临床价值）。

## 3. 搜索过程的负结果证据（22 个候选, 全部 2+ seeds, 诚实归档）

结构: SE@4 / GRN@34 / GRN@4 / TeLU / XRM(0.005/0.01/0.02) / CSAP(硬/软) / PFESA /
SDGW / DRC / WR全深度 / SOP二阶头 / GRN+XRM / XRM+cw / FGS+XRM / CSAP+FGS / CSM(4规模)。
训练: ls0.05 / cw1.3-1.5 / focalγ2 / ACM m0.1-0.2 / EMA0.999 / 跨seed soup。
结论: v2c(96.38) 为该协议下强局部最优; 所有扰动均值回归至 95.2-96.4 噪声带。
（唯一例外: FGS 以 sens_M +2.1 的一致改善入选, 见上表。）

## 4. 复杂度

- v6f params ≈ 28.27M（+0.45M, +1.6%）/ MACs ≈ 4.63G（thop, 不计 Sobel 通道 conv 的
  少量增量）; 与 baseline 同量级。
- thop 口径注意: FGS 新通道使 stem MACs +33%（stem 本身极小, 网络级 <1%）。

## 5. 泛化验证（R5 链进行中, 结果待补）

- v6f × 3 seeds × {BUSI, heelBone, covid, CUB, TCGA}; 小域 baseline 复用同日 r3 批
  tiny（同管线同批, 确定性已验证）; TCGA 双模型同批。
- 历史（旧批）参考: CH 在 heelBone +0.90 居首 / CIFAR 居首——预期 v6f 主导增益来自
  CH, FGS 为中性/小正。

## 6. 交付物索引

- 网络: models/optimize/convnext_cle_v5.py（ConvNeXtV5: v6f/v2m 工厂）
- 结果: experiments/results_cle_ac_v1_c.csv（CA..CP 批全记录）
- 权重: weights_cle_ac_v1_c/{convnext_v2m,convnext_v6f}_seed*/（含 run_meta 审计）
- 报告: experiments/R3_REPORT.md(CSM轮) / R4_REPORT.md(16候选证伪) / 本文件
- git: 逐批提交, 全可追溯

## 5a. 泛化结果（R5 链, v6f vs 同批 tiny, 3 seeds Macro-F1 均值）

| 域 | tiny | **v6f** | Δ | 备注 |
|---|---|---|---|---|
| heelBone(跟骨X光) | 87.95 | **88.74** | **+0.79** | ✓ |
| covid_v1 | 98.42 | **98.63** | **+0.21** | ✓(FGS 把 v2c 的 -0.03 拉正) |
| **CUB-200(细粒度)** | 77.98 | **80.53** | **+2.55** | ✓✓ 全家族最高(超历史最强 ConvNeXt-S 79.98) |
| BUSI(乳腺超声) | 87.28 | 86.80 | -0.48 | 小样本域(546张), 与 v2c(-0.36)相当 |
| TCGA(病理6类) | 运行中 | 运行中 | — | 过夜, ~17:00 |

逐seed: heel 89.13/91.28/85.81; covid 98.88/98.49/98.51; CUB 81.37/79.02/81.20; BUSI 86.29/87.20/86.90。
