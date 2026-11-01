# EPNet 设计文档 —— 数据驱动的三创新点（2026-11-01 会话）

前作: SF-ConvNeXt (convnext_v2bc=WR+CH, F1 0.9600/3s, Kappa+1.83 vs baseline 0.9476/5s)。
本作目标: 新网络 vs baseline ≥+0.5pt(任一evaluate.py指标) 且均值不低于 0.9642(前作4s)。

## 0. 数据驱动依据 (experiments/cle_image_analysis.py, 2026-11-01)

train 120张/类 + v2bc-seed42 test 错误样本取证:

| 特征 | B | M | N | M→B漏诊(8) | M正确(40) | B→M过诊 |
|---|---|---|---|---|---|---|
| fft_hi 高频占比 | **0.0112** | 0.0103 | 0.0096 | **0.0086** | 0.0101 | 0.0117 |
| haar_det 细节能量 | 0.0049 | 0.0051 | 0.0049 | 0.0046 | 0.0049 | 0.0049 |
| 局部std max/mean | 5.70 | 5.51 | 5.15 | 5.76 | 5.45 | 5.60 |
| 亮度 | 0.2677 | 0.2331 | 0.2398 | 0.2476 | 0.2496 | 0.2947 |

结论:
1. CLE 为纯灰度(RGB三通道冗余=0) — 频域分析/单通道统计完全成立。
2. **M→B 漏诊样本高频占比低 15%** (0.0086 vs 0.0101): 恶性核异型的高频线索被抑制/淹没,
   需自适应增强 — 但不能盲目: 良性本身高频最多(B 0.0112>M 0.0103), B→M 过诊样本恰是
   高频最多的良性(0.0117)。=> 需要"结构条件下的频率重标定"(创新点B 双路交互门控)。
3. **局部异质性显著且类间有序** (max/mean 5.15~5.70): GAP 之外的 max 证据真实存在,
   GMP 携带独立判别信息(创新点A)。
4. 前作混淆矩阵: 恶性→良性 8 例是主要错误源(恶性sens 0.949-0.960)。

## 1. 三创新点(结构化替换, 计算图不可旁路; 前作V1零门控注入被wd压死的教训)

### A: DPH — Dual-Pooling Multi-scale Head (决策端, 前作CH的升级)
   每尺度 s2/s3/s4 取 [GAP; GMP] 投影拼接 -> 6*768=4608-d -> Linear(num_classes)。
   GAP=全局统计证据, GMP=最显著局部异型证据(多灶肿瘤"最差病灶定级"的临床原则)。
   不可旁路: head 输入维度结构改变, 强制消费双池化证据。
   预训练兼容: head 本就新初始化; 投影层同前作CH。

### B: FESR — Frequency Edge-Structure Recalibration (深层, stage3/4 每块 dwconv 后)
   FFT -> 高斯掩码(u=base_ratio*min/max边比)分离 低频=结构S / 高频=边缘E
   -> 各路空间统计归一化(PFESA式: (x-mean)/var -> sigmoid)
   -> **交互门控 g = sigmoid(w1*S_map + w2*E_map)** (可学习2标量, 无零初始化)
   -> x' = x * (1 + g)  (重标定而非替换, 初始即非恒等)
   与PFESA(分割跳跃连接, 两路独立相加)的区别: 分类block内 + 交互门控建模
   "规则结构下的边缘(良性腺缘) vs 紊乱结构下的边缘(恶性异型)"的条件频率解释。
   参数量: ~2 标量/块。不可旁路: 乘性重标定且无wd可压门控参数。

### C: LSDG — Local-Statistics Direction-Gated conv (浅层 stage2 每块 dwconv 后)
   H/V 条带深度卷积(k=7, 分离方向边缘) -> 拼接 -> 1x1 融合
   -> 局部std归一化门控: g = sigmoid( (|E| - mean_|E|) / (std_|E|+eps) )  (像素级, 参数free)
   -> x' = x + E * g  (残差注入但门控无参数、由局部统计决定, 不可被wd压死)
   依据: CLE speckle噪声空间不均一(局部std门控抑制低信噪区响应), 胃体腺体方向性结构
   (H/V条带), 恶性时方向紊乱。参数: 条带dwconv 2*C*7 + 融合 C*C。
   注: stage2 是前作未触碰区(前作结论"浅层细纹理应保持预训练7x7"指不替换dwconv,
   本模块不替换、在其后增强, dwconv预训练保留)。

## 2. 消融与叠加链计划
   快筛(Batch BA): dph / fesr / lsdg 单独 x seed42,43 (6任务, 并发3)
   叠加(Batch BB+): A+B / A+C / B+C -> A+B+C 三合一, 每环3seeds
   成功线: 任一指标 vs baseline(+0.5pt) 且 Macro-F1 均值 >= 0.9642(前作4s)

## 3. 跨域计划(≥2数据集有效)
   covid_v1 / busi_v1 / cub_v1 / heelBone 中选2-3, 3seeds, vs 各域convnext_tiny。

## 4. 命名
   工作名 convnext_dph / convnext_fesr / convnext_lsdg / 两两 / convnext_dph_fesr_lsdg。
   定型后起正式网络名。
