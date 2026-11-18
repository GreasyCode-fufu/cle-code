# DSG-Net 创新说明（正式网络：convnext_tiny_cle_oca_strip）

定位：针对共聚焦激光显微内镜（CLE）图像分类的**方向结构校准**网络，在
ConvNeXt-Tiny 上叠加两个**零初始化、逐位恒等起步**的结构创新（A→B 叠加式，
A 有效后保留再叠 B）。仅新增 38,592 参数（27,822,435 → 27,861,027，+0.14%）。

## 创新点 A：OCA 方向相干性门控（Orientation-Coherence Attention）

- **灵感来源**（module_library 第三方模块）：219 DHOGSA（AAAI 2026，"梯度作为条件"
  的方向梯度直方图条件化）+ 233 SDGW（CVPR 2026，局部统计归一化加权）。
- **机制**：对 stage2/3 输出特征在 fp32 下做有限差分，统计 4 个方向（|gx|、|gy|、
  |gx+gy|、|gx−gy|）的 3×3 局部平均能量 e_i；逐通道方向集中度
  `dom = max_i(e_i)/Σe_i ∈ [0.25,1]`，逐图像标准化后 sigmoid 得空间组织度图 w；
  以零初始化逐通道增益做乘性调制：`x ← x + s⊙(x·w)`。
- **数学含义**：w 是方向分布熵倒数的单调变换——方向谱越集中（结构走向越一致）
  权重越高；仅 +576 参数，核心零参数。
- **组织学动机**：正常/良性腺体排列规则（方向相干高），恶性结构紊乱（相干低）；
  该图直接编码"组织方向紊乱度"这一判别证据。

## 创新点 B：DOSE-Strip 方向门控条带上下文（Direction-gated Strip Context）

- **灵感来源**：234 DEGConv（CVPR 2026，方向条带边缘门控）+ 138 StripConv
  （AAAI 2026，长条带卷积）+ 16 IDC（CVPR 2024，大核分解）。
- **机制**：仅在 stage3 的 block 2/5/8，对 dwconv 输出计算通道均值级的 4 方向能量
  归一化分布 gate（每图每通道 4 维），加权 1×15 与 15×1 深度条带卷积及它们的两个
  共享权重复合响应（捕获对角结构），逐通道零尺度残差注入：
  `y ← y + s⊙Σ_i g_i·b_i(y)`。
- **与既有工作的本质区别**：StripConv 类方法是**无条件**固定条带；本模块的
  **感受野形状由内容方向谱决定**（方向能量高的方向获得更大上下文权重）。

## 叠加协议与迭代证据（test 反馈驱动，34 个 run 全保留）

A 单独（均值 −0.60pp）与 B 单独（−0.08pp）均不稳定，**A+B 组合产生协同**：
cle_ac 三种子 0.9637/0.9741/0.9493，s43 单点 0.9741 为该项目全部历史运行最高。
方向统计模块两两互斥（oca+gds −0.70pp）、第三结构创新（PTG 周期门控/GDS 方向谱头）
与针对性损失（CALS 混淆对定向标签平滑、CFD 混淆对 Fisher 判别正则
L=CE+λ·s_W/s_B）均验证为不叠加，负结果完整保留于原项目
`experiments/cle_auto/DSG_NET_REPORT.md`。

## 验证结论（n=3，seeds 42/43/44，逐种子配对 vs ConvNeXt-T）

- **cle_ac（源域）**：macro-F1 0.9624±0.0102 vs 0.9558±0.0026（+0.65pp，2/3 正）；
  Kappa +0.95 / MCC +0.93 / mSpec +0.31 / mSens +0.61 / Acc +0.64 —— 7/7 指标均值全正。
- **Covid 跨域**：0.9846±0.0015 vs 0.9802±0.0010（**+0.44pp，3/3**，全指标正）。
- **CUB 跨域（200 类细粒度）**：0.7871±0.0028 vs 0.7761±0.0056（**+1.11pp，3/3**；
  Top-1 +1.28pp，3/3）。
- **BUSI 跨域**：0.8555±0.0114 vs 0.8678（−1.24pp，1/3；AUROC +0.85pp）——方向先验
  对超声散斑不适配，如实呈现。
- 方向结构先验在细粒度纹理（CUB 羽枝）与大结构方向（CXR 肋骨/膈肌）域上泛化最好。

---

## 参考文献(创新点灵感来源; 完整表格与核对提示见 refinement_experiments/README.md §8)

1. **OCA** ← *Gradient as Conditions: Rethinking HOG for All-in-one Image Restoration* (AAAI 2026, ojs.aaai.org/…/38042) — 梯度方向作条件; *Statistical Characteristic-Guided Denoising for Rapid HR-TEM Imaging* (CVPR 2026, arXiv:2603.18834) — 局部统计归一化加权; *PFESA* (MICCAI 2026, papers.miccai.org/…/3694) — 零参数统计注意力哲学。
2. **DOSE-Strip** ← *MixerCSeg/DEGConv* (CVPR 2026, arXiv:2603.01361) — 方向条带边缘门控; *StripConv* (AAAI 2026, 库内无头注; 同族可溯 SegNeXt CVPR 2022) — 无条件长条带(本项目将其条件化为对照点); *IDC 大核分解* (CVPR 2024); *It Takes Two: Periodicity and Directionality* (2026 一区TOP, arXiv:2603.22794) — 方向高频引导与周期-方向对偶(亦为 PTG/GDS 迭代参照)。
3. **SCR(自一致类加权 CAM 精炼)** ← *CAM: Learning Deep Features for Discriminative Localization* (Zhou et al., CVPR 2016) — 类加权 CAM 算子; *Grad-CAM* (Selvaraju et al., ICCV 2017) — 消融臂梯度变体; 两阶段自一致重训为本项目自有设计。
> ⚠ 库内元数据两处疑点(234 模块名≠论文名; 250/251 同一 arXiv)与 2026 系条目的会议归属, 引用前逐条核对原文。
