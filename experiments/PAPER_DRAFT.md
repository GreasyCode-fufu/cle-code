# （草稿——数字以最终实验为准）SF-ConvNeXt：面向共聚焦激光显微内镜图像分类的尺度–频率感知网络

> 方法与实验部分（中文 SCI 风格）。最终网络注册名 `convnext_v2abc`；基线 ConvNeXt-T。
> 表格数值将在全部实验完成后填入/更新。

---

# 2 方法

## 2.1 任务定义与总体架构

共聚焦激光显微内镜（confocal laser endomicroscopy, CLE）可在体获得约 1000 倍放大、近组织学分辨率的消化道黏膜显微图像。本文将其形式化为三分类问题：良性病变（Benign_lesions）、恶性病变（Malignant_lesions）与正常胃底体黏膜（body_of_stomach_normal），记输入 $x \in \mathbb{R}^{3\times224\times224}$，目标为类别 $y \in \{1,2,3\}$。

CLE 图像的判读依据（Mainz 分类等）集中于两类线索：**腺体结构的规则性**（良性/正常腺体呈圆形或椭圆形、大小一致、排列整齐；恶性腺体大小不一、融合、排列紊乱）与**细胞级细节**（核增大、荧光增强、核浆比升高、极性紊乱与细胞边界破坏）。前者是中低频、跨尺度的结构性模式，后者是高频、细粒度的局部纹理。标准 ConvNeXt-T 在这两方面存在结构性欠缺：其 4×4/stride-4 茎干在第一层即发生 16 倍降采样，对细胞级高频纹理造成不可逆混叠；每个块内仅使用单一 7×7 深度卷积，感受野与频带均固定；分类头仅对最深 stage 的特征做全局平均池化，丢弃了跨尺度证据。

为此，本文提出 **SF-ConvNeXt（Scale–Frequency-aware ConvNeXt）**，在 ConvNeXt-T 的三个深度上分别引入结构化改造（图略）：

- **WS（Wavelet Stem，小波细节保真茎干）**：以 Haar 小波分解 + 两级温和卷积下采样替换原一步 4×4/stride-4 茎干，在降采样的同时显式保留高频子带（§2.2）；
- **WR（Wavelet-Replace Block，频域多尺度卷积块）**：将 stage-3/4 中每个块的 7×7 深度卷积整体替换为 WTConv——由继承预训练权重的 7×7 基卷积与 Haar 级联小波路径构成，使感受野按层数指数扩展并具备频带选择性（§2.3）；
- **CH（Concat multi-scale Head，三尺度拼接分类头）**：分类头输入为 stage-2/3/4 三级特征经投影、池化后的拼接向量，从结构上强制分类器消费跨尺度证据（§2.4）。

三项改造均遵循两条设计约束：**(i) 预训练兼容**——除被替换组件外，所有 ConvNeXt 原始参数的键名保持不变（被替换的 7×7 深度卷积权重通过键重映射迁移至 WTConv 的基卷积分支），ImageNet-1k 预训练权重经严格审计后可完整加载；**(ii) 不可旁路（non-bypassable）**——每个创新点均以替换而非并联门控方式集成。我们在预实验中发现（见 §3.5），零初始化门控的并联分支在权重衰减（5×10⁻²）作用下会被压向恒等映射，训练后其输出贡献仅为主干特征的 0.05%–0.4%，将门控置零后测试 Macro-F1 变化不超过 0.35 个百分点——此类"可旁路"设计带来的表观提升不可归因于模块本身的表征贡献，因此本文全部创新点均采用结构化替换设计。

## 2.2 小波细节保真茎干（WS）

原茎干 $S_0(x)=\mathrm{LN}(\mathrm{Conv}_{4\times4,s=4}(x))$ 以 stride 4 一步降采样，等效于先低通后欠采样，细胞核与腺体边缘的高频信息在第一层即被混叠损失。WS 改为两级温和路径：

$$
\mathrm{LL},\mathrm{LH},\mathrm{HL},\mathrm{HH} = \mathrm{DWT}_{ Haar}(x),\qquad
\tilde x = \big[\mathrm{LL};\mathrm{LH};\mathrm{HL};\mathrm{HH}\big]\in\mathbb{R}^{12\times112\times112},
$$

$$
S_{WS}(x)=\mathrm{LN}\big(\mathrm{Conv}_{3\times3,s=2}(\tilde x)\big)\in\mathbb{R}^{96\times56\times56},
$$

其中 Haar 分析滤波器组为固定常数（$[\tfrac{1}{\sqrt2},\tfrac{1}{\sqrt2}]$ 与 $[\tfrac{1}{\sqrt2},-\tfrac{1}{\sqrt2}]$ 的外积，无学习参数），四个子带拼接后经单个 3×3/stride-2 卷积投影至 96 维。与原茎干相比：降采样由"一步 16 倍"变为"两步各 2 倍"，且高频子带（LH/HL/HH）以显式张量形式参与后续线性组合而非被直接丢弃。茎干 LayerNorm 的预训练参数保持迁移；3×3 卷积（约 1.0×10⁴ 参数）随机初始化。该设计仅为 0.01M 参数的结构替换。

## 2.3 频域多尺度卷积块（WR）

ConvNeXt 块的 token 混合器为单一 7×7 深度卷积，感受野与频带固定。腺体结构的自相似性跨越多个尺度（单腺体 → 腺体簇 → 黏膜架构），需要兼具大感受野与频带选择性的空间混合。WR 将 stage-3、stage-4（共 12 个块）中的 7×7 深度卷积替换为 WTConv 算子（Kernel size 7 基卷积 + Haar 级联小波路径）：

$$
\mathrm{WTConv}(u)=\mathrm{Base}(u)+\mathcal{IW}\Big(\Psi_2\big(\mathcal{W}_2(\ \Psi_1(\mathcal{W}_1 u))\big)\Big),
$$

其中 $\mathrm{Base}(u)=\gamma_b\odot \mathrm{DWConv}_{7\times7}(u)$（**直接继承该块预训练的 7×7 深度卷积权重**，$\gamma_b$ 初始为 1）；$\mathcal{W}_l$ 为第 $l$ 级 Haar 分解（对上一级低频子带作用，空间减半、通道×4），$\Psi_l$ 为作用于各子带的 3×3 分组深度卷积（分组数=4C，带可学习逐通道尺度 $\sigma$，初始 0.1），$\mathcal{IW}$ 为逆小波重构。两级级联使等效感受野以指数增长而参数仅线性增加；重复作用于低频子带使算子天然 emphasis 于结构/形状信息，而基卷积分支保持局部高频响应。stage-1/2 不作替换：消融实验（§3.4）表明浅层细纹理通路应保持预训练原样。

替换后块内其余路径（LayerNorm → PWConv(4C) → GELU → PWConv(C) → LayerScale $\gamma$ → 随机深度/残差）与原 ConvNeXt 块完全一致。

## 2.4 三尺度拼接分类头（CH）

原分类头仅使用 stage-4 的全局池化特征。恶性/良性鉴别同时依赖深层语义（腺体架构紊乱）与浅层细节（核形态），单一尺度池化构成信息瓶颈。CH 将分类头输入改为三级拼接：

$$
v=\big[\mathrm{LN}_4(\mathrm{GAP}(f_4));\ \mathrm{LN}_3(\mathrm{GAP}(P_3 f_3));\ \mathrm{LN}_2(\mathrm{GAP}(P_2 f_2))\big]\in\mathbb{R}^{2304},
\qquad \hat y = \mathrm{Linear}(v),
$$

其中 $f_i$ 为 stage-$i$ 输出，$P_3\in\mathbb{R}^{768\times384},P_2\in\mathbb{R}^{768\times192}$ 为 1×1 投影（无偏置），$\mathrm{LN}_4$ 复用预训练的最终 LayerNorm，$\mathrm{LN}_2,\mathrm{LN}_3$ 为新 LayerNorm。由于分类器的每个 logit 都必须读取全部 2304 维输入，跨尺度证据的融合在结构上不可旁路。

## 2.5 实现与训练协议

所有模型在相同协议下训练以保障公平比较：AdamW（lr=1×10⁻⁴，weight decay=5×10⁻²），余弦退火（T_max=300），批大小 32，输入 224×224，混合精度训练；训练增广为 256² resize + 水平/垂直翻转 ±15° 旋转 + 随机裁剪；预训练模型使用 ImageNet 归一化。以验证集 Macro-F1 选择最优 checkpoint（早停耐心 30，最多 300 轮）。预训练权重加载采用严格审计：除分类头与白名单内新增段外，任何键缺失/多余即报错终止，杜绝静默错配。每个配置以固定种子（42/43/44，部分加 45/46）重复训练，报告均值与标准差及最优单次结果。实验环境：单卡 RTX 4090，PyTorch 2.x，conda 环境 sunfuxiang。

---

# 3 实验与结果

## 3.1 数据集与评价指标

数据集 cle_ac_v1（患者级划分的 CLE 胃黏膜三分类数据集）：训练 2077（良性 830 / 恶性 371 / 正常 876）、验证 427（195/79/153）、测试 576（208/176/192）张；恶性类样本最少（训练集 17.8%），类别不平衡明显，故以 **Macro-F1 为主指标**，同时报告 Accuracy、Weighted-F1、Balanced Accuracy、Macro-Sensitivity/Specificity、Macro-AUROC(OvR)、Cohen's Kappa 与 MCC（evaluate.py 标准协议：中心裁剪推理、与训练一致的归一化、thop 复杂度统计）；对最终模型额外给出 1000 次 bootstrap 的 95% 置信区间。

## 3.2 与主流网络的对比实验

（表：convnext_tiny/small、swin-t(vanilla)、pvt-t、resnet34、vit-b、fasternet-t0、repvit-m1 ×3 seeds mean±std + 本方法）

## 3.3 组件递进消融

（表：baseline → +WS → +WS+WR → +WS+WR+CH，3 seeds；及 v2 单组件/两两组合）

## 3.4 结构消融

（表：WR 的插入位置 s234 vs s34、子带核宽 k3 vs k5、分解级数 l1 vs l2 —— 来自 v1 系列同构对照）

## 3.5 机制分析：门控判决实验与注入贡献度

（v1 门控系列：门控范数表、注入贡献比表、门控置零前后 F1 与预测翻转表；论证可旁路设计的失效与结构化设计的必要性）

## 3.6 效率分析

（表：Params/MACs/FPS）

## 3.7 临床指标与置信区间

（最终模型 per-class Sens/Spec/Prec/F1/AUROC、混淆矩阵、bootstrap CI）

## 3.8 统计稳定性

（多种子 mean±std；与基线的逐种子配对差）
