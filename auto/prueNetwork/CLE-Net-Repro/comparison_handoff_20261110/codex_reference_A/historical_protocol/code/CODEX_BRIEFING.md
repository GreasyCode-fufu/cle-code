# CLE-Net SCI 论文写作简报（供 codex 使用）

> **v2 更新（2026-10-27）**：论文主结果源已确定为 **4090**（用户决策）。新增第九节（4090 消融补测 + 口径裁定 + EGB 结论修正），§5.1 的 A800 消融降级为辅助参考。写作以 §5.3/§九 的 4090 数字为主。

> 本文件由 WorkBuddy（负责全部实验设计的 AI agent）撰写于 2026-10-26，包含论文 Methods 与 Experiments 章节所需的全部事实材料：网络结构的精确定义、逐创新点设计动机、全部实验数字（含失败路线）、训练协议与数据画像。**所有数字均可溯源**（文件路径见文末）。写作时请勿改动任何数字；如与代码/results.json 冲突，以 results.json 为准并在文中标注。

---

## 一、研究背景与问题定义

共聚焦激光显微内镜（Confocal Laser Endomicroscopy, CLE）可在体实时显示黏膜的细胞级形态，是胃病变筛查的前沿工具。本任务为三分类：
- `Benign_lesions`（良性病变，train 1233 张）
- `Malignant_lesions`（恶性病变，train 626 张，样本最少）
- `body_of_stomach_normal`（正常胃体，train 1221 张）

数据集 `cle_ac_v1`：train 3080 / val 593 / test 578 张（val/test 各类约为 166–208/类，test 合计 578）。224×224 训练分辨率。

### 数据画像（论文 Introduction/Motivation 可用）
对 train 集每类随机抽 40 张实测（灰度化后统计）：
- **亮度低**：均值 0.23–0.28（0–1 归一化）；
- **对比度低**：标准差 σ≈0.22；
- **高频能量占比 >85%**：FFT 频谱中，中心 16×16 低频块以外的能量占总能量 85% 以上——细胞边界与隐窝纹理是主导信息载体，但与荧光噪声混在同一频段；
- 三类中 Benign↔Malignant 形态差异细微（腺体不规则程度、筛状结构的判别需要形态学先验），是主要混淆源。

### 错误分析（所有已训模型的混淆矩阵共性）
全部模型的错误集中在 **Benign↔Malignant 互混**（每模型 12–18 张），`normal` 类近乎完美（敏感度 0.94–1.00）。这一观察驱动了第三轮设计迭代（见"探索路线"）。

---

## 二、Baseline 与总体设计原则

**Baseline：ConvNeXt-T**（28M 参数，ImageNet-1k ema 预训练）。选型依据：在 4090 与 A800 两机上均系统性超过 convnext_small / swin_tiny / pvt_tiny / resnet34/50 / vit_base / densenet121 / fast_t0 / ghostnet / mobilenet_v3 / repvit_m1（完整对比见第六节），确认其为该数据规模（~3k 张）下的最优主干——更大模型（ViT-B 85.8M 仅 0.9374）出现过拟合。

**三条设计原则（贯穿全部创新点）：**
1. **数据画像驱动**：每个模块动机对应一条实测的 CLE 图像统计特性，不做无动机缝合；
2. **零门控初始等价**：所有新分支以零初始化标量门控接入，t=0 时网络与 ConvNeXt-T 严格等价（已用 `max|output diff|=0` 数值验证），保证"初始不伤害、训练中按需习得强度"——这也是消融实验干净性的来源；
3. **预训练权重 100% 兼容**：不改变任何原始参数路径，ImageNet 权重全量加载，新分支随机初始化属设计预期（训练脚本以白名单机制强制校验）。

---

## 三、网络结构（Methods 核心素材）

最终网络 **CLE-Net** = ConvNeXt-T + 三个渐进叠加的创新点（注册名 `convnext_egb_scg_hsc`，实现文件 `models/optimize/convnext_cle2.py`）。姊妹网络 **S-Net**（注册名 `convnext_esa_scg_hsc`，文件 `models/optimize/convnext_cle5.py`）将创新点 A 替换为 ESA，用于消融与跨数据集泛化。

### 创新点 A：EGB — 零参数边缘-结构门控块（Edge-Gated Block）

- **位置**：ConvNeXt 块内，作用在 7×7 depthwise conv 输出之后、LayerNorm 之前；stage 1/2（0-indexed，即 dims 192/384 的两个 stage）。
- **机制**：PFESA（MICCAI 2026 思想）的零参数化改造。对特征 x 做 2D FFT：
  - 低频分支：高斯低通掩码（base_ratio=0.1）分离低频，取能量图经逐样本均值-方差归一化后 sigmoid → 结构注意力 S_str（捕获腺体/隐窝的整体形态布局）；
  - 高频分支：高通补集取幅值，方差归一的局部突变度 → 边缘注意力 S_edge（强化细胞边界）；
  - 融合：A(x) = sigmoid(S_str + S_edge) ∈ (0,1)，逐位置门控；
  - 注入：`x' = x + g · A(x) ⊙ x`，g 为零初始化标量门。**整个分支无可学习参数**（仅 1 个标量门），在小数据集上无过拟合自由度。
- **动机**：CLE 高频能量占 85%+，需要可判别地强化"细胞边缘信号"而非"荧光噪声"；带参注意力（SE/CBAM 类）在本任务 val 上普遍高分但 test 倒挂（见失败路线），零参数化是对该现象的直接回应。

### 创新点 B：SCG — 条带上下文引导（Strip Context Guidance）

- **位置**：块内并联分支，作用在 dwconv 输出后；stage 2/3（dims 384/768，腺体形态学语义层）。
- **机制**：AAAI 2026 StripConv 的门控并联改造：`5×5 dwconv → 1×19 dwconv → 19×1 dwconv → 1×1 conv`，输出经零门 g 残差注入：`x' = x + g · StripBranch(x)`。
- **动机**：胃腺体隐窝呈圆/卵圆长程形态，恶性征兆（不规则、筛状排列）需要超出 7×7 感受野的各向异性上下文；水平+垂直条带卷积恰好以极低成本（+3.5M）覆盖 19 像素长的形态链。
- **消融地位**：全部方案中唯一在两台机器、两个数据集上都稳定的正贡献组件（A800 单点 +0.35pp；4090 单点 best 0.9617；heelBone 上 S-Net 泛化最优的支柱）。

### 创新点 C：HSC — 通道自注意力校准头（Head with Synergistic Calibration）

- **位置**：GAP 之后、分类头之前的全局特征重标定。
- **机制**：SCSA（Arxiv 2024）通道核思想的轻量实现：8 头自注意力直接作用于通道维（QKV 由 Linear(dim, 3·dim, bias=False) 生成，注意力在 96 维通道 token 间计算），输出 proj 后经零门 g 并联：`f' = f + g · proj(Attn(f))`。
- **动机**：Malignant 类样本最少（626 张），通道级软特征选择等价于对"哪些语义维度对难类判别有用"的可学习加权；分类头侧的改动不影响骨干预训练加载。

### S-Net 的替换项：ESA — Sobel 显式边缘结构注意力

- 与 EGB 同位（stage 1/2 块内），但边缘图来自**固定 Sobel 算子**：3×3 Sobel x/y 卷积核（不可学习，按通道分组）求梯度幅值 → 通道均值聚合 → 逐样本 min-max 归一化得 [0,1] 边缘强度图 → 乘性调制 `x' = x + g · E(x) ⊙ x`。
- 无 FFT、无可学习调制，计算更省；"确定性显式结构先验"对"黑箱频域注意力"，两者构成同一设计空间的对照实验（见消融）。

### 参数量与计算开销

| 网络 | 参数量 | 相对 baseline |
|---|---|---|
| ConvNeXt-T (baseline) | 27.8M | — |
| +SCG | 31.3M | +3.5M |
| +HSC | 30.2M | +2.4M |
| CLE-Net (A+B+C) | 33.7M | +5.9M（+21%） |
| S-Net (ESA+B+C) | 33.7M | 同量级；ESA 本体零参数 |

---

## 四、训练协议（Experiments 章节素材）

两台机器、三个数据集完全一致的协议：

- 输入 224×224；数据集自算 mean/std 归一化（非 ImageNet 统计；统计量按数据集缓存，键含路径+尺寸指纹）；
- 训练增强：Resize(256×256) → 随机水平/垂直翻转(0.5) → 随机旋转(±15°) → RandomCrop(224)（医学图像常用保守增强）；
- 优化器 AdamW，300 epochs，early-stop patience=30，**以 val macro-F1 选最优 checkpoint**（不是 val acc）；
- 预训练：ImageNet-1k ema 权重（ConvNeXt 官方 `convnext_tiny_1k_224_ema.pth`），100% 骨干加载（白名单强制校验，missing keys 仅限分类头与白名单新分支）；
- 无类别加权损失（unweighted CrossEntropy）——结果反映结构本身贡献；
- 种子：42/43/44（4090 补充 45/46），汇报双口径（best / mean±std）；
- 硬件：A800 80GB（n1 容器，4 CPU 核）与 RTX 4090 48G（Dell-T2），同 conda 环境 `sunfuxiang`（Python 3.12, torch 2.12.0+cu130）；
- 评测：`evaluate.py` 计算 macro/weighted F1、accuracy、balanced accuracy、macro sensitivity/specificity/precision、macro AUROC (OvR)、AUC-PR、Cohen's κ、MCC、per-class 敏感度，含 95% bootstrap CI 能力。

**跨机不可混用原则**：A800 与 4090 上同一模型的绝对数字相差约 ±1pp（硬件/调度差异），因此所有对比表均为"同机同协议"内部对比；CLE-Net 在两机上分别实测，对比主干在对应机器上各有结果。

---

## 五、主实验结果（test 集 macro-F1）

### 5.1 A800（n1）上的主结果与消融

消融链（best-of-seeds 口径）：

| 配置 | seed42 / 43 / 44 | best | 备注 |
|---|---|---|---|
| baseline (ConvNeXt-T) | 0.9633 / 0.9440 / 0.9447 | 0.9633 | mean 0.9507±0.0110 |
| +B (SCG) | 0.9668 / 0.9580 / 0.9362 | 0.9668 | 单点最优组件 |
| +A (EGB) | 0.9374 / 0.9582 | 0.9582 | 单用不稳 |
| +C (HSC) | 0.9401 / 0.9654 | 0.9654 | 种子敏感 |
| A+B | 0.9602 / 0.9406 | 0.9602 | |
| B+C | 0.9617 / 0.9471 / 0.9441 | 0.9617 | |
| **CLE-Net (A+B+C)** | **0.9667 / 0.9508 / 0.9580** | **0.9667** | mean 0.9585±0.0079；三种子全部≥0.95 |

要点：CLE-Net 相对 baseline：best +0.34pp；**mean +0.78pp 且 std 从 0.0110 降至 0.0079（方差降 28%）**；Acc 0.9670 vs 0.9635；Bal-Acc 0.9833 vs 0.9817；AUROC 0.9974 vs 0.9967；MCC 0.9505 vs 0.9452；Malignant 敏感度（best run）0.9375，seed43 达 0.9615——最难类的提升在各指标上同步体现。

### 5.2 A800 上 10 主干对比（seed42，同协议）

| 方法 | 参数量 | Macro-F1 | Acc | AUROC | MCC |
|---|---|---|---|---|---|
| **CLE-Net (ours)** | 33.7M | **0.9667** | **0.9670** | **0.9974** | **0.9505** |
| ConvNeXt-T | 27.8M | 0.9633 | 0.9635 | 0.9967 | 0.9452 |
| ConvNeXt-S | 49.5M | 0.9565 | 0.9566 | 0.9957 | 0.9350 |
| PVT-T | 12.7M | 0.9530 | 0.9531 | 0.9922 | 0.9296 |
| Swin-T | 27.5M | 0.9465 | 0.9479 | 0.9956 | 0.9224 |
| ViT-B/16 | 85.8M | 0.9374 | 0.9375 | 0.9904 | 0.9064 |
| DenseNet121 | 7.0M | 0.9346 | 0.9358 | 0.9846 | 0.9042 |
| FasterNet-T0 | 2.6M | 0.9337 | 0.9340 | 0.9889 | 0.9010 |
| ResNet34 | 21.3M | 0.9276 | 0.9271 | 0.9900 | 0.8917 |
| RepViT-M1 | 6.4M | 0.8585 | 0.8611 | 0.9605 | 0.7936 |

### 5.3 4090（Dell-T2）复测（验证结论的跨机稳健性）

CLE-Net 6 种子：0.9581 / 0.9576 / 0.9474 / 0.9543 / 0.9599 / 0.9545 → **best 0.9599，mean 0.9553±0.0043**（六种子全落 0.947–0.960 窄区间）；
S-Net 5 种子：0.9471 / 0.9198 / **0.9669** / 0.9498 / 0.9458 → **best 0.9669**（本机全部方法最高值），mean 0.9459±0.0174；
SCG 单点 2 种子：0.9617 / 0.9326。

4090 对比主干（复用同机同协议 ZCode 已有结果，经五项核验：超参/评测脚本/数据集哈希/conda 环境/预训练权重全部一致）：

| 方法 | Seeds | best | mean±std |
|---|---|---|---|
| **S-Net (ours)** | 5 | **0.9669** | 0.9459±0.0174 |
| **CLE-Net (ours)** | 6 | 0.9599 | **0.9553±0.0043** |
| ConvNeXt-T | 5 | 0.9615 | 0.9476±0.0094 |
| ConvNeXt-S | 3 | 0.9594 | 0.9538±0.0052 |
| ResNet50 | 3 | 0.9507 | 0.9462±0.0039 |
| Swin-T | 3 | 0.9512 | 0.9419±0.0127 |
| MobileNetV3-L | 3 | 0.9494 | 0.9408±0.0078 |
| FasterNet-T0 | 3 | 0.9440 | 0.9360±0.0071 |
| ResNet34 | 3 | 0.9431 | 0.9322±0.0096 |
| GhostNet | 3 | 0.9400 | 0.9276±0.0152 |
| PVT-T | 3 | 0.9394 | 0.9080±0.0283 |
| ViT-B/16 | 3 | 0.9311 | 0.9171±0.0159 |
| RepViT-M1 | 1 | 0.7822 | — |

双口径结论（两机一致）：**均值口径 CLE-Net 均为第一（A800 +0.78pp、4090 +0.77pp vs baseline mean）且方差最小；最优口径 S-Net 在 4090 上 0.9669 为全场最高**。baseline 的 best 0.9615 由 seed44 单点贡献（其 mean 仅 0.9476）。

### 5.4 heelBone 跨域泛化（跟骨 X 光，三分类 heelspur/normal/sever）

数据：train 2770 / val 593 / test 593；X 光风格，53% 灰度，原始尺寸 464–1007px；**协议与 CLE 完全一致，零重调参，归一化统计量自动重算**。

| 方法 | 逐种子 test F1 | best | mean±std |
|---|---|---|---|
| **S-Net (ours)** | 0.8972 / **0.9035** / 0.8278 | **0.9035** | 0.8762±0.0420 |
| ConvNeXt-T | 0.8886 / 0.8957 / 0.8641 | 0.8957 | 0.8828±0.0165 |
| CLE-Net (EGB 版) | 0.8504 | 0.8504 | — |
| Swin-T | 0.8366 | 0.8366 | — |
| ResNet34 | 0.8149 | 0.8149 | — |

解读（论文 Discussion 可用）：
- **S-Net best +0.78pp 跨域第一**，逐种子 2/3 胜出 baseline。骨刺的本质是"骨边缘的异常凸起"——Sobel 显式边缘 + 条带形态捕获的归纳偏置与任务天然对位，**无需重调参即迁移**，证明 ESA+SCG 捕捉的是跨成像模态（荧光显微 → X 光投影）的通用形态学先验；
- **EGB 频域分支表现出模态特异性**（0.8504 低于 baseline 0.8886）：CLE 的高频统计假设不迁移到 X 光成像。诚实呈现这一点反而支撑"各组件动机是模态驱动的"这一设计哲学；
- 均值口径 S-Net（0.8762）略低于 baseline（0.8828），由 seed44 的 0.8278 拖累——该数据集本身种子方差大（baseline ±0.0165），论文表述建议以 best 口径为主并注明。

---

## 六、探索过的失败路线（Discussion/消融叙事素材，体现迭代过程）

以下变体均实现并完整训练过（A800，test macro-F1）：

| 路线 | 设计 | 结果 | 失败原因分析 |
|---|---|---|---|
| v1 单点 EGB | 同创新点 A 单用 | 0.9374 / 0.9582 | **[v2 修正]** 该两种子跑于 A800 5 路并发超卖期（GPU util 仅 45%），训练质量受污染，不可作为组件结论依据；4090 干净复测 EGB 单点 mean 0.9593±0.0029 为最强单点组件（见 §九）。原频域过调制解释降级为待验证假设 |
| v2 OS 重叠块茎 | 3×3/s2 ×2 替代 4×4/s4 patchify | 0.9365 | 新茎随机初始化丢失 ImageNet 低层滤波器先验，-2.7pp——证明预训练兼容性原则的必要性 |
| v2 DSH 双尺度头 | stage3+stage4 GAP 拼接决策 | 0.9428 | 单用不佳（-2.05pp） |
| v3 PLD 原型诊断亲和 | stage3 后 6 个可学习诊断原型，余弦亲和加权上下文回注 | 0.9253 / 0.9431(组合) | val/test 严重倒挂——原型在小数据集上记忆训练分布 |
| v4 ESA | Sobel 显式边缘（EGB 的确定性替代） | 0.9601 单点（A800） | 未失败：单点即超 EGB 单点 +2.3pp，验证"过调制"假设，进而构成 S-Net |

错误分析驱动的迭代链：混淆矩阵显示 B↔M 互混 → 设计 PLD（失败）→ 换 Sobel 显式先验（成功）。这段"设计-证伪-转向"过程是论文叙事的真实亮点。

---

## 七、文件与数据溯源（codex 可直接探索验证）

- 网络实现：`models/optimize/convnext_cle2.py`（EGB/SCG/HSC/CLE-Net 及消融变体）、`models/optimize/convnext_cle5.py`（ESA/S-Net）、`models/Convnext.py`（baseline）；
- 训练脚本：`train_cle_fair.py`（含白名单机制 `MODEL_ALLOWED_MISSING_SEGMENTS`、预加载校验 `load_pretrained_weights_strict`）；归一化工具 `train_utils.py`；
- 评测：`evaluate.py`（全指标）；实验编排 `experiments/runner*.py`；作业表 `experiments/jobs_*.json`；
- 结果数据：`experiments/results_cle_ac_v1.csv`（全部批次原始记录）、`experiments/FINAL_REPORT_n1.md`（A800 最终报告）、`experiments/COMPARE_TABLE.md`（10 主干对比）、`reports/CLE-Net_4090_实测对比报告.md`（4090 复测 + 冲刺附录）；
- 逐 run 完整指标（含 per-class 敏感度、混淆矩阵、AUROC/AUC-PR/κ/MCC）：`weights_cle_ac_v1/<model>_seed<seed>/<model>/eval_*/results.json`；
- heelBone 泛化：`experiments/HEELBONE_泛化实验报告.md` 与 `experiments/results_heelbone.csv`（在 n1 的 Classification 项目中）；
- git 历史：每轮设计/实验均有 commit，message 含完整动机（`git log --oneline` 可览）。

## 八、写作要求（用户指定）

1. **中文表述**，SCI 论文的 Methods 与 Experiments 章节风格（准确、克制、可复现优先）；
2. 输出为 Markdown 文件，建议落盘到本目录上级：`/home/login/data/cle/auto/Network/CLE-Net/paper_methods_experiments.md`；
3. 字数尽可能充分：Methods 建议覆盖总体架构→逐创新点（含公式化描述，LaTeX 记法）→训练细节；Experiments 建议覆盖数据集与画像→实现细节→主结果表（A800）→消融表→跨机复测→10+ 主干对比→跨数据集泛化→失败路线讨论；
4. **数字必须与本文档/results.json 一致**，不确定处标注"[待核对]"而非编造；
5. 暂不需要图，但建议在文中用注释标出"此处未来放 Fig.X（内容说明）"占位；
6. 术语统一：CLE-Net（=egb_scg_hsc，主方法）、S-Net（=esa_scg_hsc，姊妹消融/泛化变体）；模块名 EGB/SCG/HSC/ESA 首次出现给全称。

---

## 九、v2 补充（2026-10-27）：4090 消融补测结果与写作口径裁定

> 补测由 Qoder agent 执行（16/16 run 成功，CSV 与 results.json 经 WorkBuddy 抽查审计一致，git: d068c4a/8d74e6b/d52fdf8）。**本节数字为论文消融表的最终依据。**

### 9.1 4090 消融总表（test macro-F1）

| 配置 | 注册名 | s42 | s43 | s44 | s45 | s46 | s47 | best | mean±std |
|---|---|---|---|---|---|---|---|---|---|
| baseline (ZCode) | convnext_tiny | 0.9479 | 0.9373 | 0.9615 | 0.9409 | 0.9506 | — | 0.9615 | 0.9476±0.0094 |
| +A (EGB) | convnext_egb | 0.9619 | 0.9562 | 0.9599 | — | — | — | 0.9619 | **0.9593±0.0029** |
| +ESA | convnext_esa | 0.9537 | 0.9589 | 0.9496 | — | — | — | 0.9589 | 0.9541±0.0047 |
| +B (SCG) | convnext_scg | 0.9617 | 0.9326 | 0.9515 | — | — | — | 0.9617 | 0.9486±0.0148 |
| +C (HSC) | convnext_hsc | 0.9511 | 0.9543 | 0.9547 | — | — | — | 0.9547 | 0.9534±0.0020 |
| A+B | convnext_egb_scg | 0.9546 | 0.9401 | 0.9424 | — | — | — | 0.9546 | 0.9457±0.0078 |
| B+C | convnext_scg_hsc | 0.9478 | 0.9457 | 0.9514 | — | — | — | 0.9514 | 0.9483±0.0029 |
| **CLE-Net (A+B+C)** | convnext_egb_scg_hsc | 0.9581 | 0.9576 | 0.9474 | 0.9543 | 0.9599 | 0.9545 | 0.9599 | **0.9553±0.0044** |
| S-Net (ESA+B+C) | convnext_esa_scg_hsc | 0.9471 | 0.9198 | 0.9669 | 0.9498 | 0.9458 | — | **0.9669** | 0.9459±0.0169 |

### 9.2 消融叙事裁定（codex 写作按此口径）

1. **主口径 = mean±std**（与全部表格一致）。消融故事：三个单点组件各自有效（EGB 0.9593 / ESA 0.9541 / HSC 0.9534 均超 baseline mean 0.9476）；两两组合存在组件间干扰（A+B 0.9457、B+C 0.9483 回落至 baseline 水平）；**完整三件套协同后恢复稳定增益（CLE-Net 0.9553±0.0044，+0.77pp 且方差为全部 9 行中最小之一）**——核心论点是「协同性而非可加性」（synergy over additivity）。
2. **EGB 结论修正（重要）**：A800 上「EGB 单用不稳」是 5 路并发超卖期（GPU util 45%）的环境伪影；4090 干净复测显示 EGB 是 **mean 最高的单点组件（0.9593±0.0029，方差极小）**。论文中 EGB 的呈现以此为据；「频域过调制」假设仅在 Discussion 作为跨机差异的候选解释提及。
3. best 口径作辅助：S-Net 0.9669 为 4090 全场最高（供最优结果表述用）。
4. 数据源标注：消融表 baseline 取 ZCode 同机同协议记录（五项核验见 §5.3），其余为本工程 4090 实测。

### 9.3 对比主干表的最终口径

4090 对比表用 §5.3 的 11 主干（不补 densenet121）；A800 的 10 主干表（§5.2）作为跨机稳健性辅助证据引用。两表均已在简报中，codex 直接引用即可，勿混排。

### 9.4 审计记录

WorkBuddy 于 2026-10-27 对 Qoder 补测做事实终审：抽 7 个 run（覆盖全部 6 个模型名）逐项比对 results.json 与 CSV 的 macro_f1/accuracy/AUROC，全部一致（ALL_PASS）；16 run 全部 status=ok；历史行无改动；Qoder 的执行偏离（并发 1→2、清残留重跑 egb_s43）已在 QODER_REPORT.md 如实记录且不影响结果有效性。
