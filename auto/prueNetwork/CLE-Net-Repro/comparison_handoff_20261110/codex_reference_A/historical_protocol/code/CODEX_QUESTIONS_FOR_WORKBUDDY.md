# CLE-Net 论文写作前事实核对与待答问题

**致：WorkBuddy（网络设计与实验负责人）**  
**整理：Codex｜日期：2026-10-27（Asia/Shanghai）**  
**项目：`/home/login/data/cle/auto/Network/CLE-Net`，SSH：`ubuntu-server`**

## 本文件用途与当前状态

用户最新要求是：先不撰写方法章节，将疑问写入文件，由 WorkBuddy 阅读并回复。因此，本文件是事实核对与问题清单，**不是 Methods/Experiments 正文**；本轮没有生成 `paper_methods_experiments.md`。

已读 `CODEX_BRIEFING.md` v2，理解写作决策为：4090 是主结果源，以第九节和 §5.3/§5.4 为主，mean±std 为主口径，A800 为辅助。以下问题源于简报与实际源码、训练日志、元数据及原始评测文件之间的差异。为避免把设计意图写成已经实现或验证的事实，需先确认。

本轮核对范围：

- 只连接了 ubuntu-server，没有连接 n1 或其他校园服务器。
- 读取了 4090 主表及消融涉及的 **62 份 test results.json**（本工程 29 份、ZCode 33 份），逐项对照 CSV 的 macro-F1、accuracy、AUROC；四位小数结果全部一致。聚合统计从 JSON 未舍入值计算。
- 读取本工程 **29 份 run_meta.json**、对应训练日志，以及 EGB、ESA、CLE-Net、S-Net 的 seed42 已训 checkpoint。
- 检查模型文件 Git 历史；以相同 baseline 参数进行 CPU 初始化前向验证，并统计模型参数。没有重训或重新计算测试预测，没有修改模型、训练脚本或历史结果。
- 核对期间源码 HEAD 为 `fc466283c6c55224477ecf5b31fcddf366f7c2ed`。主要模型文件从部署提交 `2a64f43` 到该 HEAD 没有变化。

**请优先回复 Q01、Q02、Q04、Q05、Q07、Q08、Q09。** 对已由源码或原始结果确认的差异，希望回复“是否属于简报笔误、采用什么最终表述”；若认为存在另一份实际训练实现，请给出对应文件、commit、运行目录或权重证据。

建议直接在同目录新建 `WORKBUDDY_REPLY_TO_CODEX.md`，按本文 Q 编号逐项答复。请区分：①现有证据已确认；②设计解释/假设；③需要补充证据或实验。当前请求是先解释与定位来源，不要求为了匹配简报而修改历史记录，也不要求立即补跑实验。

---

## Q01｜最终模型的模块位置与“渐进叠加消融”是否需要改写？

**核对到的实现：** `models/optimize/convnext_cle2.py` 的 `CLENet.__init__` 先把指定 stage 替换为 EGBBlock，再把 SCG stages 整块替换成 SCGBlock。后一次替换覆盖前一次 EGB，未在同一个块内保留两条分支。`convnext_cle5.py` 的 SNet 同理。

下表均采用 **0-indexed stage**；四阶段宽度为 96/192/384/768，深度为 3/3/9/3。

| 注册名 | 实际 EGB stages | 实际 ESA stages | SCG stages | HSC |
|---|---|---|---|---|
| convnext_egb | 1、2、3 | — | — | 无 |
| convnext_esa | — | 1、2 | — | 无 |
| convnext_scg | — | — | 2、3 | 无 |
| convnext_hsc | — | — | — | 有 |
| convnext_egb_scg | 1 | — | 2、3 | 无 |
| convnext_scg_hsc | — | — | 2、3 | 有 |
| CLE-Net / convnext_egb_scg_hsc | 1 | — | 2、3 | 有 |
| S-Net / convnext_esa_scg_hsc | — | 1 | 2、3 | 有 |

这不是仅看当前源码推测：四份 seed42 checkpoint 的参数键确认 EGB 单点有 15 个边缘门、ESA 单点有 12 个边缘门，而完整网络只有 stage1 的 3 个边缘门，以及 stage2/3 的 12 个 SCG 门。29 份 run_meta 的加载缺失键也一致。

**与简报差异：** 简报称 EGB 位于 stages1/2，并将 A→A+B→A+B+C 称为渐进叠加，但未说明单点与组合中的 EGB 作用阶段改变。

**请回复：**

1. 这种按 stage 分工、后构造 SCG 覆盖 EGB/ESA 的实现，是原本设计，还是构造时的非预期覆盖？
2. Methods 是否应忠实写成“stage1 边缘分支 + stages2/3 条带分支 + 分类前 HSC”？
3. 消融表是否应增加“实际作用阶段”说明，并把当前实验称为配置比较，而不把差异全部归因于同一位置组件间的协同/干扰？
4. 是否已有“仅 stage1 的 EGB 单点”等固定位置对照？若有，请提供已完成 run 的路径；若无，可将严格析因验证列为局限。

## Q02｜HSC 的注意力轴应如何命名？

**核对到的实现：** `HSCHead.forward` 将 768 维特征经 QKV 投影后，得到 `Q,K,V.shape=(B,8,96)`。因此：

`Q @ K.transpose(-2,-1)` 的形状是 **(B,8,8)**，不是 (B,8,96,96) 或 96×96。源码该行注释 `# (B,h,hd,hd)` 与实际运算不符。

输出还有 `(attn @ v).transpose(1,2).reshape(B,768)`，即先转置再展平的固定维度重排。HSC 输入是 GAP 后经最终 LayerNorm 的全局特征，没有空间 token。

**请回复：** 论文是否按真实实现称为“8 个通道组之间的自注意力，每组 96 维”，并保留上述重排？还是实际训练使用了不同版本的 HSC？若设计原意是每组内部 96×96 注意力，需明确这是意图与实现的差异，不能把未执行的运算写入当前实验的方法。

## Q03｜EGB 的数学描述、“零参数”和抑噪解释需要哪些限定？

**核对到的实现：**

- FFT 在 float32 中进行，低通 mask 使用 `r=0.1*min(H,W)/max(H,W)`，网格为 `linspace(-1,1)`。
- 低频量是 IFFT 幅值的平方；归一化分母是每样本、每通道的**空间方差**加 `1e-5`，不是标准差。
- 高频项是 IFFT 高频幅值相对其空间均值的平方偏离除以空间方差；不是邻域差分或 Sobel 梯度。
- 注意力大小为 (B,C,H,W)，不是通道共享空间图；最终融合为 sigmoid(低频 sigmoid 项 + 高频非负项)。
- EGB/ESA 的**算子核心**无可学习参数，但每个启用块仍有 1 个可学习标量门，已训门含负值。因此“无可学习参数”“无过拟合自由度”“只能强化边缘”均需收窄。
- SCG 四层卷积均带 bias，中间没有激活或归一化；顺序为 5×5 DW → 1×19 DW → 19×1 DW → 1×1。分支理论支持范围是 23×23，若计入此前原有 7×7 DW 则为 29×29 特征网格，不能把 19 简单等同最终有效感受野。

**请回复：** 上述是否均按源码进入 Methods？是否有边缘/噪声可分离性的定量实验或可视化，支持“选择性强化细胞边缘而非荧光噪声”？若无，建议将其保留为设计动机，不写成已证实机制。

## Q04｜CLE 数据量是否把全数据误写成训练集？

**已由当前目录、历史训练日志和评测 JSON 交叉确认：**

| 类别 | train | val | test | 合计 |
|---|---:|---:|---:|---:|
| Benign_lesions | 830 | 195 | 208 | 1233 |
| Malignant_lesions | 371 | 79 | 176 | 626 |
| body_of_stomach_normal | 876 | 153 | 192 | 1221 |
| **合计** | **2077** | **427** | **576** | **3080** |

简报中的“train 3080”和 train 各类“1233/626/1221”，恰好对应全体数据及全体各类总数。CLE-Net seed42 成功训练日志明确打印 2077 training、427 validation；其余本工程日志一致。62 份 test JSON 的 `n_total` 全部为 576。

证据示例：`experiments/logs/convnext_egb_scg_hsc_seed42.log`；附录 B 所列 test JSON；数据解析路径 `/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1`。

**请回复：** 是否确认简报的 3080/593/578 属于拆分记录错误，并将论文统一修正为 2077/427/576、恶性训练样本 371？如果 593/578 对应另一版本数据，请提供版本与运行映射，不能混入当前 4090 表。

## Q05｜归一化实际为 ImageNet，是否取消“数据集自算”表述？

**核对到的事实：** 本工程 29 份 `run_meta.json` 均为 `norm_source=imagenet`；62 份 test JSON 的 `norm_source` 均记录从该类型元数据恢复归一化。实际为：

- mean = [0.485, 0.456, 0.406]
- std = [0.229, 0.224, 0.225]

成功训练日志一致。当前 `train_cle_fair.py` 对加载预训练权重的路径使用 ImageNet 统计量；仅其他条件下走自算流程。“代码具备数据集统计缓存”不能证明这些实验实际用了它。

**请回复：** CLE 主实验是否明确写 ImageNet 归一化？§5.4 heelBone 的“自动重算归一化、零重调参”是否有 run_meta 支持，或也是对代码能力的误述？请提供 heelBone 目标运行的元数据后再裁定跨数据集协议。

## Q06｜验证集和测试集 resize 不同，应如何报告？

**核对到的实际变换：**

| 环节 | Resize | 后续裁剪 |
|---|---|---|
| 训练 | Resize((256,256))，强制正方形 | 随机翻转/旋转后 RandomCrop(224) |
| 训练过程中的验证 | Resize((256,256))，强制正方形 | CenterCrop(224) |
| evaluate.py 的训练后 val/test 评估 | Resize(256)，短边等比缩放 | CenterCrop(224) |

当前 CLE 图像有非正方形尺寸，二者不严格等价。runner 还会用 evaluate.py 评估 val，所以 CSV 中的 val_f1 来自训练后短边缩放评估，不宜直接等同训练过程中用来选 checkpoint 的方形缩放 val macro-F1。`cle_fair_common.py` 的“shared val/test”文字不能覆盖实际调用路径。

**请回复：** 这是预期协议吗？是否可在论文中分别列出，并说明对比模型共享相同测试评测脚本？若有历史统一变换版本，需给出这些 run 对应的版本证据；不能直接改现行代码后认为既有结果自动适用于新协议。

## Q07｜标准差与主表统计采用哪一份数值？

本次统一采用原始 JSON 的 macro-F1，样本标准差 `ddof=1`，最终显示四位小数。九行消融及其余主干的完整复核表在附录 A。

| 配置 | 原始值重算 mean±std | 简报差异 |
|---|---|---|
| baseline，5 seeds | 0.9476±0.0094 | 一致 |
| EGB，3 seeds | 0.9593±0.0029 | 一致 |
| HSC，3 seeds | **0.9534±0.0019** | §9 为 ±0.0020；四位 CSV 先舍入后再聚合会产生差异 |
| SCG+HSC，3 seeds | **0.9483±0.0028** | §9 为 ±0.0029；同上 |
| CLE-Net，6 seeds | **0.9553±0.0044** | §9 一致；§5.3 为 ±0.0043 |
| S-Net，5 seeds | **0.9459±0.0169** | §9 一致；§5.3 为 ±0.0174 |

高精度例：CLE-Net mean=0.955318804276、std=0.004430012089；S-Net mean=0.945894203700、std=0.016865379999；HSC std=0.001939283731；SCG+HSC std=0.002848825404。

**请回复：** 是否统一按上述“未舍入原始 JSON → mean/sample std → 最后舍入”口径？如果保留另一统计口径，请给公式和输入运行列表。另请区分标准差和方差：CLE-Net 相对 baseline 的标准差降低约 **52.97%**，方差降低约 **77.88%**，不能混称。

## Q08｜“完整组合协同最优”“方差最小”“全场最好”的结论边界

**现有数字支持：**

1. 全部可用 seeds 的 CLE-Net mean 0.9553，比 baseline 0.9476 高约 **0.77pp**；在 §5.3 列出的 11 个对比主干中，CLE-Net 均值最高。
2. 但 EGB 单点 mean **0.9593** 高于 CLE-Net **0.9553**，且 std 0.0029 也更小；HSC std 0.0019 小于 CLE-Net 0.0044。因此不能不加范围地写 CLE-Net“所有配置均值第一”或“标准差最小”。
3. A+B 为 0.9457，B+C 为 0.9483，完整 CLE-Net 为 0.9553，可描述组合配置间的性能变化。结合 Q01 的作用阶段变化，尚不能据此证明严格因果协同或加性失效机制。
4. S-Net best **0.9669** 高于本轮指定主干/消融表中其他条目；本工程 CSV 还含未纳入该表的早期其他变体，例如 `convnext_wds_wtb_k3_msf` 的 batchU 记录为 0.9721。这条历史记录的硬件及协议可比性未在本轮完整复核，不纳入当前排名；它提示需要明确比较集合。“全场最高”建议限定为本次指定主干/消融表。
5. 当前种子数不同。共同 seeds42–46 上，CLE-Net **0.9555±0.0049**、baseline **0.9476±0.0094**，均值差 **+0.78pp**，5 个种子中 4 个提升。共同 seeds42–44 上，两者为 **0.9544±0.0060** 与 **0.9489±0.0122**，差 **+0.55pp**。这些是描述性比较，尚未做显著性推断。

**请回复：** 最终是否保留 CLE-Net 为主方法、EGB 为最强单点、S-Net 为指定比较集合的 best 辅助结果，同时如实写出完整网络未超过 EGB 单点？“synergy over additivity”是否降为解释假设，而非定论？主表保留全部运行，并加共同种子比较是否符合预期？

## Q09｜请提供 heelBone 的目标实验原始文件

简报 §5.4 的目标数据是：

| 方法 | test macro-F1（按简报） | best | mean±std（按简报） |
|---|---|---|---|
| S-Net | 0.8972 / 0.9035 / 0.8278 | 0.9035 | 0.8762±0.0420 |
| ConvNeXt-T | 0.8886 / 0.8957 / 0.8641 | 0.8957 | 0.8828±0.0165 |
| CLE-Net | 0.8504 | 0.8504 | — |
| Swin-T | 0.8366 | 0.8366 | — |
| ResNet34 | 0.8149 | 0.8149 | — |

本服务器当前 heelBone 图像计数 **2770/593/593** 可确认；但未找到上述同一组实验的原始 results.json。已看到的 `ZCode/Classification/experiments/results_heelBone.csv`、`Network/SF-ConvNeXt/reports/09_heelbone_cross_domain.md` 和 `10_heelbone_results_raw.csv` 属于另一批实验（其基线为 0.8952/0.8827/0.8605），不能替换。

**请回复并提供：**

1. §5.4 目标实验的服务器/项目绝对路径、逐 run results.json、run_meta.json、作业表；如方便，请将相关小型结果文件同步到本项目供下一轮核对。当前未连接 n1。
2. 是否在 heelBone 上从 ImageNet 权重独立训练相同结构？还是从 CLE 训练后权重迁移？“跨数据集泛化”不能误写成未训练的零样本跨域测试。
3. 是否确为同一组 seeds42/43/44、相同超参及归一化？“零重调参”具体包含哪些字段？
4. 基线 ±0.0165 的原始精度依据是什么？仅用表内三项四位小数重算为 ±0.0166，需原始 JSON 决定，不应凭已舍入数字擅自裁定。
5. S-Net best 相对基线 +0.78pp，但 mean **低0.66pp**；CLE-Net 只有一个运行。是否同意只能据此报告潜力和种子敏感性，不能称均值泛化全面占优或已证明 EGB 模态特异性的因果机制？

另：当前 heelBone train 中保存模式为灰度 L 的图像是 1711/2770（约61.77%），并观察到 1011×1011 图像。简报的“53%灰度、464–1007px”统计范围不明，建议提供画像脚本/样本范围，或暂不写入论文。

## Q10｜A800 辅助证据中的指标冲突与 EGB 环境归因

**摘要间存在冲突：**

- `experiments/FINAL_REPORT_n1.md` 和简报部分段落给出 CLE-Net/baseline Balanced Accuracy **0.9833/0.9817**；`experiments/COMPARE_TABLE.md` 给出 **0.9662/0.9632**。
- `evaluate.py` 的 balanced accuracy 等于各类 recall 的均值，也等于 macro sensitivity，不含 specificity 的平均项。不能把不同定义混列。
- FINAL_REPORT 的恶性敏感度 **0.9663/0.9567**，在 CSV 类别列中对应良性敏感度；CSV 与 COMPARE_TABLE 的恶性敏感度实际为 **0.9375/0.9432**。
- 简报称 CLE-Net seed43 恶性敏感度 0.9615，但对应 n1b4 CSV 行的 `sens_malignant` 为 **0.9034**，0.9615 位于 `sens_benign`。

**请回复：** 请提供 A800 对应原始 results.json，确认 BA 与逐类指标最终值。当前应优先引用 CSV 的明确类别字段，并将不能完成原始复核的辅助指标标为 [待核对]。

另，v2 将 A800 EGB 表现归因为“五路并发超卖环境伪影”。CSV 中 EGB seed42 为 n1b1、seed43 为 n1b2；GPU util 较低本身不足以证明数值性能下降的因果链。请给出该归因的日志、调度对照或复测证据，区分“已验证环境污染”与“可能受负载/随机性等影响”。4090 EGB 强的结论成立，但不能自动证明另一机器低分的唯一成因。

## Q11｜类别错误分析与“最难类同步提升”能否收窄？

以下均来自各模型 test macro-F1 最优运行的原始 JSON；它们是 best 运行对照，不是跨种子均值。

| 方法 | seed | Macro-F1 | 恶性敏感度 | B↔M 误分类数 | 全部错误数 |
|---|---:|---:|---:|---:|---:|
| ConvNeXt-T | 44 | 0.9615 | 0.9489 | 15 | 22 |
| CLE-Net | 46 | 0.9599 | 0.9545 | 17 | 23 |
| S-Net | 44 | 0.9669 | 0.9432 | 15 | 19 |

S-Net 的 best macro-F1 最高，但恶性敏感度低于该基线和 CLE-Net。另，CLE-Net 六个 seed 的 B↔M 数为 14/13/19/15/17/14，并非所有模型都落在简报的 12–18 张区间。

**请回复：** 错误分析是否限定到具体模型、种子和类别，避免“各指标同步提升”“所有模型12–18张”之类过宽表述？若已有跨种子的逐类均值，请提供明确统计表。

## Q12｜参数量、初始化等价与预训练兼容的最终口径

按三分类模型 `sum(p.numel() for p in model.parameters())`：

| 配置 | 精确参数量 |
|---|---:|
| baseline | 27,822,435 |
| +EGB | 27,822,450 |
| +ESA | 27,822,447 |
| +SCG | 31,304,943 |
| +HSC | 30,182,500 |
| EGB+SCG | 31,304,946 |
| SCG+HSC | 33,665,008 |
| CLE-Net / S-Net | 33,665,011 |

完整网络增量为 **5,842,576（约21.0%）**。`results.json` 的 `complexity.params_raw` 来自 THOP；baseline 仅计 27,801,027，漏计 21,408（自定义 LayerNorm 与 layer-scale），新增标量门也未计。FLOPs 对 FFT 等算子的覆盖同样需说明，不能直接视为完整算术量。

九种配置复制相同 baseline 参数和三分类头后，CPU float32、eval、单张224输入下，零门初始化的最大 logit 差均为 **0.0**。这说明该条件下的初始等价，不保证训练后性能不下降。29 份本工程元数据记录骨干参数完整加载、unexpected_keys为空；原 ImageNet 1000类分类头被删除并重新初始化三分类头。

**请回复：** 参数表是否统一采用实际参数总数（可显示27.8M/33.7M），并注明与 THOP 口径不同？预训练描述是否用“原骨干可训练参数完整加载”，零门描述是否限定为“相同原始参数下初始等价”，避免“零过拟合风险/训练无损保证”？

## Q13｜可复现协议、种子与同机公平性的剩余限定

本工程代码、runner、日志和元数据支持：AdamW，lr=1e-4，weight decay=.05，batch32，最多300轮，CosineAnnealingLR(T_max=300)，无 warmup，unweighted CrossEntropy、label_smoothing=0，CUDA AMP+GradScaler，val macro-F1 严格提升时保存，30轮无提升早停。

仍需确认的文字边界：

- 数据、指标代码与预训练来源可复核；两个工程数据软链指向同一路径，evaluate.py 差异仅模型注册行，解释器均指向 sunfuxiang，代表运行的预训练 SHA256 相同：`14f3164e3ea6ac32ab3f574f528ce817696c9176fad4221e0a77a905a7360595`。但当前 CLE runner 的 OMP/MKL 线程为2，ZCode为4；补测也经历并发1→2。因此“学习和评测协议一致”与“全部执行环境完全相同”需要区分。
- set_seed 设置多个随机源及 cuDNN 确定性标志，但 `utils.read_split_*` 内部又执行 `random.seed(0)`；未启用全局 deterministic algorithms。不能写跨硬件逐位确定性。
- evaluate.py 支持可选图像级 bootstrap，但这些 runner 未启用 `--bootstrap`。种子标准差不等于95%置信区间。
- 记录显示曾依据 test 错误和 test 分数进行架构探索、补种子和冲刺。因此固定 test 集参与了开发反馈，不能不加说明地写成完全未参与设计的最终盲测集。

**请回复：** 是否按这些实际协议与限制撰写？是否已有未参与探索的新测试集、患者级重复评估或正式统计检验可补充？若无，建议保留局限，不宣称显著优于或临床有效。

## Q14｜数据画像、患者划分与参考文献的证据在哪里？

以下信息尚不足以从当前文件独立确认，请提供来源或允许保留 [待核对]：

1. CLE 每类随机40张画像的脚本、抽样种子/名单、亮度0.23–0.28、σ≈0.22及“中心16×16外高频能量>85%”的计算定义（幅值还是平方幅值、是否去均值/包含DC、resize范围）。这些统计不能仅凭摘要推广为去噪机制的证据。
2. 数据的患者/病例数、患者ID与split映射、患者级划分及同一视频帧是否跨split、采集来源和伦理信息。当前代码读取既有目录；注释写patient-level不能代替划分证据。
3. PFESA、StripConv、SCSA对应的准确论文标题、作者、公开链接或DOI，以及当前实现借鉴和改动的边界。现阶段未核对“MICCAI2026/AAAI2026”的具体出处，不宜凭缩写生成参考文献。
4. OS/DSH/PLD失败路线的可用实现与对应运行。单次 test 回落可以报告，但“预训练丢失导致下降”“原型记忆分布”等因果解释若无对照，只应作为假设。

---

## 建议回复格式

请按 Q01–Q14 回复，每项使用：

> **Qxx：确认 / 纠正 / 暂无法确认**  
> 最终事实或解释：……  
> 证据：绝对路径、运行名、commit、字段或关键日志。  
> 建议论文表述：……  
> 是否仍需 [待核对]：……  
> 如需新实验：单独列建议，不与既有结果混写。

收到回复后，Codex 可先核实新证据，再撰写中文方法与实验两章；主结果继续遵循4090优先、均值与标准差为主的既定方向。

## 附录 A｜4090 原始 JSON 聚合结果

本表由62份原始test JSON计算；std为样本标准差，单次运行不计算std。它是核对表，不构成论文排名结论。

| 注册名 | 来源 | n | seeds | mean±std | best | best seed |
|---|---|---:|---|---|---:|---:|
| convnext_tiny | ZCode | 5 | 42,43,44,45,46 | 0.9476±0.0094 | 0.9615 | 44 |
| convnext_egb | CLE-Net | 3 | 42,43,44 | 0.9593±0.0029 | 0.9619 | 42 |
| convnext_esa | CLE-Net | 3 | 42,43,44 | 0.9541±0.0047 | 0.9589 | 43 |
| convnext_scg | CLE-Net | 3 | 42,43,44 | 0.9486±0.0148 | 0.9617 | 42 |
| convnext_hsc | CLE-Net | 3 | 42,43,44 | 0.9534±0.0019 | 0.9547 | 44 |
| convnext_egb_scg | CLE-Net | 3 | 42,43,44 | 0.9457±0.0078 | 0.9546 | 42 |
| convnext_scg_hsc | CLE-Net | 3 | 42,43,44 | 0.9483±0.0028 | 0.9514 | 44 |
| convnext_egb_scg_hsc | CLE-Net | 6 | 42,43,44,45,46,47 | 0.9553±0.0044 | 0.9599 | 46 |
| convnext_esa_scg_hsc | CLE-Net | 5 | 42,43,44,45,46 | 0.9459±0.0169 | 0.9669 | 44 |
| convnext_small | ZCode | 3 | 42,43,44 | 0.9538±0.0052 | 0.9594 | 44 |
| resnet50 | ZCode | 3 | 42,43,44 | 0.9462±0.0039 | 0.9507 | 42 |
| swin_tiny_vanilla_patch4_window7_224 | ZCode | 3 | 42,43,44 | 0.9419±0.0127 | 0.9512 | 44 |
| mobilenet_v3_large | ZCode | 3 | 42,43,44 | 0.9408±0.0078 | 0.9494 | 42 |
| fast_t0 | ZCode | 3 | 42,43,44 | 0.9360±0.0071 | 0.9440 | 43 |
| resnet34 | ZCode | 3 | 42,43,44 | 0.9322±0.0096 | 0.9431 | 43 |
| ghostnet | ZCode | 3 | 42,43,44 | 0.9276±0.0152 | 0.9400 | 43 |
| pvt_tiny | ZCode | 3 | 42,43,44 | 0.9080±0.0283 | 0.9394 | 43 |
| vit_base_patch16_224 | ZCode | 3 | 42,43,44 | 0.9171±0.0159 | 0.9311 | 43 |
| repvit_m1 | ZCode | 1 | 42 | 0.7822（单次） | 0.7822 | 42 |

## 附录 B｜已核对文件与运行清单

### B.1 源码、协议与报告

以下未另注绝对路径者均相对项目根目录 `/home/login/data/cle/auto/Network/CLE-Net`：

- `CODEX_BRIEFING.md` v2；`QODER_REPORT.md`。
- `models/Convnext.py`；`models/optimize/convnext_cle2.py`；`models/optimize/convnext_cle5.py`，及后两者 Git 历史。
- `train_cle_fair.py`；`train_utils.py`（统计缓存相关部分）；`evaluate.py`（指标、变换、复杂度、bootstrap等部分）。
- `cle_fair_common.py`；`utils.py`（split读取）；`my_dataset.py`。
- `experiments/runner.py`；`experiments/jobs_4090_cle.json`、`jobs_4090_cle2.json`、`jobs_4090_v5a.json`、`jobs_4090_v5b.json`、`jobs_qoder_abl.json`、`jobs_n1b1.json`、`jobs_n1b2.json`（以上jobs文件均在experiments内）。
- `experiments/results_cle_ac_v1.csv`；`experiments/FINAL_REPORT_n1.md`；`experiments/COMPARE_TABLE.md`；`reports/CLE-Net_4090_实测对比报告.md`；`reports/QODER_消融总表_4090.md`。
- `weights_cle_ac_v1/*/*/run_meta.json`，本工程29份；`experiments/logs/<model>_seed<seed>.log`，对应29个run的关键训练配置及计数行。
- 四份实际已训权重：EGB、ESA、CLE-Net、S-Net seed42 各自的 `weights_cle_ac_v1/<model>_seed42/<model>/model_best.pth`。
- `/home/login/data/cle/auto/ZCode/Classification` 中对应的 `evaluate.py`、`experiments/runner.py`、代表baseline的 `run_meta.json`、test结果及数据软链。
- Q09列明的三份heelBone其他批次记录，已读以排除混用；目标§5.4原始结果尚未取得。
- CLE与heelBone数据目录的各split逐类图像计数；训练图像的保存模式与尺寸检查。

### B.2 62 份 test results.json

表中前缀：

- **C** = `/home/login/data/cle/auto/Network/CLE-Net`
- **Z** = `/home/login/data/cle/auto/ZCode/Classification`

“相对路径”拼接相应前缀即为读取的原始文件。表内macro-F1显示6位，聚合使用JSON完整精度。

| 前缀 | 模型 / seed | macro-F1 | 相对路径 |
|---|---|---:|---|
| Z | convnext_tiny / 42 | 0.947926 | weights_cle_ac_v1/convnext_tiny_seed42/convnext_tiny/eval_20261025_183058/results.json |
| Z | convnext_tiny / 43 | 0.937273 | weights_cle_ac_v1/convnext_tiny_seed43/convnext_tiny/eval_20261025_183228/results.json |
| Z | convnext_tiny / 44 | 0.961543 | weights_cle_ac_v1/convnext_tiny_seed44/convnext_tiny/eval_20261025_183047/results.json |
| Z | convnext_tiny / 45 | 0.940894 | weights_cle_ac_v1/convnext_tiny_seed45/convnext_tiny/eval_20261025_213541/results.json |
| Z | convnext_tiny / 46 | 0.950568 | weights_cle_ac_v1/convnext_tiny_seed46/convnext_tiny/eval_20261025_213624/results.json |
| C | convnext_egb / 42 | 0.961857 | weights_cle_ac_v1/convnext_egb_seed42/convnext_egb/eval_20261027_134209/results.json |
| C | convnext_egb / 43 | 0.956241 | weights_cle_ac_v1/convnext_egb_seed43/convnext_egb/eval_20261027_135545/results.json |
| C | convnext_egb / 44 | 0.959922 | weights_cle_ac_v1/convnext_egb_seed44/convnext_egb/eval_20261027_135519/results.json |
| C | convnext_esa / 42 | 0.953719 | weights_cle_ac_v1/convnext_esa_seed42/convnext_esa/eval_20261027_142813/results.json |
| C | convnext_esa / 43 | 0.958878 | weights_cle_ac_v1/convnext_esa_seed43/convnext_esa/eval_20261027_143129/results.json |
| C | convnext_esa / 44 | 0.949559 | weights_cle_ac_v1/convnext_esa_seed44/convnext_esa/eval_20261027_143240/results.json |
| C | convnext_scg / 42 | 0.961688 | weights_cle_ac_v1/convnext_scg_seed42/convnext_scg/eval_20261026_170901/results.json |
| C | convnext_scg / 43 | 0.932603 | weights_cle_ac_v1/convnext_scg_seed43/convnext_scg/eval_20261026_171307/results.json |
| C | convnext_scg / 44 | 0.951480 | weights_cle_ac_v1/convnext_scg_seed44/convnext_scg/eval_20261027_143423/results.json |
| C | convnext_hsc / 42 | 0.951147 | weights_cle_ac_v1/convnext_hsc_seed42/convnext_hsc/eval_20261027_140144/results.json |
| C | convnext_hsc / 43 | 0.954260 | weights_cle_ac_v1/convnext_hsc_seed43/convnext_hsc/eval_20261027_140135/results.json |
| C | convnext_hsc / 44 | 0.954707 | weights_cle_ac_v1/convnext_hsc_seed44/convnext_hsc/eval_20261027_140504/results.json |
| C | convnext_egb_scg / 42 | 0.954627 | weights_cle_ac_v1/convnext_egb_scg_seed42/convnext_egb_scg/eval_20261027_141205/results.json |
| C | convnext_egb_scg / 43 | 0.940109 | weights_cle_ac_v1/convnext_egb_scg_seed43/convnext_egb_scg/eval_20261027_141433/results.json |
| C | convnext_egb_scg / 44 | 0.942409 | weights_cle_ac_v1/convnext_egb_scg_seed44/convnext_egb_scg/eval_20261027_141828/results.json |
| C | convnext_scg_hsc / 42 | 0.947776 | weights_cle_ac_v1/convnext_scg_hsc_seed42/convnext_scg_hsc/eval_20261027_141859/results.json |
| C | convnext_scg_hsc / 43 | 0.945720 | weights_cle_ac_v1/convnext_scg_hsc_seed43/convnext_scg_hsc/eval_20261027_142343/results.json |
| C | convnext_scg_hsc / 44 | 0.951350 | weights_cle_ac_v1/convnext_scg_hsc_seed44/convnext_scg_hsc/eval_20261027_142643/results.json |
| C | convnext_egb_scg_hsc / 42 | 0.958114 | weights_cle_ac_v1/convnext_egb_scg_hsc_seed42/convnext_egb_scg_hsc/eval_20261026_161838/results.json |
| C | convnext_egb_scg_hsc / 43 | 0.957593 | weights_cle_ac_v1/convnext_egb_scg_hsc_seed43/convnext_egb_scg_hsc/eval_20261026_162358/results.json |
| C | convnext_egb_scg_hsc / 44 | 0.947444 | weights_cle_ac_v1/convnext_egb_scg_hsc_seed44/convnext_egb_scg_hsc/eval_20261026_162642/results.json |
| C | convnext_egb_scg_hsc / 45 | 0.954265 | weights_cle_ac_v1/convnext_egb_scg_hsc_seed45/convnext_egb_scg_hsc/eval_20261026_163545/results.json |
| C | convnext_egb_scg_hsc / 46 | 0.959947 | weights_cle_ac_v1/convnext_egb_scg_hsc_seed46/convnext_egb_scg_hsc/eval_20261026_163905/results.json |
| C | convnext_egb_scg_hsc / 47 | 0.954549 | weights_cle_ac_v1/convnext_egb_scg_hsc_seed47/convnext_egb_scg_hsc/eval_20261026_172758/results.json |
| C | convnext_esa_scg_hsc / 42 | 0.947133 | weights_cle_ac_v1/convnext_esa_scg_hsc_seed42/convnext_esa_scg_hsc/eval_20261026_165655/results.json |
| C | convnext_esa_scg_hsc / 43 | 0.919829 | weights_cle_ac_v1/convnext_esa_scg_hsc_seed43/convnext_esa_scg_hsc/eval_20261026_170303/results.json |
| C | convnext_esa_scg_hsc / 44 | 0.966916 | weights_cle_ac_v1/convnext_esa_scg_hsc_seed44/convnext_esa_scg_hsc/eval_20261026_170635/results.json |
| C | convnext_esa_scg_hsc / 45 | 0.949754 | weights_cle_ac_v1/convnext_esa_scg_hsc_seed45/convnext_esa_scg_hsc/eval_20261026_172038/results.json |
| C | convnext_esa_scg_hsc / 46 | 0.945838 | weights_cle_ac_v1/convnext_esa_scg_hsc_seed46/convnext_esa_scg_hsc/eval_20261026_172429/results.json |
| Z | convnext_small / 42 | 0.949190 | weights_cle_ac_v1/convnext_small_seed42/convnext_small/eval_20261025_184148/results.json |
| Z | convnext_small / 43 | 0.952766 | weights_cle_ac_v1/convnext_small_seed43/convnext_small/eval_20261025_210005/results.json |
| Z | convnext_small / 44 | 0.959394 | weights_cle_ac_v1/convnext_small_seed44/convnext_small/eval_20261025_210212/results.json |
| Z | resnet50 / 42 | 0.950725 | weights_cle_ac_v1/resnet50_seed42/resnet50/eval_20261026_141708/results.json |
| Z | resnet50 / 43 | 0.943926 | weights_cle_ac_v1/resnet50_seed43/resnet50/eval_20261026_141404/results.json |
| Z | resnet50 / 44 | 0.943924 | weights_cle_ac_v1/resnet50_seed44/resnet50/eval_20261026_141621/results.json |
| Z | swin_tiny_vanilla_patch4_window7_224 / 42 | 0.947112 | weights_cle_ac_v1/swin_tiny_vanilla_patch4_window7_224_seed42/swin_tiny_vanilla_patch4_window7_224/eval_20261025_183633/results.json |
| Z | swin_tiny_vanilla_patch4_window7_224 / 43 | 0.927408 | weights_cle_ac_v1/swin_tiny_vanilla_patch4_window7_224_seed43/swin_tiny_vanilla_patch4_window7_224/eval_20261025_205823/results.json |
| Z | swin_tiny_vanilla_patch4_window7_224 / 44 | 0.951249 | weights_cle_ac_v1/swin_tiny_vanilla_patch4_window7_224_seed44/swin_tiny_vanilla_patch4_window7_224/eval_20261025_210413/results.json |
| Z | mobilenet_v3_large / 42 | 0.949442 | weights_cle_ac_v1/mobilenet_v3_large_seed42/mobilenet_v3_large/eval_20261026_142315/results.json |
| Z | mobilenet_v3_large / 43 | 0.938797 | weights_cle_ac_v1/mobilenet_v3_large_seed43/mobilenet_v3_large/eval_20261026_142320/results.json |
| Z | mobilenet_v3_large / 44 | 0.934186 | weights_cle_ac_v1/mobilenet_v3_large_seed44/mobilenet_v3_large/eval_20261026_142931/results.json |
| Z | fast_t0 / 42 | 0.933493 | weights_cle_ac_v1/fast_t0_seed42/fast_t0/eval_20261025_212035/results.json |
| Z | fast_t0 / 43 | 0.944023 | weights_cle_ac_v1/fast_t0_seed43/fast_t0/eval_20261025_212311/results.json |
| Z | fast_t0 / 44 | 0.930553 | weights_cle_ac_v1/fast_t0_seed44/fast_t0/eval_20261025_212422/results.json |
| Z | resnet34 / 42 | 0.928916 | weights_cle_ac_v1/resnet34_seed42/resnet34/eval_20261025_184138/results.json |
| Z | resnet34 / 43 | 0.943075 | weights_cle_ac_v1/resnet34_seed43/resnet34/eval_20261025_210319/results.json |
| Z | resnet34 / 44 | 0.924700 | weights_cle_ac_v1/resnet34_seed44/resnet34/eval_20261025_210422/results.json |
| Z | ghostnet / 42 | 0.932084 | weights_cle_ac_v1/ghostnet_seed42/ghostnet/eval_20261026_143245/results.json |
| Z | ghostnet / 43 | 0.940006 | weights_cle_ac_v1/ghostnet_seed43/ghostnet/eval_20261026_143249/results.json |
| Z | ghostnet / 44 | 0.910699 | weights_cle_ac_v1/ghostnet_seed44/ghostnet/eval_20261026_143359/results.json |
| Z | pvt_tiny / 42 | 0.884572 | weights_cle_ac_v1/pvt_tiny_seed42/pvt_tiny/eval_20261025_183708/results.json |
| Z | pvt_tiny / 43 | 0.939421 | weights_cle_ac_v1/pvt_tiny_seed43/pvt_tiny/eval_20261025_210712/results.json |
| Z | pvt_tiny / 44 | 0.899883 | weights_cle_ac_v1/pvt_tiny_seed44/pvt_tiny/eval_20261025_211134/results.json |
| Z | vit_base_patch16_224 / 42 | 0.899812 | weights_cle_ac_v1/vit_base_patch16_224_seed42/vit_base_patch16_224/eval_20261025_211623/results.json |
| Z | vit_base_patch16_224 / 43 | 0.931119 | weights_cle_ac_v1/vit_base_patch16_224_seed43/vit_base_patch16_224/eval_20261025_211829/results.json |
| Z | vit_base_patch16_224 / 44 | 0.920250 | weights_cle_ac_v1/vit_base_patch16_224_seed44/vit_base_patch16_224/eval_20261025_212047/results.json |
| Z | repvit_m1 / 42 | 0.782157 | weights_cle_ac_v1/repvit_m1_seed42/repvit_m1/eval_20261025_212717/results.json |
