# CLE 自动研究账本

更新：2026-10-24 10:49（北京时间）。前八阶段完成，尚未确认新增结构贡献。第八阶段固定二阶统计未达跟进标准；第九阶段已启动冻结独立预训练表征对照，零新增神经网络训练。

## 数据研究已完成

患者分组 5 折 × 3 种子得到训练图像 A–E 等级，完成全量、A–C、等级加权三个训练对照。完整 test 的 A–C 均值为 0.8984，全量为 0.8722。等级反映模型与现有标签的一致性，不能视为人工确认的正确标签。恶性类别的筛除率明显更高，后续保留完整 val/test。

这些 test 结果已经使用过，因此后续结构选择只依据开发集。第二轮采用等级加权训练，是因为原数据对照的验证集均值为 0.8820，高于 A–C 的 0.8724；不是依据 test 排名选择。

## 第一轮结构研究已完成：18 次训练

对照普通卷积、原 LSConv 算法语义和 CLE 局部形态—上下文改造，分别使用 A–C、等级加权数据和三个配对种子。所有单元均保存配置、checkpoint 哈希、逐图预测和支路开关诊断。

| 训练数据 | 模块 | 验证集 F1 均值 | 相对对应基线的变化（百分点） | 患者配对区间（百分点） |
|---|---|---:|---:|---|
| A–C | 普通卷积 | 0.8735 | +0.11 | [-1.24, +1.53] |
| A–C | LSConv 参考 | 0.8754 | +0.29 | [-1.63, +2.25] |
| A–C | CLE 改造 | 0.8685 | -0.39 | [-2.20, +1.63] |
| 等级加权 | 普通卷积 | 0.8744 | -0.76 | [-3.19, +1.26] |
| 等级加权 | LSConv 参考 | 0.8819 | -0.002 | [-2.04, +1.69] |
| 等级加权 | CLE 改造 | 0.8732 | -0.87 | [-2.28, +0.39] |

区间全部跨零，未检出稳定收益。支路实际扰动约 42%–64%，因此不能把本轮阴性结果解释为门控没有打开。区间以训练好的模型为条件，没有计入整个训练、选择流程的不确定性，也未进行多重比较校正。

结果提交：`b32e225`。静态模型的池化诊断脚本提交：`0e1c0c9`。诊断发现直接使用高阶 GeM 或极端区域响应一般降低验证表现；多数错误仍含正确类别的局部响应，但这些响应的空间感受野重叠，不能据此声称具有病理区域标注。

## 第二轮已完成：有全局约束的空间证据聚合

预注册的不同机制：从改变卷积特征转向改变全局聚合，至少保留一半 GAP 描述子；局部小波高频只是可学习正负方向的提示，不把高频强度直接等同于病变。

六个方案：GeM、参数匹配的普通门控注意力、完整 CLE 证据聚合、去小波、固定融合系数、去全局上下文；每项种子 42/43/44，共 18 次训练。完整模型新增 180,993 参数，普通注意力新增 181,136 参数。患者映射、训练来源和完整验证集不变。本轮不运行 test。

2026-10-24 05:11:38 启动独立监督进程 PID 672103。并发 2，每进程 CPU 线程 2。首批训练已输出正常 epoch 日志。

预检查通过：Haar 能量守恒与奇数尺寸梯度、AMP 有限更新、注意力归一化、GAP 占比下界、频率系数非零梯度、同种子原始主干/分类头完全一致、关闭新增模块与原主干逐位一致、真实图像 CUDA 反向传播。

代码与预注册：`72d84ae`；冻结配置与复用基线：`14087cf`；预检查：`bddd615`。状态：`experiments/pooling_study/status.json`；日志：`experiments/pooling_study/logs/pipeline.log`；入口：`experiments/run_pooling_study.py`。

18 项全部完成，GAP 基线 0.881958，完整设计 0.878639，去小波 0.882177；各方案患者配对区间均跨零。结果提交 `3527f1d`，独立完成审计 `debed64`；中文解读见 `experiments/pooling_study/ROUND2_RESULTS_ZH.md`。checkpoint 哈希、60 个 epoch、538 张完整验证预测与患者隔离均核验通过。

## 第三轮已完成：共享卷积的分辨率路径与对比度路由

错误集中于困难亚型：按文件名统计，基线 dysplasia 平均错误率约 34%、cancer 6.6%、normal 1.7%。文件名存在与当前类别不一致的情况，不以此改标签或追加亚型监督。这只能定位研究问题，不能确认噪声。原图约 800×800；本轮保持既有 224 输入，仅检验 layer4 的空间采样与尺度融合。

六项：单纯高分辨率空洞路径、固定双路径融合、匹配 SK 式路由、CLE 对比度条件路由、去空洞、静态路由。各三种子，共 18 次。两路径共享卷积，双路径采用独立 BN。完整设计与 SK 对照新增参数均为 106,560，计数的卷积/矩阵乘法 FLOPs 相同；高分辨率与双路径有额外推理计算开销，不宣称零成本。

预注册与代码 `be9c095`，冻结配置 `d0cd4a3`，预检查 `32f8d5c`。功能卷积与独立 Conv2d 参考的前向/梯度相符；配对初始化、AMP、实际路径梯度、初始正权重及匹配预算均通过。2026-10-24 05:55:35 独立监督进程 PID 678152 启动。入口 `experiments/run_scale_study.py`，状态与日志在 `experiments/scale_study/`。

强晋级要求完整设计相对 GAP 三种子均正、均值至少 +1.0pt 且患者区间下界大于零；相对固定融合和匹配 SK 的均值均至少 +0.5pt 且区间下界大于零。较弱信号保留，不宣称确认。即使晋级也需要独立患者级确认，不能凭反复使用的开发集宣布论文贡献。

18 项均完成，完整设计 0.879506，GAP 0.881958，匹配 SK 0.881109，全部患者比较区间跨零，未晋级。结果 `ac6f236`，审计 `3682b3d`。实际高路权重约 0.500/0.632/0.503，但描述子变化约 4%，提示两路池化特征较接近，不证明任何机制普遍无效。

输入尺度诊断固定六个既有模型，不训练、不用 test。原 GAP 从 224 的 0.88196 降到 336 的 0.77251、448 的 0.46731；空洞控制同样退化。逐张复核全部 538 张原尺度输入与冻结缓存逐位一致。源码 `c8d5f3b`，结果与复核 `f9a07a9`。这说明固定模型尺度敏感，不能估计高分辨率重训练是否有效。中文解释与逐图诊断已交付 `outputs/CLE_architecture_study/round3/`。

## 第四轮已完成：同源输入的全局—局部联合网络

仅缓存原 train+val 共 2924 张图为 512 像素，明确排除 test。各方案共用 448 裁剪。六项为同源 global224、标准 single448、固定多视图、普通 gated-MIL 式注意力、全局 query 条件聚合、去全局 query；各三个种子，共 18 次。主要基线是新训同源 global224；旧 256/224 基线只作背景，避免将预处理变化误归给模块。

局部图为同一 448 视野的四块非重叠 224 原始像素。卷积与分类头共享，局部 BN 独立，不对局部图赋图像标签作为单独监督。可学融合至少保留一半全局特征。完整模型与普通注意力新增参数 148672/148736，计数 FLOPs 相差远小于 1%；五视图计算约为全局单视图五倍，不能声称零推理成本。真实批量 32 AMP 反向峰值显存约 3.0 GiB，支持并发 2。

预注册与代码 `967ba74`，冻结配置 `c693123`，预检查 `8b0864e`。已验证缓存哈希/范围、切块重组、初始化、关闭等价、全局及全部局部描述子梯度、独立 BN 统计、注意力梯度和预算。2026-10-24 06:56:38 启动监督进程 PID 685886。入口 `experiments/run_multiview_study.py`，状态/日志位于 `experiments/multiview_study/`；预注册 `experiments/ROUND4_PROTOCOL.md`。

强晋级要求完整设计相对新同源 global224 三种子均正，均值至少 +1.0pt 且患者区间下界正；相对固定融合及普通注意力均值各至少 +0.5pt 且区间下界正；相对 single448 均值为正。否则保留阴性/弱信号，不能将更高分辨率本身的收益宣称为新路由贡献。独立患者级确认要求不变。

18 项完成，完整设计 0.880436、同源 global224 0.883140、single448 0.880061、固定融合 0.885287。区间全部跨零，未晋级。结果 `70cf9fd`，独立审计 `d040cd1`；缓存/逐图权重/独立 BN 均正常。局部融合系数从 0.25 降至约 0.083，描述子变化约 3.7%，不能据此证明局部细节无效。报告、审计和完整逐图预测已复制 `outputs/CLE_architecture_study/round4/`。

## 第五轮已完成：训练目标与患者贡献

在前四轮没有可靠结构收益后，转查监督和患者分布前提。训练 185 个患者组图数范围 1–103，最多的 10 组占 667/2386≈28%。本轮同一普通 ResNet34/224，共六项：full_ce、grade_ce、label_smooth、GCE(q=.7)、ELR(β=.7/λ=3)、patient_balance。两个既有 CE 对照的六个模型已核验签名、SHA256和验证分数，直接复用；仅新增四项×三个种子=12 次训练，不重复有效模型。

患者权重在每个原类别内均衡患者贡献并保持该类总权重不变，权重范围约 0.09–9.4，按类别有效样本量约 319/187/254；这些权重不代表临床可靠性。GCE 不截断样本；ELR 仅增加时间一致性正则，原标签始终用于 CE，不输出替换标签。均保留完整训练图和完整 val/test，不评估 test。

预注册/代码 `f1ad52d`，冻结配置 `22d086b`，预检查 `fddb388`。已验证 GCE 解析梯度/CE 极限、ELR 官方公式与仅当前训练索引更新、真实图像 AMP 梯度、患者贡献约束以及新 CE 训练步与原实现参数更新逐位一致。ELR 原代码为 MIT，附来源与许可。2026-10-24 08:43:06 启动监督进程 PID 694841，入口 `experiments/run_robust_study.py`，状态与日志 `experiments/robust_study/`。原四轮和数据研究冻结文件不变。

强信号需相对 full_ce 三种子均正、均值至少 +1.0pt 且患者区间下界正；相对 grade_ce 均值至少 +0.5pt 且区间下界正。欲归因于噪声适应，GCE/ELR 还须超过 label_smooth 至少 +0.5pt 且区间下界正。即使通过也不证明标签错误，需要独立患者级确认。若只有平滑或患者平衡有效，应据实归为通用正则或患者分布因素。

12 项新训练完成，另复用 6 个原模型。GCE 0.877132、ELR 0.876379、标签平滑 0.866266、患者平衡 0.857841；均没有超过等级加权基线 0.881958，也未达到预注册晋级标准。结果 `4fa76f1`，独立审计和中文解读 `03b0e68`。三份 ELR 辅助状态只含 2386 个训练条目、每图 visits=60、有限值与哈希正确。完整运行记录与六基线预测已交付 round5。

训练末期准确率约 99%–100%，但最后验证分数全部低于最佳轮次；29 张验证图至少被 15/18 个模型判错。这提示泛化约束及难例复核值得研究，不能直接证明标签错误或预训练知识遗忘。ELR 恶性召回增加伴随更多良性被判恶性，不能单独当作整体收益。

## 第六轮已完成：预训练主干更新约束

四种成熟训练约束：仅分类头（1539 可训练参数）、仅 layer4+分类头（13115907）、冻结所有 BN 运行统计但训练全部参数、预训练起点距离惩罚（后两项均 21286211）。所有新项统一采用既有等级加权食谱、原 256/224 缓存/增强、2386 训练图和完整 538 验证图；没有新推理模块，不评 test，不改标签。起点约束 .01/2×非 fc 参数偏移平方和，保留原 AdamW，是 L2-SP 思想的独立实现附加约束，非原论文精确复现或新颖结构。

12 次新增训练，复用六个 full_ce/grade_ce 参考。预注册和代码 `fd16e2e`，配置 `053d4df`。首次一步 AMP 梯度检查在原初始缩放 65536 下溢出，正式训练尚未开始；改用稳定缩放 128 检查，正式训练仍保留原动态缩放。该修改与签名更新已完整记录 `1d4594b`。预检查 `fd259f6` 通过配对初始化、冻结参数无变化、四种 BN 状态、真实图像有限梯度和起点惩罚解析梯度。

2026-10-24 09:15:32 独立监督 PID 700300 启动，并发 2、CPU 线程 2。入口 `experiments/run_transfer_study.py`，策略 `experiments/transfer_policies.py`，状态和日志 `experiments/transfer_study/`。强信号需相对 grade_ce 三种子均正、均值至少 +1.0pt 且患者区间下界正，同时相对 full_ce 均值为正。固定食谱下的冻结结果不能代表充分调优的线性探测上限；成熟约束不能当作新模块发表。

第六轮结果 `5e25aa1`，审计与中文结论 `0be1115`。冻结 BN 均值 0.889178，对等级加权基线 +0.72pt，患者区间 [-0.86,+2.29]pt，三个种子变化 +1.57/+0.94/-0.34pt，未达到晋级标准。其患者等权准确率也没有提高。其余仅分类头 0.845186、仅 stage4 0.879003、起点约束 0.882276。最佳 checkpoint 冻结参数/BN 缓冲逐位核验通过，所有权和完整预测正常。完整交付见 round6。

## 第七轮已完成：复用原基线的患者 OOF 内部复核

不继续按固定验证集扩增归一化模块。沿用数据阶段每个种子的原五个外层患者折及各自内层患者验证划分；普通 BN 的原 15 个全量 CE OOF 模型签名、checkpoint SHA256、60 epoch 和外层预测已重新验证，直接复用。仅新增 15 个冻结 BN 模型，种子 42/43/44。原 train2386/185 组参与，原固定 val/test 不训练、不选择、不评分；不使用 A–E 等级或权重，避免外层信息通过全局等级泄漏。共比较 30 模型，但新增训练只有 15 项。

直接复用不可改写的 core.run_worker 训练代码，仅通过新模型实例的 train(mode) 行为冻结 36 个 BN 运行统计，所有参数含 affine 可训练，正常 eval，不增加推理模块。旧 256 缓存包含原全部数据，但加载器实际只访问原 train 索引；不可把它说成排除了 test 的缓存。

预注册/代码 `3047f83`，冻结配置 `516a78b`，预检查 `0002a37`。通过 15 基线哈希与原始签名、外层预测和指标、精确三集合患者隔离、缓存完整 SHA、每个种子每图恰一次外层覆盖、初始化 eval 等价、BN 缓冲不更新和 affine 非零梯度等检查。2026-10-24 09:45:59 监督 PID 704262 启动，并发 2、CPU 线程 2。

入口 `experiments/run_bn_oof_study.py`，协议 `experiments/ROUND7_PROTOCOL.md`，状态/日志 `experiments/bn_oof_study/`。主要指标为每种子合并五折后对完整 2386 图计算 macro-F1，再作种子配对差；以 185 患者组进行 20000 次共享自助抽样。强诊断信号需三种子均正、均值至少 +1pt 且区间下界正。属于开发后内部患者 OOF 复核，不是独立外部验证；重叠训练折和选择不确定性不能被该条件区间完整覆盖。若有信号，再单独验证数据×BN交互与合理的新结构，不能将成熟 BN 控制包装成创新。

第七轮结果 `3e1376c`，独立审计与中文解读 `8eb130d`。普通 BN OOF 均值 0.859045，冻结 BN 0.869416；种子增幅 +0.81/+1.83/+0.47pt，均值 +1.04pt，185 患者区间 [-0.53,+2.63]pt，仍未达强诊断标准。恶性召回种子平均均约 0.6950，不能说恶性召回改善；患者等权准确率均值约 0.8812→0.8908。30 模型来源、60 epoch、冻结缓冲、内层选择、外层指标、六份 2386 行合并 OOF 和患者重采样独立核验通过。完整交付 round7。

## 第八阶段已完成：固定一阶/二阶纹理统计诊断

不沿弱 BN 信号继续枚举归一化模块。复用第七轮 30 个已审计 CNN，分别在普通 BN 和冻结 BN 主干上，比较 GAP、GAP+通道标准差、GAP+固定投影协方差。每层级三个种子、五外层折，共六个方案、90 次解析岭分类头拟合，**零新增 CNN 训练**。用途是检验均值之外的统计信息，再决定是否值得训练新的统计聚合网络。

每个主干仅抽取各自 fit 和 outer 图像的 layer4 特征，全部索引属于原 train0..2385；原 val/test 不加载。外层单独批次抽取以复现原 CNN 概率。协方差为固定正交 512→32 投影的中心化协方差，上三角符号平方根描述子；不是矩阵平方根或 MPN-COV 复现。图像内标准差与协方差属于成熟统计，不能声称病灶定位、人工确认或新颖机制。

各描述子标准化参数只用 fit 患者，按块维数缩放，固定 Ridge alpha=1、无调参。三种描述子维数 512/1024/1040 不完全匹配，因此若有正信号仍须参数/计算对照，不能直接归因统计结构；岭 softmax 分数不作概率校准或 A–E 分级。

预注册与实现 `53b8900`，配置 `abd35e5`，预检查 `57c2a61`。已通过源 checkpoint 签名、原患者三集合、缓存 SHA、正交投影、NumPy 独立统计参考、位移/位置排列检验、fit-only 变换不受留出特征变化影响、岭解析解和两类主干原始 OOF 概率复现。前两份特征缓存可复用。

2026-10-24 10:19:27 独立监督 PID 708513 启动；并发 2、CPU 线程 2。入口 `experiments/run_statistics_study.py`，统计公式 `experiments/texture_statistics.py`，协议 `experiments/ROUND8_PROTOCOL.md`，状态/日志 `experiments/statistics_study/`。特征缓存 `.cache_statistics_study`、岭参数 `weights/statistics_study` 不入 Git，元数据锁定哈希。每单元一个旧 CNN、三个描述子，30 单元不是 30 次神经网络训练。

主要比较是各主干内 covariance / std 相对 GAP 岭对照，完整合并 OOF 后作患者配对重采样；四个比较同时报告普通 95% 区间和 Bonferroni 家族区间 [.625,99.375] 分位数。跟进信号需三种子均正、均值至少 +1pt、家族区间下界正，并在另一主干层级同向。属于内部诊断，不直接宣称架构贡献。所有前七轮冻结源和配置保持不变。

第八阶段结果 `a84d366`，独立审计与中文解读 `c2b671e`。普通 BN 的 GAP/加标准差/加投影协方差均值 .862553/.861927/.859587；冻结 BN .871950/.872489/.872603。四比较变化 -.30 至 +.07pt，普通与家族区间均跨零，未达继续训练统计结构标准。30 源 CNN 概率最大差为 0.0；90 个 fit-only 变换、岭正规方程最优性、保存头重算预测及18份 OOF/家族区间独立审计通过。完整交付 round8，包含小型岭参数；大特征缓存和 CNN 留服务器。

## 第九阶段已启动：独立预训练表征基线

不继续堆二阶统计模块。比较冻结的 ImageNet ResNet34 GAP 与公开自监督 DINOv2 ViT-S/14 的 CLS、patch均值、CLS+均值（主方案）。原15患者 fit/inner/outer 划分不变；4个新方案×3个患者划分seed×5折=60次解析岭拟合、零新增神经训练，另外复用第八阶段普通/冻结 BN 微调 CNN 的六份 GAP 岭 OOF 参考。三seed是患者划分重复，不是三个公开预训练模型初始化。

两新主干独立预训练、全部参数冻结，原 train2386 上仅逐图推理；原固定 val/test 不加载、不分级、不改标签。沿用 center224/resize256 与 ImageNet normalization，提取 FP32 并关闭 TF32。头仅 fit 患者拟合均值/标准差和固定 alpha1 Ridge。预训练数据、架构、精度、维数和计算存在差别，不能把收益归因新结构或单一预训练机制。

官方 DINOv2 小模型权重 88,283,115 字节，实验锁定 SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`，官方来源与 Apache2.0 已核实并附许可。通过已有 timm1.0.27 加载，无新依赖安装；所有推理 key 严格匹配，仅丢弃不用的 mask_token，并将位置编码从1370插值为257（224输入）。这是声明了位置插值方式的 timm port，不声称精确复现官方 offset 插值推理。普通 CNN 特征参数约21284672，ViT21628800；不宣称计算预算匹配或最新最强模型。

预注册/代码 `d927e36`，冻结配置 `037e8dd`，预检查 `f18a444`。原患者三集合和覆盖、原缓存 SHA、DINO strict加载/插值独立参考/CLS输出、两主干推理前后状态不变、冻结特征子集复现、fit-only变换和岭解析解均通过。2026-10-24 10:48:18 监督 PID713224 启动，并发2、CPU线程2。

入口 `experiments/run_representation_study.py`，模型加载 `experiments/pretrained_representations.py`，协议 `experiments/ROUND9_PROTOCOL.md`，状态/日志 `experiments/representation_study/`。独立预训练特征仅抽取一次，缓存 `.cache_representation_study/features.npz` 为2386×1280，岭参数在 `weights/representation_study`。主方案 dino_concat 预先固定，五预定义比较报告普通与5比较家族区间 [.5,99.5] 分位数。强跟进需其相对冻结预训练 ResNet 及两个既有微调 GAP 岭参考均三seed正、均值≥+1pt且家族下界正。次方案的正信号需另行预注册复核，不能事后换主终点。

## 后续判据

先完成本轮全部配对种子和消融，以完整开发集和患者配对区间分析。出现正信号后进行患者级独立确认、数据 × 结构交叉、原始模块与参数/计算匹配对照。严格外层 CV 中，等级必须在外层训练折内重算或由真正独立的分级器提供；不得直接复用会接触外层留出标签的等级。

若本轮仍无可靠收益，先做错误分布与消融分析，再预注册不同机制的下一轮，保留阴性结果。候选最终冻结后才运行一次完整 test 审计，并明确已有 test 被历史实验使用的限制。最终可交付可复现网络与诚实实验结论，不预先承诺论文创新性或录用。


## Round9 completed and audited
Frozen pretrained ResNet .840943; DINO CLS .866158, mean .871247, concat primary .873057. Prior finetuned GAP ridge .862553/.871950. Concat exceeds frozen ResNet by 3.21pt (family CI [1.23,5.49]pt) but not reliably prior finetuned references; full primary gate false. No neural training; 60 fit-only ridge fits, 18 full OOF sets. Independent audit passed original patient splits, all source/cache/head hashes, ridge optima, predictions and family intervals. Results fe007f2. Next exploratory step is controlled CLE domain adaptation, not an assertion R9 passed.


## Round10 started: controlled CLE token adaptation
Exploratory domain-adaptation study after R9 full primary gate failed. Six variants x seeds42/43/44 =18 new neural trainings: frozen head, full fine-tune, Q/V LoRA rank24, generic local rank48, CLS-context local rank24 primary and exact-budget mean-context control. Full original train2386/val538, no grades/weights/test/relabeling; same original224 transforms. PEFT trainable149763/153411/152547, whole counted FLOPs ratio <1.01. Matched initial source/head and AMP outputs, independent algebra and 2-step gradients, frozen tensors and actual batch32 memory checks passed. Code/protocol a328665; initial config ddb89b0; preflight-only partial qkv audit-key mapping fix 4c62b0d before any training, no forward or recipe change; revised signatures recorded. Preflight a6f3871. Source/config/protocol now frozen. Supervisor PID 717520 at 2026-10-24T11:22:21.839150+08:00. Main candidate requires all five preregistered family comparisons to pass. No new contribution or label-noise claim yet.


## Round10 completed and independently audited
18 models/60ep/full val538. Frozen head .865834, full DINO adaptation .892762, LoRA .872516, local .875194, primary CLS context .875779, mean context .872422. All five primary family intervals cross zero; complete gate false. CLS candidate vs local only+.06pt and vs full -1.70pt. Descriptor effect75-96%, adapter-off disagreement22-56%, so not dead mechanism. Full adaptation final train99.99%, final val .865606 versus best .892762, checkpoint selection/generalization need patient OOF. Results47816ef. Independent audit reproduced all18 model on/off predictions, source tensor freezing/full drift, bounded per-image contexts, signatures/60ep/splits/full coverage and patient family intervals. Next is 15 full-adaptation original-patient OOF checks, no further similar adapter enumeration.
