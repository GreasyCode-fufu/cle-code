# CLE A–C 新窗口结构研究，第一轮预注册

研究身份：ac_structure_20261024_window01；聊天 01a16d4e-48fe-7e22-8c4a-c6de0f4d6f76。此任务重新授权新的结构研究，不恢复旧完整数据队列。共享库严格只读；新增源码、配置、结果、权重均在专用目录。没有临床部署或论文创新、录用承诺。

## 数据和历史限制

唯一数据为 ac_dataset_v1/dataset_config.json 和 dataset_manifest.json 冻结的 train2077/182患者、val427/39患者、test576/41患者。原标签、等级阈值、患者split和索引全部不变。历史test被看过，AC构建用过其标签；本任务只可称固定选择后分布的内部复核，不能称未接触外部验证。等级是模型一致性代理，不证明医生标签正确。

旧.cache_data_study含3595图。本任务加载器把所有getitem限制为冻结train或val索引；训练只能train，checkpoint选择只能val。逐epoch记录实际访问索引集合及训练多重计数。test仅在最终候选和统一比较方案的哈希冻结后一次核验。模型/超参选择不能读取新test结果或其错误。条件于AC选择的患者OOF若后续使用，也不称完全独立验证。

## baseline冻结

本轮baseline固定为DINOv2 ViT-S/14官方无register预训练，timm1.0.27移植，CLS+末层patch均值768线性头，全参数微调。选择依据仅既有val三seed均值：DINO full .982784 > 原CNN .973445 > lowLR CNN .968198、frozen DINO .968072。并未依据历史test排序。历史test不能证明DINO优于CNN。本轮比较期间不更换。

用户在正式训练前追加授权：可先探索DINO，若优化难提点，可换其它基础baseline展开设计。本次把DINO作为第一阶段强参照；若本轮没有稳定开发增益，下一阶段可以考察其它主干并依据开发数据、公平训练/调优选出有竞争力的基础baseline。必须在该阶段结构比较前再次预注册冻结，保留DINO和此前全部结果，并披露任务中的baseline阶段转换。不能事后把当前阴性结果改成新baseline上的成功，也不能降低baseline预算或按test换baseline。最终一次统一test前须固定最终主要baseline和候选；此后不再按test切换。

严格复用ac_backbone_recipe中seed45/46/47的三项有效DINO full，按原任务身份/源码签名/config SHA/checkpoint SHA/60日志/最早最高val checkpoint/427逐图概率验证。原CNN作为历史成熟参照保留。新网络与baseline采用相同数据、增强、epoch、优化器和seed；baseline没有被减少调参或训练预算。此食谱不是各架构穷尽调优的全局最优；不得将本轮训练预算对比说成每个架构充分优化。额外baseline食谱调整必须同等给主baseline和候选机会，不能偷偷替换主要baseline。

食谱：60epoch，batch32，AdamW wd .05，encoder LR1e-5，head/新增模块LR1e-4，cosine降到0，CE无平滑/样本权重，clip1，动态FP16 AMP。原resize256缓存，train H/V flip、rotate15、crop224；eval center224和ImageNet normalize。记录AMP跳步。checkpoint取最早的最大val macroF1，不能挑末轮或seed。种子45/46/47全报告，baseline结果复用而非重训。

## 首轮假设与重训对照

CLE含细胞/腺体纹理及组织排列，末层均值可能丢失中层局部关系。仅据train/val曲线：DINO训练末期100%，最佳val轮次0/43/35且末轮较低，说明最终表示及读出稳定性值得研究，不证明机制或标签噪声。

自主实验设计relational：在相同完整微调DINO第4/8/12层取归一化patch，共享384→96投影和LN。每深度拼接patch均值和patch与3×3邻域均值的逐通道乘积均值；以可学softmax深度权重合并，192→768零初始化残差加到原768描述子。初始前向与baseline完全一致，所有主干参数可训练。这里的邻接语义相关不是原像素亮度对比，也没有病灶区域标注。与旧冻结主干晚层局部adapter、亮度路由和LS核改造不同；成熟的多层特征、二阶乘积、残差融合本身不构成已证明新颖性。

重训no_neighbor：邻域改为token自身，保留逐token二阶能量；检验空间邻接贡献。no_depth：三次读末层，参数、下游次数一致；检验跨深度贡献。final_only：同时只末层和自身乘积，作为同容量同计算读出参照。四者所有参数相同，paired初始化共享部分完全相同。保留off诊断，但不能用off替代重训消融。邻域操作是含中心3×3有效像素平均、边缘不补零计数；没有手工图像质量阈值或新筛选。

official4：忠实实现DINO官方线性评估读出公式（最后四个block CLS拼接，加末层patch均值，1920线性头），本轮统一采用全微调食谱，所以不是官方线性probe完整复现。新分类头统一用PyTorch默认初始化，而官方线性probe为normal(.01)/零bias；这一差别公开记录。官方归一化/选择层定义需核对，timm位置插值offset差异沿用披露。ConvNeXt-Base torchvision已缓存ImageNet1K权重作为现代层级CNN成熟参照，严格加载官方状态；同60epoch食谱，stochastic depth=.1，非等参数/等计算。其预训练和计算差异不能归因CLE模块，也不能凭这一固定食谱单次表现声称已充分调优的其它基础baseline。

第一轮6项×3seed=18新训练，3个主要baseline复用。不把所有成熟模块盲目拼接或把文件名会议年份当真实论文。全库静态AST/依赖/源码hash梳理，优先审核实际选用的实现；未与原文核实的模块仅列库存，不称忠实复现。第一轮后只按开发误差和统计证据决定新机制，不能把18项完成当整个目标完成，也不无限枚举近似readout。

## 指标、开发晋级和最终成功

主指标macroF1（labels固定0/1/2，zero_division0），同时accuracy、label1恶性召回、患者等权accuracy、3×3混淆矩阵、参数、计数FLOPs（两FLOP/MAC；忽略项明示）、吞吐/时间/显存和资源。患者等权accuracy为先每患者图像accuracy再平均。

首轮主要问题relational vs frozen primary baseline；机制对照为no_neighbor/no_depth/final_only/official4。共享39患者20000次有放回配对抽样，每次所有seed共用同一患者重数，先逐seed算macroF1再均值配对差。报告普通95%与5比较Bonferroni家族[.5,99.5]分位区间；不是将seed平均概率当单模型。开发强信号：三seed差均>0、平均提升>=.005、五比较家族下界>0（对baseline）；声称邻接和跨深度机制还须对相应两消融同样同向且家族下界>0。未通过保留阴性，不必停整个研究。baseline在val已98.28%，val剩余空间很小；不能以val+1.5作为最终成功的替代。

最终统一test前，冻结一个主要候选及必要对照/全部seed/checkpointSHA/源码SHA/推理配置/比较规则。主要成功必须三配对seed的test平均macroF1提升严格>.015，三个seed均正，41患者配对bootstrap普通95%下界>0；若冻结多主要比较则先规定家族校正并报告。点估计未过或跨seed不稳或下界<=0均不能报告可靠达标。不能依据新test错误改结构、超参、清单再试test。若核验失败诚实保留；后续研究只能继续开发证据，其同test验证已消耗，需明确新的确认方案或外部数据条件，不能循环测试。

## 资源、审计、持久工作

先真实数值检查、单任务/2/4/更多并发小段吞吐测量，根据总样本/s、CPU单核、GPU利用/显存和内存余量选择，动态增加并发不设旧6/8上限，受真实显存/CPU/队列约束；每2秒资源记录。微基准不作有效训练结果、随机状态不复用。正式每epoch记录loader wait、compute、samples/s和总耗时。GPU已满不堆无用CPU任务，不干预其他Agent进程。故障查日志和同流水线存活进程，训练complete且推理失败只补推理。新脚本checkpoint严格哈希签名，完成结果不可覆盖重训。

训练前提交本协议、配置、任务和通过的数值预检。独立审计脚本重算全部指标/最早最佳轮/预测/有限权重/严格重加载/源与checkpointhash/患者互斥/预算，不依赖训练摘要断言成功。Git只精确暂存本窗口新目录/源码及本窗口账本，保留旧冻结文件，检查login所有权。本窗口heartbeat自动衔接分析、审计和下一机制，正常训练和一般阴性保持安静，可靠达标/真实故障/必要信息才通知。中文报告、结构说明、复现与使用文档同步本窗口outputs；大型权重留服务器。

参考源：
- https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py
- https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
- https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
- https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
