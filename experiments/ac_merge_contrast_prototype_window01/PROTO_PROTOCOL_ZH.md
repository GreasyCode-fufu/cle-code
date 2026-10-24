# Swin首层PatchMerging空间对比：公开权重CPU数值原型

当前POWER十五模型控制继续原注册，不提前改动或停止；本原型无CLE图/新神经推理/训练/test，正式后续结构和保留组件须等POWER完整对照。原M方向矩是当前工作信号，未确认q或患者aux增量，不将它们当已确认组件。

只读主干复用审计已核原Swin A/B六个完成权重与预测，不重训原参考。Swin B是原图核SHA后448缓存/crop384，物理2504仅train/val，不是从256缓存上采样。工作Swin B的选择依赖历史test，必须披露；固定ConvNeXt-Tiny_A224 B0不替换/削弱。这个原型测试的是向已有竞争力Transformer迁移的兼容性，不能据公开初始化检查宣称CLE效果。

机制假设：原方向矩补充stem对角度响应的幅度信息，而PatchMerging将2×2邻域拼接、LN后线性压缩，可能压缩每通道局部空间异质性。提出在首merge增加零初始化C→2C线性残差，保留原merge、LayerNorm和reduction参数对象与原输出。先将4C拼接向量做不带学习仿射的LN，得到四个u_q，h=sqrt(mean_q((u_q−mean_q u_q)²)+1e−6)−1e−3。平坦邻域h严格0，eps下缩放只近似不变。不带仿射避免原LN各象限gamma/beta破坏统计的象限置换性质；原学习LN仍原样服务主分支，整网不保证旋转等变/不变。

只作用Swin首merge(dim96)，新增18432权重+192偏置=18624参数，原ImageNet起点logit/原参数ID/RNG应严格保留，无已训练权重热启动。假设角度离散与空间局部异质性可互补，必须后续累计/去一验证；不能把标准差、LN、残差或其组合自动称创新。

候选算子对照先数值明确、尚非正式CLE训练：local_std=局部每2×2对比；mean=相同新分支参数的四邻均值（平均信息/优化控制）；global_std=同局部h图先每图空间均值再广播，移除空间定位（全局信息控制）。拟议完整模型包括纯Swin、仅原Mom、Mom+local、无Mom+local、Mom+mean、Mom+global；未来正式任务/筛选冻结另行进行，不从尚未完成POWER结果挑配方。

独立数值：NumPy直接pad/拆邻域/不带仿射LN及均值/方差，正交Haar高通能量/4的独立方差oracle（只是成熟算子等式，不是HWD完整网络），平坦/单点/低能量有限、同均值不同对比反例、偶数空间90°描述协变、奇数完全保留Torchvision右/下padding但不宣称旋转性质、doublegradcheck、两合成更新。公开Swin384各变体起点logit/RNG/原参数、有限反向和新分支非零梯度/预算。原Swin384 Mom已有数值证明不是正式CLE训练，不重复任何有效CLE模型。

成熟参照：[Swin论文](https://arxiv.org/abs/2103.14030)与[Microsoft原始代码](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py)。官方原码以B×HW×C和固定偶数H/W工作；实际Torchvision父基线为NHWC并允许右/下pad奇数。保留两份源及许可证，纯PatchMerging类受限AST的输出/输入和参数梯度在相同权重偶数输入上对齐；奇数原码断言拒绝、runtime支持。保持concat→LN→Linear的实际顺序，不把它错当SwinV2的reduction→LN。

[Haar Wavelet Downsampling论文](https://www.sciencedirect.com/science/article/pii/S0031320323005174)已存在wavelet下采样先验；其摘要可核，但全文当前403，不能虚称全文逐式或完整模块复现。本原型不是用全部LL/LH/HL/HH拼接学习的HWD，Haar只作精确数学oracle。共享库仍只读，未安装旧软件或新包。零残差适配本身也为成熟思想。

本原型仅合成张量和公开ImageNet权重，所有特征eps/统计为内部运算，不改变标签/等级阈值/清单。A–C历史test查看/构建标签依赖及自适应内部开发边界不变，未来患者bootstrap仍仅条件区间。不保证提点、SCI新颖或录用，整体目标active。
