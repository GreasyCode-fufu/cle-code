# 中间层聚焦对比调制：无CLE图像CPU原型

原CNN方向矩M继续保留。SAL完整15模型与12直接控制已完成；内容注意力候选相对M的single/C4三seed平均差均负、条件区间均负，其余控制没有构成该候选新增贡献。结果与全部阴性保留，不泛化否定全部内容建模。最终+1.5个百分点是2–3个互补网络设计组合相对固定B0的累计目标，不要求单点各自达到1.5。

本次假设是让空间上下文直接改变中间特征，再由原后续网络处理这些特征。插入ConvNeXt第二stage末尾（encoder[0][3]后，下一下采样之前），224输入对应192×28×28。保持原stem Mom、原全部特征层对象和参数、GAP/learned norm/head；新的特征残差不增加末层logit读出。

公式：逐位置无仿射channel LN1e-5→无bias192→24投影。投影经1×1分出query q、context z及三个无softmax门控g（f和h均无bias，以去掉中心化取消的常数调制及全局门控bias）。z依次经3×3与5×5深度卷积和GELU，形成短/长上下文；最后一级空间均值再GELU形成全局上下文。按三个位置门控聚合，1×1通道映射得m。候选使用q×(m−空间均值m)，逐位置无仿射24通道LN1e-5→零且无bias的24→192投影，再加回原特征。

五arm：mom_focalcontrast累计；focalcontrast_only真去18528Mom；mom_focalraw不中心化m、保留成熟Focal乘法核；mom_misaligned对中心化m的784位置作固定default_rng(20261025)置换后与未动q相乘，保持各自边缘分布但改变配对；mom_additive改乘法为q+(m−mean(m))，同参数但交互函数不同。五arm均新增11832参数；每图逻辑新增18552576 FLOPs（原224空间28×28），不将参数相同视为完全相同有效表达容量。nominal累计27852795参数/9.130393344G，去M27834267/8.928083712G；须CPU完整真实模型计数验证。mom_only/plain仅初始化参照，原Mom/B0有效三seed不重训。

成熟来源：NeurIPS2022 Focal Modulation Networks，https://papers.neurips.cc/paper_files/paper/2022/file/1b08f585b0171b74d1401a5195e986f1-Paper-Conference.pdf ；作者官方https://github.com/microsoft/FocalNet/blob/main/classification/focalnet.py 。官方代码与Microsoft MIT LICENSE字节保留；仅受限AST调用FocalModulation.forward、最后proj设Identity以比较其投影前调制核，未执行整网、训练入口或第三方导入。官方零padding核心与本实现raw核心的零padding模式做完整输出/输入及参数梯度oracle。正式候选深度卷积采用replicate padding，属于明确适配，不把与官方零padding不同的边界行为说成完全复现。共享模块库只读参考，未修改其文件。

此处的rank压缩、replicate边界、中间特征零残差、调制器中心化和无仿射LN均是适配。成熟Focal调制与M组合不自动是SCI创新。中心化m的空间均值为0，不代表q×m或最终残差空间均值为0。常数空间输入下contrast/misaligned的归一化前描述应为0（浮点容差）；raw/additive不具有该约束。replicate边界与任意已学习卷积核不保证旋转或平移等变，更不证明病灶定位、医学结构或稳健性。全局均值分支仍存在；没有声称仅局部。固定错配置换破坏规则邻接。

CPU只用随机合成及公开ImageNet起点，不读取CLE train/val/test图。先冻结源/config，再进行独立NumPy线性/显式replicate卷积/GELU/门控/中心化/LN oracle、doublegradcheck、退化矩形与常数有限、批独立、同完整通道边缘分布异空间布局反例、两合成更新；三seed×七完整公开224起点的logit/RNG/原参数ID应严格一致。零up使up首步梯度正，down/f/h/两DW首步0、第二步正；M首步梯度另检。初始零输出不代表上游已首步训练。

逻辑2FLOP/MAC包含全部新1×1/深度卷积和原三额外方向卷积；不计LN/GELU/中心化/逐元素门控/replicate padding/访存。CPU通过后仍须新目录正式冻结累计/去一/成熟控制，原Q/M A224 CE/60epoch同recipe、实际Mom/B0六完整427val概率严格重建、CUDA32两有限更新和分层梯度、白名单/缓存/实际吞吐/显存余量预检通过后才能正式开训。

train训练→val最早最佳选权重→test→下一设计；test均为自适应内部开发，历史查看且AC构建用其标签，患者bootstrap只为条件区间，不校正选择偏差，不称独立泛化或外部验证。整体active，不保证提点、新颖或录用。

