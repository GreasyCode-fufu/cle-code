# CLE中间层聚焦对比：完整十五模型轮

15×60epoch全完成；427val/576test四角度两独立路线概率、实际权重/source/SHA、每epoch2077唯一索引及患者互斥审计通过。所有seed与阴性保留，本轮结束不等于整体完成。
历史条件数值字段（不是单点准入门槛）方案：无。仅为自适应内部条件数值结果，不代表新颖、独立泛化或外部验证。

|方法|s45 F1|s46 F1|s47 F1|均值|对B0差pp|条件CI pp|
|---|---:|---:|---:|---:|---:|---|
|mom_focalcontrast_single|94.66776%|95.62079%|94.18210%|94.82355%|-0.15789|[-1.36520,1.04956]|
|mom_focalcontrast_c4|96.11335%|96.52406%|94.89694%|95.84479%|0.86335|[-0.37564,2.14868]|
|focalcontrast_only_single|96.15317%|95.23493%|94.35786%|95.24866%|0.26722|[-0.80348,1.52123]|
|focalcontrast_only_c4|96.00288%|95.43774%|94.73352%|95.39138%|0.40994|[-0.67333,1.50887]|
|mom_focalraw_single|94.88045%|95.47388%|95.80491%|95.38641%|0.40497|[-0.39258,1.36904]|
|mom_focalraw_c4|95.60572%|95.64533%|96.32947%|95.86017%|0.87873|[0.08265,2.03070]|
|mom_misaligned_single|95.46720%|95.61971%|96.33579%|95.80757%|0.82613|[-0.18139,2.82242]|
|mom_misaligned_c4|96.67373%|96.17252%|96.31383%|96.38670%|1.40526|[0.28606,3.43214]|
|mom_additive_single|96.33720%|95.10442%|95.47265%|95.63809%|0.65665|[-0.10066,1.86499]|
|mom_additive_c4|96.50315%|95.98949%|95.98368%|96.15877%|1.17733|[0.30848,2.58611]|

## 累计/去一/成熟控制

mom_focalcontrast在第二stage192×28×28加中心化Focal特征残差；focalcontrast_only真去18528Mom；focalraw去调制器中心化；misaligned改变q与m位置配对；additive用加法代乘法。均新增11832参数。原Mom/后续stage/GAP/learnednorm/head保留。up首步正、down/core首步0第二步正，正式CPU/CUDA已核验。最终2–3网络点组合累计+1.5，单点按父模型贡献、seed、条件不确定性和成本评估。

|mom_focalcontrast比较|配对seed差pp|平均差pp|条件CI pp|
|---|---|---:|---|
|single_vs_M_moments|-0.97605,-0.36933,-1.80490|-1.05010|[-2.90675,-0.03431]|
|single_vs_focalcontrast_only|-1.48541,0.38586,-0.17576|-0.42510|[-0.99750,-0.02058]|
|single_vs_mom_misaligned|-0.79944,0.00107,-2.15368|-0.98402|[-3.16063,-0.00757]|
|single_vs_mom_additive|-1.66943,0.51637,-1.29055|-0.81454|[-2.42734,0.33472]|
|single_vs_mom_focalraw|-0.21269,0.14691,-1.62280|-0.56286|[-1.65557,0.42859]|
|single_vs_B0|-0.58371,1.39496,-1.28491|-0.15789|[-1.36520,1.04956]|
|c4_vs_M_moments|-0.57525,0.17791,-1.44207|-0.61314|[-2.28293,0.32393]|
|c4_vs_focalcontrast_only|0.11048,1.08632,0.16343|0.45341|[-0.21683,1.12945]|
|c4_vs_mom_misaligned|-0.56038,0.35154,-1.41689|-0.54191|[-2.30195,0.50143]|
|c4_vs_mom_additive|-0.38979,0.53457,-1.08674|-0.31399|[-1.69756,0.72198]|
|c4_vs_mom_focalraw|0.50764,0.87873,-1.43253|-0.01539|[-1.06247,0.82227]|
|c4_vs_B0|0.86188,2.29823,-0.57007|0.86335|[-0.37564,2.14868]|

所有新arm对原Mom/成熟ECA/aug同视图比较见mechanism_comparisons.json，B0_C4/DINO_C4/Swin_C4等强参照完整保留于round_results。原Mom及固定B0有效三seed复用，不重训或削弱调参预算；成熟C4收益不能自动归因新结构。

|方案|accuracy|恶性召回|患者等权accuracy|参数|逻辑GFLOPs|
|---|---:|---:|---:|---:|---:|
|mom_focalcontrast_single|94.90741%|89.77273%|92.32258%|27852795|9.130393344|
|mom_focalcontrast_c4|95.89120%|92.23485%|92.49021%|27852795|36.521573376|
|focalcontrast_only_single|95.31250%|90.53030%|93.14960%|27834267|8.928083712|
|focalcontrast_only_c4|95.42824%|91.47727%|92.51068%|27834267|35.712334848|
|mom_focalraw_single|95.42824%|93.37121%|91.84903%|27852795|9.130393344|
|mom_focalraw_c4|95.89120%|93.56061%|93.23926%|27852795|36.521573376|
|mom_misaligned_single|95.83333%|92.61364%|95.12663%|27852795|9.130393344|
|mom_misaligned_c4|96.41204%|93.56061%|95.25718%|27852795|36.521573376|
|mom_additive_single|95.65972%|94.50758%|92.48902%|27852795|9.130393344|
|mom_additive_c4|96.18056%|94.31818%|93.77057%|27852795|36.521573376|

原Mom27840963/9.111840768G；本轮累计27852795/9.130393344G，去Mom27834267/8.928083712G；B0为27822435/8.909531136G。C4四倍前向，不作为第二/第三网络设计；逻辑2FLOP/MAC含新全部1×1/深度卷积及原三额外旋转卷积，排除LN/GELU/中心化/逐元素门控/replicate padding/访存。成熟Focal适配与Mom组合不自动新颖，无临床定位或解释保证。实际吞吐另测。

## 实际资源

|真实父训练数|覆盖秒|时间加权GPU|GPU至少80%时长|CPU|峰显存MiB|
|---|---:|---:|---:|---:|---:|
|0|78.94|26.444%|12.821%|6.436%|1309|
|1|2220.73|87.739%|86.496%|8.470%|4295|

连续1136行覆盖2299.668秒，全部15次launch/PID/PPID采样核对；39行status滞后按真实父子进程恢复。低GPU区间保留，0父进程可能为管理器推理/CPU分析，不都为空闲。无记录不补造，不外推设计/CPU阶段满载，不虚称Agent连续在线。profile cap不是正式并发。

|arm|epoch图秒|加载等待比例|元数据队列图秒|
|---|---:|---:|---:|
|mom_focalcontrast|897.20|1.595%|818.31|
|focalcontrast_only|1083.15|2.019%|972.93|
|mom_focalraw|898.23|1.611%|817.71|
|mom_misaligned|893.84|1.593%|814.54|
|mom_additive|897.56|1.598%|817.66|

## 复现、使用与科学边界

固定首val2386实际checkpoint单图CLI对原batch32最大概率差6.377696990966797e-05<预设1e-4，类别一致；非三seed效果或临床验证。原Q/M A224 recipe、结构、预测命令与实际吞吐见STRUCTURE_AND_USE_ZH.md；实际权重SHA/最早valbest见weights_and_tasks.json。


首版默认cuDNN B1误差0.00012111663818359375超过固定1e-4，原失败保留。结果后仅增加native CUDA卷积预测包装器（显式关闭cuDNN、AMP FP16及真实B1保留），同图/同checkpoint/同阈值复核通过。该后端修复不是事前注册，默认旧CLI未被宣称通过；见usage_smoke/cli_resolution_v2.json及NUMERICAL_USAGE_CORRECTION_ZH.md。单图native路径未沿用训练cuDNN吞吐实测结论。

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



## 正式15模型事前注册

5arm×45/46/47全部60epoch/batch32，原Q/M A224同公开ImageNet/seed head、CE，encoder1e-5、新模块/head1e-4、AdamW.05/cosine0/clip1/AMP/stochasticdepth.1，HV翻转/15度旋转/随机224、workers2/thread1。原Mom/B0有效3seed复用不重训，固定B0/标签/等级/清单/患者split/采样不改。

预注册mom_focalcontrast对原Mom/focalcontrast_only/mom_focalraw/mom_misaligned/mom_additive/B0两mode12整数Fraction比较，41患者20k条件CI，另保留原Mom/ECA/aug和同C4 B0/DINO/Swin强参照。3/200只是保留的历史数值审计，最终2–3网络设计组合累计目标；单点保留看相对父模型新增贡献、seed稳定性、条件不确定性和成本，可信小增量可保留。C4不作为第二/第三网络设计。不给成熟Focal模块或组合自动贴创新结论。

先正式CPU及CUDA32各arm两有限更新：up首步正、down/core首步0第二步正，Mom首步正/drop真去18528；中间模块全部headlr组。注意AdamW权重衰减可能在零梯度首步改变上游权重，不把首步无学习梯度说成绝对权重不变。原M3实际权重mom_only严格加载+B0三共六完整427val概率误差0，缓存SHA/test训练拒绝/2077索引/每完整目标实测吞吐/6GiB余量/97%最小有效并发全过后才开训。三seed全部60epoch/val最早最佳登记实际weight后test，独立427val576test×4两路线审计。所有阴性保留，原Mom保留，普通阴性不自停整体。
