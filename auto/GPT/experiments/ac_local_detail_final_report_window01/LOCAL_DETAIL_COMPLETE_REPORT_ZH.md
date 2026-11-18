# CLE局部细节强度对比：完整十五模型轮

15×60epoch全完成；427val/576test四角度两独立路线概率、实际权重/source/SHA、每epoch2077唯一索引及患者互斥审计通过。所有seed与阴性保留，本轮结束不等于整体完成。
历史条件数值字段（不是单点准入门槛）方案：无。仅为自适应内部条件数值结果，不代表新颖、独立泛化或外部验证。

|方法|s45 F1|s46 F1|s47 F1|均值|对B0差pp|条件CI pp|
|---|---:|---:|---:|---:|---:|---|
|mom_detailcontrast_single|95.43147%|94.35776%|95.30176%|95.03033%|0.04889|[-0.92414,0.86986]|
|mom_detailcontrast_c4|96.66167%|95.27433%|95.99827%|95.97809%|0.99665|[0.05312,2.15175]|
|detailcontrast_only_single|96.00405%|94.18455%|95.65436%|95.28099%|0.29955|[-0.42872,1.68314]|
|detailcontrast_only_c4|96.35595%|94.95491%|95.83381%|95.71489%|0.73345|[-0.07087,2.07092]|
|mom_detailrawgate_single|95.59315%|94.70451%|95.65591%|95.31786%|0.33642|[-0.48545,1.54131]|
|mom_detailrawgate_c4|96.09211%|95.45228%|95.99224%|95.84554%|0.86410|[-0.12326,2.35231]|
|mom_misaligned_single|95.06100%|96.32232%|95.12583%|95.50305%|0.52161|[-0.39885,1.97391]|
|mom_misaligned_c4|96.14793%|95.79855%|96.00097%|95.98249%|1.00105|[-0.16801,2.59127]|
|mom_highpass_single|95.57695%|94.53067%|95.99726%|95.36830%|0.38686|[-0.41865,1.55624]|
|mom_highpass_c4|96.61399%|95.45228%|96.34382%|96.13669%|1.15526|[0.32050,2.62016]|

## 累计/去一/成熟控制

mom_detailcontrast在第一stage96×56×56加每通道局部细节强度对比残差；detailcontrast_only真去18528Mom；detailrawgate去q中心化；misaligned错配q与D；highpass仅原始D。均新增9216可学习参数，仅up可学习、两步均正；固定滤波核无可学习上游。原Mom/后续stage/GAP/learnednorm/head保留，正式CPU/CUDA已核验。最终2–3网络点组合累计+1.5，单点按父模型贡献、seed、条件不确定性和成本评估。

|mom_detailcontrast比较|配对seed差pp|平均差pp|条件CI pp|
|---|---|---:|---|
|single_vs_M_moments|-0.21234,-1.63236,-0.68525|-0.84332|[-2.39558,-0.14239]|
|single_vs_detailcontrast_only|-0.57257,0.17321,-0.35261|-0.25066|[-1.68513,0.49237]|
|single_vs_mom_misaligned|0.37048,-1.96456,0.17593|-0.47272|[-1.59366,0.16092]|
|single_vs_mom_highpass|-0.14548,-0.17292,-0.69551|-0.33797|[-1.37218,0.35857]|
|single_vs_mom_detailrawgate|-0.16168,-0.34676,-0.35416|-0.28753|[-1.36055,0.32361]|
|single_vs_B0|0.18000,0.13193,-0.16526|0.04889|[-0.92414,0.86986]|
|c4_vs_M_moments|-0.02694,-1.07182,-0.34074|-0.47983|[-1.82209,0.24152]|
|c4_vs_detailcontrast_only|0.30571,0.31942,0.16447|0.26320|[-0.58508,0.83308]|
|c4_vs_mom_misaligned|0.51373,-0.52422,-0.00270|-0.00440|[-0.88054,0.64986]|
|c4_vs_mom_highpass|0.04768,-0.17795,-0.34555|-0.15860|[-1.10260,0.30195]|
|c4_vs_mom_detailrawgate|0.56955,-0.17795,0.00603|0.13255|[-0.77835,0.69387]|
|c4_vs_B0|-0.01431,0.16742,0.00190|0.05167|[-0.52794,0.68264]|

所有新arm对原Mom/成熟ECA/aug同视图比较见mechanism_comparisons.json，B0_C4/DINO_C4/Swin_C4等强参照完整保留于round_results。原Mom及固定B0有效三seed复用，不重训或削弱调参预算；成熟C4收益不能自动归因新结构。

|方案|accuracy|恶性召回|患者等权accuracy|参数|逻辑GFLOPs|
|---|---:|---:|---:|---:|---:|
|mom_detailcontrast_single|95.08102%|91.85606%|91.49964%|27850179|9.180481536|
|mom_detailcontrast_c4|96.00694%|93.18182%|93.25147%|27850179|36.721926144|
|detailcontrast_only_single|95.31250%|92.04545%|93.52111%|27831651|8.978171904|
|detailcontrast_only_c4|95.71759%|93.37121%|93.62607%|27831651|35.912687616|
|mom_detailrawgate_single|95.37037%|91.28788%|93.07260%|27850179|9.180481536|
|mom_detailrawgate_c4|95.89120%|92.42424%|94.17485%|27850179|36.721926144|
|mom_misaligned_single|95.54398%|92.42424%|93.51124%|27850179|9.180481536|
|mom_misaligned_c4|96.00694%|93.56061%|93.50234%|27850179|36.721926144|
|mom_highpass_single|95.42824%|91.47727%|93.01927%|27850179|9.175062528|
|mom_highpass_c4|96.18056%|93.18182%|94.38980%|27850179|36.700250112|

原Mom27840963/9.111840768G；本轮累计27850179/9.180481536G，去Mom27831651/8.978171904G；Mom+highpass9.175062528G；B0为27822435/8.909531136G。C4四倍前向，不作为第二/第三网络设计；逻辑2FLOP/MAC含新1×1与两次固定功能性F.conv2d（highpass一次）及原三额外方向卷积，排除LN/sqrt/中心化/逐元素/padding/访存。成熟二项滤波适配与Mom组合不自动新颖，无临床定位或解释保证。实际吞吐另测。

## 实际资源

|真实父训练数|覆盖秒|时间加权GPU|GPU至少80%时长|CPU|峰显存MiB|
|---|---:|---:|---:|---:|---:|
|0|91.12|31.797%|22.223%|5.264%|1307|
|1|2427.39|89.578%|88.899%|8.056%|4514|

连续1244行覆盖2518.501秒，全部15次launch/PID/PPID采样核对；44行status滞后按真实父子进程恢复。低GPU区间保留，0父进程可能为管理器推理/CPU分析，不都为空闲。无记录不补造，不外推设计/CPU阶段满载，不虚称Agent连续在线。profile cap不是正式并发。

|arm|epoch图秒|加载等待比例|元数据队列图秒|
|---|---:|---:|---:|
|mom_detailcontrast|812.94|1.450%|742.79|
|detailcontrast_only|962.34|1.742%|873.54|
|mom_detailrawgate|817.18|1.459%|747.51|
|mom_misaligned|771.21|1.370%|708.08|
|mom_highpass|873.07|1.549%|795.27|

## 复现、使用与科学边界

固定首val2386实际checkpoint单图CLI对原batch32最大概率差7.845461368560791e-05<预设1e-4，类别一致；非三seed效果或临床验证。原Q/M A224 recipe、结构、预测命令与实际吞吐见STRUCTURE_AND_USE_ZH.md；实际权重SHA/最早valbest见weights_and_tasks.json。

# 第一stage局部细节强度对比：CPU原型事前协议

本轮仍为未确认设计，保留原CNN方向矩M。原MFC完整15×60正常终态，不恢复、不重训。本轮不是最终+1.5个百分点的单点闸门，最终目标是2–3个互补设计相对固定B0的累计贡献。原Mom有效权重只作固定参照，正式新模型从相同公开ImageNet起点训练。

位置在ConvNeXt第一stage末encoder[0][1]之后、首次stage间下采样前，A224对应96×56×56。逐位置无仿射channelLN(1e-5)得到z，固定replicate边界二项3×3低通B；D=z−B(z)，R=sqrt(B(D²)+1e-6)，q=R/(R+spatialmean(R)+1e-6)。每通道q独立，不是会被channelLN消去的单一空间正标量。候选descriptor为channelLN((q−spatialmean(q))*D)，零无bias96→96的1×1投影后加回原特征，继续原下采样/全部后续层/GAP/norm/head。

五arm：mom_detailcontrast累计；detailcontrast_only真去18528 Mom参数；mom_detailrawgate不对q中心化；mom_misaligned固定default_rng20261025打乱3136位置的中心化q，D不动；mom_highpass只用D的成熟高通控制。每arm同9216新增可学习参数，仅up可学习，无上游参数首步/二步梯度链可声称。固定核buffer不训练。所有原参数对象、数值、随机数状态和初始logits保持；up首步及第二步正，所有梯度有限，原Mom梯度另检。

所有blur为stride1，原下采样未替换。B成熟低通不意味着最终非线性高通残差是低通或抗混叠；不证明整网等变、临床定位或稳健。中心化q不使qD或LN后残差空间均值必为零。replicate边界不同于周期FFT，FFT oracle用replicate扩展再线性卷积，不拿循环卷积冒充边界等价。

成熟来源Richard Zhang ICML2019 Making Convolutional Networks Shift-Invariant Again： https://proceedings.mlr.press/v97/zhang19a.html ，官方 https://github.com/adobe/antialiased-cnns 。原Adobe2019源码与CC BY-NC-SA4.0许可保留references，非商业研究适配，不能误称MIT。不执行第三方整网/训练入口，只受限AST提取BlurPool.forward，通过构造replicate/3/stride1对象比较输出与输入梯度；不是全法复现，成熟核/组合不自动有SCI新颖性。

CPU无CLE图像：独立NumPy显式replicate卷积及完整descriptor、独立FFT线性卷积、官方受限算子oracle、double gradcheck、常量/1×1/单轴/矩形退化、批独立、同完整通道边缘分布异布局反例、四kind两次合成SGD更新；三seed×plain/mom_only/五arm共21个公开224初始化核logits/RNG/原参数对象数值/有限反传/真实去Mom/预算。

预算含1×1和固定F.conv2d每次9MAC/位置/通道、Mom三额外方向卷积，以2运算/MAC。候选新68,640,768操作，raw highpass新63,221,760；原Mom累计候选27,850,179参数/9,180,481,536操作，去Mom27,831,651/8,978,171,904，原Mom+highpass为9,175,062,528。不计LN/开方/中心化/逐元素/pad/访存，必须同时报告真实CUDA吞吐，不称严格全算子FLOPs相等。

CPU通过才创建独立正式目录。正式五arm×45/46/47同Q/M recipe CE60epoch/B32/encoder1e-5、新模块head1e-4/AdamW.05/cosine0/clip1/AMP/HV翻转/rotation15/random224/stochdepth.1/workers2/thread1，不改AC数据/标签/等级/患者split/白名单/固定B0。先正式CPU和CUDA32每arm两有限更新、六原Mom/B0完整427val概率严格0误差、缓存SHA/拒绝test训练/2077索引/6GiB余量/97%最佳内最小并发预检，通过再训练。train→val最早最佳→test反馈下一设计；test均自适应内部开发，41患者20k bootstrap仅未校正设计选择偏差条件CI，不称外部或独立泛化。


## 正式15模型与直接比较

五arm×45/46/47全部60epoch；原Mom/B0三seed复用不重训。预注册候选对原Mom/去Mom/rawgate/misaligned/highpass/B0两mode12直接整数Fraction比较，其中C4对B0也使用B0_C4；固定B0 single累计数值另在完整强参照表报告。41患者20k条件CI，保留ECA/普通旋转增强/DINO/Swin同C4参照，不倒置比较为候选对父模型贡献。3/200只是历史数值审计字段，不是单点准入/淘汰门槛。全部seed/普通阴性保留，原Mom保留。

正式CPU/CUDA32各arm两次有限更新只检查唯一up权重梯度首步和次步正、原Mom另正；固定核没有可学习上游，不搬用旧Focal down/core链。新detail参数全部head1e-4组。全六参照427val完整概率误差严格0。固定2077训练索引及缓存/test拒绝不变。每完整目标实测并发、6GiB余量、最佳97%内最小并发通过才训练。

首实际完整权重固定val首图2386 CLI：事前选择native CUDA卷积后端(cudnn.enabledFalse/benchmarkFalse/deterministicTrue/allow_tf32False)，原AMP FP16、真实物理B1、不填充B32，与训练默认cuDNN B32保存概率比较，容差预设1e-4且类别同。这个后端选择在本轮训练和使用结果前冻结；不是声称native与默认所有图模型都等价，也不是以native单图耗时代表训练profile。失败保留并只修分析，不重训有效权重。
