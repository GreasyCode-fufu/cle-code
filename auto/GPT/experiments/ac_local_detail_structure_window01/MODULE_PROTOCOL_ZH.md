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
