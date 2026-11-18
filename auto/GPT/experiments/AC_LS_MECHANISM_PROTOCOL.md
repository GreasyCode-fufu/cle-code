# A–C第二轮：LS参照的新随机种子复核与机制对照

首轮普通ResNet34为.968521、CLE对比度上下文候选.972476、卷积控制.972734、去对比度.974248、LS方程参照.977491。候选四个预定家族区间均跨零，不能证明对比度/新结构有效。停止继续枚举相似CLEContext组合。LS较高是开发线索，需补足参数控制和新训练随机性复核；本轮不将其称为已确认晋级。

冻结原A–C train2077/val427/test576不变；本轮只加载train和val。三个新固定训练seed45/46/47，不挑种子。五流程各三seed，共15新CNN：plain、conv_ls_budget、ls_signed、ls_positive、ls_global。plain是普通ResNet34同等前向；ls_signed严格复用首轮LS预测器/GroupNorm/有符号权重/通道模共享核/BN和.1残差，在layer3后。conv_ls_budget沿相同PlainControl公式，宽度在32..256步8中按新增参数最接近LS94432选择并冻结。不是与首轮44808参数候选匹配的旧卷积控制。

ls_positive只将同形状的GroupNorm输出在9个核位置softmax，参数/通道/感受野不变；ls_global将原有符号预测核在空间平均后广播到各位置，仍是每图输入条件核，去掉逐位置自适应，全部预测器仍训练。两者与ls_signed参数初始化逐位相同，函数不同不要求初始开启分支输出相同。它们是重训消融，不是对最佳checkpoint的关闭诊断。全骨干/分类头配对相同，关闭残差输出均等于普通网络。

全部同公开ImageNet ResNet34、224输入、原256缓存/增强、60epoch/batch32、原AdamW1e-4wd.05/cosine0/动态AMP，无clip/早停/等级权重；直接复用不可改core.run_worker。CPU线程1、loader2，不导入run_multiview_study。GroupNorm后有符号权重不是概率；不做医生质量/病灶定位解释。LS为成熟方法方程参照，非完整官方LSNet或Triton性能复现；新消融不包装新颖网络。

在新输出前冻结四比较：ls_signed对plain和conv_ls_budget为共同主要比较，另对ls_positive和ls_global为机制比较。39患者组共享20000bootstrap RNG20261024；普通95和四比较家族分位[.625,99.375]。主要两项均三个seed差正、均值至少+.5百分点且家族下界正才称内部重复支持；此新判据不能回头改首轮。若仅对plain有效，不能排除容量/普通卷积解释。某机制贡献需其对应重训对照家族下界正，其他比较不能替代。均只描述已按val选出的模型，同一患者反复开发、新训练seed不是新患者或独立临床确认。

预检查同骨干/头、ls_signed与原LS参照相同state及AMP输出、三核法相同参数初值、正核和为1/全局核空间不变、禁分支等价、真实batch32有限AMP梯度与更新、参数/FLOP/峰值显存。全部源/config/清单/任务/协议提交冻结后再训练。每模型保存60行日志、最佳checkpoint签名/SHA、427图on/off概率/指标/诊断。独立审计要重建全部15模型和禁分支全val推理，复算患者区间、普通BN计数/有限参数/源漂移/最早最高val选择/数据患者隔离和所有权。无test评分。

资源初始6并发，可在GPU持续<80%、CPU<75%、显存余量足时增至8，GPU已100不盲加；每2秒记录真实CPU/单核/I/O/内存与GPU，0.5秒补队列。保留其他Agent。完整阴性/报告/审计/代码/配置到outputs/CLE_AC_dataset/architecture_round2，权重缓存留服务器。最终候选冻结后统一测固定A–C test576，不按test选结构或重新筛清单。
