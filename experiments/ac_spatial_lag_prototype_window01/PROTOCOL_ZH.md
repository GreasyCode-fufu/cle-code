# 两间隔空间关联：无CLE图像CPU原型

ODD全15模型的10个固定B0数值门槛均false；候选对原Mom未确认累计增量，对同容量raw3/std/even控制也未确认优势。因此不把末层有界奇统计或这些控制带成已确认组件。原CNN方向矩工作机制保留。本轮假设：ODD类空间排列不变描述会丢失邻接关系，最后特征图的两种空间间隔关联可能补充原stem方向均值/标准差。此为完整结果后的新假设，非事前ODD假设或临床因果解释。

设末层B×768×7×7特征按每图每通道全空间均值中心化c。对间隔l=1、3，分别取不环绕、不padding的水平与垂直有效重叠对a,b。候选每轴rho=mean(ab)/(sqrt(mean(a²)+1e−6)*sqrt(mean(b²)+1e−6))，两可用轴等权平均，拼接两间隔共1536维，经无仿射channel LN eps1e−5和零1536→3投影加原logits。原avgpool/learned norm/head完整保留，新4611参数；保留Mom另18528参数。没有学习前即改变公开logit的池化替换，没有CLE checkpoint热启动。

五arm：mom_lag原Mom加完整描述；lag_only真去18528Mom；mom_permuted对原7×7特征按固定种子20261025的49位置置换后计算同一描述，保持每通道全部边缘分布/空间矩，破坏邻接；mom_duplicate把同一lag1重复两份，去掉第二空间间隔但保留1536维/同参数，重复只优化重参数化、无新描述能力；mom_acf用同有效对sum(ab)/(sum_all(c²)+1e−6*HW)这一成熟全能量ACF归一控制，仍水平/垂直平均并拼接两个间隔。所有控制同无仿射LN及4611零投影。mom_only/plain仅原型初始化参照。

候选是全局中心化后的重叠向量归一内积，不是重新逐对中心化的Pearson相关，也不等于NIST全能量归一。LN前rho在[-1,1]，LN后没有该界。有限eps仅近似正比例缩放不变；标准化并不自动稳健。两轴平均对未置换的描述在90度旋转/反射下保持；固定置换控制不具此性质，普通CNN整网也不保证等变。不声称病灶定位、真实临床拓扑、某像素/病理尺度或方向被保留；平均会丢失绝对位置与方向。lag3只有少量重叠对，估计方差可能较高，需明确消融。

成熟来源：[NIST autocorrelation](https://itl.nist.gov/div898/handbook/eda/section3/eda35c.htm)给出延迟乘积与全中心能量归一，mom_acf是该一维公式在两个空间轴的明确适配，非原文2D医学方法；[GNPP作者页面/摘要](https://arxiv.org/abs/1607.06514)已研究神经响应空间共现，证明空间关系作为方向并非新概念。这里没有复现GNPP算法或声称其作者代码验证，未复制第三方实现。零残差、LN、自相关及组合不自动创新。NumPy零padding FFT只作为线性空间相关分子的独立数值oracle，不是学习时傅里叶模块，也不同于POWER的四方向DFT统计。

本阶段只做NumPy显式有效对/零padding FFT/成熟ACF公式、同边缘分布异布局与同lag1异lag3反例、双精度gradcheck/退化有限/批独立及原公开初始化检查，CLE图/训练/val/test图均0。CPU通过后仍需独立正式注册、完整CUDA batch32与六完整427val概率重建、缓存隔离白名单和吞吐profile。正式若实施，仍是原AC不改标签/等级/split/采样，原M同recipe CE/60epoch/三seed，val选权重→test反馈下一设计，累计/去一/错序/去第二间隔/成熟ACF对照。整体active，不保证提点/SCI新颖/录用，所有test为自适应内部开发，患者bootstrap只为未校正选择偏差的条件区间。
