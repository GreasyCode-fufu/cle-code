# 同层通道协方差谱描述：无CLE图像CPU原型

LAG全轮10门槛均false，完整对原Mom未确认增量，累计对错序/重复/成熟ACF区间均跨0；保留原CNNMom。下一假设为末层同一空间位置上的通道联合变化，补充stem方向统计。既有CROSS是stage3/4跨层相关；本次为单层对称协方差，不能把不同实现混称旧结果复现，也不能忽略旧阴性。

末层768×7×7经共享类型但独立实例的768→16无bias可学习1×1投影，共12288参数。逐通道空间中心化后C=XXᵀ/49。主分支对C+1e−4I做trace预归一、5次Newton–Schulz矩阵平方根迭代、sqrt(trace)后补偿，再减同迭代的固定ridgeI平方根floor。取上三角且非对角乘sqrt(2)，136维经无仿射LN eps1e−5、零136→3投影（411参数）加原logits。原avgpool/learned norm/head保留，新增总12699参数。floor只固定平移以令平坦描述为0，不增加信息；有限5次是平方根近似，不称精确eig平方根。

五arm：mom_covsqrt；covsqrt_only真去18528Mom；mom_rawcov相同投影下直接C、同136/LN/零分类器；mom_diag先清空非对角再同ridge/5次迭代，仅保留16个有效对角描述，上三角其余坐标恒0，名义参数相同但有效容量并不相同，明确作为去通道耦合控制；mom_channelperm对每个投影通道施加独立固定49位置置换（np.default_rng(20261025)），保留全部单通道边缘分布但改变联合配对，再算相同协方差平方根。mom_only/plain只作公开初始化参照。

协方差/矩阵平方根/谱归一是成熟方法。主要来源：[作者iSQRT-COV页面](https://www.peihuali.org/iSQRT-COV/index.html)、[作者官方实现](https://github.com/jiangtaoxie/fast-MPN-COV/blob/master/src/representation/MPNCOV.py)。官方source blob400cdf16469d9a729fda31a4a310708fd365b895，MIT LICENSE blobfcd5240305399993bb1479b8ac45cc147e77c991，均保存原字节/归属。本地无BN/ReLU投影、ridge/floor、Frobenius加权上三角、LN及保留原GAP的零logit分支是明确适配，不是官方ImageNet整网或全部MPN-COV复现，不据组合称新颖。官方数值oracle仅审阅后受限提取Covpool/Sqrtm，不运行其训练入口。

全局共同空间排列/旋转不改变该描述；独立每通道置换改变联合统计。这不是整CNN等变或病灶定位/临床拓扑。rank16与49样本、ridge绝对尺度及有限迭代均有限制；不保证稳健影响函数。LN后不保留协方差正定矩阵含义，不称黎曼网络。原始C半正定、加ridge仅对迭代矩阵确保正定，减floor/取向量后的表示另当别论。

零分类器首步使上游新增1×1投影梯度为0，这是链式法则，不伪称首步各投影梯度全正。必须检查分类器首步非零、第二次更新后1×1投影非零且所有梯度有限；保留Mom的梯度单独检查。新矩阵乘法MAC明确计入，排除统计中心化/LN/逐元素/访存；固定floor在构造时预计算、不计前向。

CPU阶段做独立NumPy公式/5步迭代/eig误差、官方受限AST数值/梯度、同边缘异耦合反例、gradcheck/退化有限/批独立/两合成更新、三seed×七变体公开起点logit/RNG/原参数对象一致。所有CLE训练/val/test图均0。只有通过后再独立正式注册、原M同recipe 60epoch/三seed，六完整427val真实权重重建、CUDA32两更新和吞吐预检通过后才开训。标签/等级/split/采样均不改，原Mom及固定B0三seed有效参考不重训。全部test自适应内部开发，条件CI不校正选择偏差；整体active，不保证提点、SCI新颖或录用。
