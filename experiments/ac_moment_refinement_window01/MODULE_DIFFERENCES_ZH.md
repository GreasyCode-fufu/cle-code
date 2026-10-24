# 来源与结构差异

C4MomentStem逐字节复用models/optimize/ac_module_window01/c4_moment_stem.py，其旋转lifting、群均值为成熟概念；零投影的均值/标准差预训练适配属于当前待验证实现。原三seed单视图改善是继续研究的信号，尚不能替代mean_only/random_moments重训；不宣称整体网络严格C4等变。

ECA逐字节复用已核验共享抽取定义，k5、gamma2、b1，与[官方显式k5](https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py)一致。moments_eca只增加该成熟通道交互，对原public ECA初始化保持配对，不称ECA为新发明。

mean_only将同一方向均值拼接两份，参数和卷积量与完整moments相同；random_moments使用已冻结的固定空间排列，初始化/RNG不改。它们只用于机制消融，不被当作推荐架构。

aug_moments使用此前普通C4增强同一独立Torch RNG种子规则，每图每epoch仍一次，基础RGB增强序列的全局随机状态不受新增quarter-turn抽样影响。所有类采用fork_rng构造新增权重，原公开encoder与同seed分类头保持一致。

独立审计将复用原M修订2的逐图重建函数，绑定到本独立实验目录和模型工厂；封装源码及原函数SHA锁定，原M文件不改。新汇总另计算与原moments、ECA、普通旋转增强的机制配对差，所有结果保留。两个既有分支的源码复用是可复现性措施，不是创新性证据。
