# 同层通道协方差CPU原型核验

已通过独立NumPy协方差/5步迭代/eig误差、官方Covpool/Sqrtm受限AST输出与梯度、同边缘异耦合反例、gradcheck/退化有限/批独立、两合成更新及三seed×七变体公开起点核验。CLE训练/val/test图均0，尚未新正式训练。

上游新1×1投影首步梯度0、第二步非零；新零分类器首步非零，不能伪称两个投影首步均正。diag名义同参数但只有16有效对角坐标，非等有效容量；5次迭代为矩阵平方根近似，不能称精确eig。

|arm|参数|逻辑GFLOPs|
|---|---:|---:|
|plain|27822435|8.909531136|
|mom_only|27840963|9.111840768|
|mom_covsqrt|27853662|9.113193776|
|covsqrt_only|27835134|8.910884144|
|mom_rawcov|27853662|9.113070896|
|mom_diag|27853662|9.113193776|
|mom_channelperm|27853662|9.113193776|

逻辑2FLOP/MAC含原Conv/Linear、三额外旋转conv、新1×1/分类器、协方差BMM及5次三BMM迭代。固定floor仅构造时预算；中心化/LN/逐元素/访存排除。实际吞吐将正式测量。

官方MIT source与license完整保存，逐字Gitblob核对。模型中的ridge/floor、Frobenius上三角权、LN和零logit残差是明确适配，非官方整网复现；成熟组合不自动创新。

config SHA 467b744fdc7630934c47f0d5e9e4d6b0fca70b40f759a913c6a48e65be1cc321；proof SHA 306dd268fe7f26d0f82cb89f313c7528133bc8a8247193cb2e6bf1ffc9384a46。原CNNMom保留，固定B0不变；全新正式15模型尚须注册、CUDA batch32两更新/真实六427val重建/缓存隔离/profile。整体active，不保证提点、SCI新颖或录用。
