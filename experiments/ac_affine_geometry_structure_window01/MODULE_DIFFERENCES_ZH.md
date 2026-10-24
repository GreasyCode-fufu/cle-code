# 共享库与成熟参考差异

Torchvision已安装官方DeformConv无mask即DCNv1；本轮保留它的零填充、交错(y,x)offset及输入/offset/权重梯度。新设计是低自由度有界局部仿射采样+原方向矩的CLE适配，不把可变形/局部仿射/可调感受野概念称新颖：[DCNv1](https://arxiv.org/abs/1703.06211)、[LDConv](https://arxiv.org/abs/2311.11587)、[FADC](https://arxiv.org/abs/2403.05369)、AffineFormer等已有相关机制。zero-offset双D差分保持公开前向起点、初始encoder梯度相消，全部保留梯度；这属于本轮配对适配，并非原DCNv1论文完整架构。

共享LDConv_2D与原作者官方model.py的类在CPU对齐了输出和输入梯度。差异：官方p_n注册buffer，共享逐forward生成；两者均只zero predictor weight，bias仍默认非零；所谓0.1梯度hook只生成局部generator且无返回，在真实parameter gradient比较中与删hook相同。两者夹紧坐标/邻居后再算插值权重，zerooffset常量输入右下角权重和实测4、内部1；这是原实现共同语义，不能把它当标准零填充DCN或直接声称“零偏移=原Conv”。未据此重训历史模型，也不代表LDConv论文所有应用都无效。

本轮预测器weight和bias显式全0，采用已独立NumPy/grid_sample核验的官方零填充DCN语义。自由offset原始98通道对照使用同框架，额外参数明确报告；同六参数打乱控制保持offset范数分布，区别只在局部对应。

共享FADC类名FADConv，官方为AdaptiveDilatedConv，均依赖MMCV自定义算子；环境mmcv当前缺失，共享ImportError后仅把父类设nn.Module，不能恢复原算子。官方AdaDR/AdaKern/FreqSelect和本轮形变不同，本文不把本轮称为FADC复现，不替换现有Torch来安装其旧环境。来源原件、blob/实际SHA及许可在上游CPU原型存档，正式轮保留引用签名，旧源码全部冻结。LDConv官方根未列许可证，不虚称MIT，原件仅内部参考，不向公共仓库重新发布。
