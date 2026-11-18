# 成熟参照和共享模块差异

四响应C4 DFT功率分解和H-Net幅值均是成熟数学/网络先验。原型upstream保留官方H-Net README、两源码、MIT许可及Git blob，官方TensorFlow1不在当前Torch环境安装。仅白名单stack_magnitudes纯函数AST以NumPy/Torch运算shim执行，原公式sqrt(max(sum_complex(x²),1e−12))输出和输入梯度均对齐。mom_rawh2忠实于这个幅值算子，不能称H-Net整网络、圆谐波滤波器学习或完整训练食谱复现。

提出的q是C4总能量稳定化的频带对比，非官方H-Net函数；permuted保持响应边际/额外参数但破坏循环邻接；duplicate_std不增加新描述信息，反映分支重参数化/优化效应。shape_only真正不建立18528参数的Mom适配器，不能以 unused adapter假装去一；预算差必须报告。

共享FMA源码静态显示空间rfft2/irfft2及实虚卷积，未直接用于方向轴DFFT，不能用文件名/venue标签当本文方法新颖性证据。群池化、DFT不变描述、显微图像旋转学习已有论文，相关主来源见upstream/prototype_protocol.md；部分全文读取403/验证码只记录可核实摘要，不虚称逐式复现。q的C4局部统计性质不保证整个普通CNN等变，eps导致缩放非严格不变；不从算子证明推导临床机制或SCI录用。
