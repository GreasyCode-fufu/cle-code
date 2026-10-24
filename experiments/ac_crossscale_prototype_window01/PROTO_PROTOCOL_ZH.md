# 二阶跨尺度读出：仅CPU原型协议

当前正式训练仍是方向矩细化轮，本文不改变其源码、recipe或任务。此处仅以合成张量与公开预训练初始化测试下一种互补机制的可计算性；没有CLE图像读取、没有新的有效训练或test评价。

假设：CLE纹理/组织形态可能取决于不同抽象层在相同位置的共同激活，单纯空间平均及局部1D通道注意力可能遗漏这些关系。以ConvNeXt阶段3的14×14和阶段4的7×7为输入，前者平均下采样到同一7×7网格，各投影16通道；逐通道中心化、方差正则后计算跨层相关矩阵，采用d/sqrt(1+||d||²)有界读出，经零初始化3类线性层加回原logits。增加19203个参数，不替换原主干/分类头。与方向矩在局部朝向层面的统计作用不同，是否互补尚未知。

这些操作借鉴成熟二阶方法，不能把跨层双线性或协方差本身宣称为新颖结构。未来正式研究必须保留忠实双线性/协方差参照，并对齐参数、数据/食谱，用破坏单侧空间对应及第一阶统计的控制检验机制。不得仅凭原型数值检查宣称CLE收益或SCI创新。

先锁定原方法源码、许可、协议和本源码SHA。CPU检查：原B-CNN TensorProduct前向/输入梯度；原MPN Covpool与中心化公式（区分默认fp32系数）；NumPy相关矩阵；联合空间排列不变、单侧排列改变、常数输入有限梯度、double gradcheck；固定维度两次合成优化及公开预训练方向矩初始化的原logits完全一致。所有检查明确不含真正数据效果。

参考：[Bilinear CNN](https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html)、[官方PyTorch代码](https://github.com/cvl-umass/bilinear-cnn)、[iSQRT-COV作者项目](https://www.peihuali.org/iSQRT-COV/index.html)、[官方实现](https://github.com/jiangtaoxie/fast-MPN-COV)、[GSoP论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Gao_Global_Second-Order_Pooling_Convolutional_Networks_CVPR_2019_paper.html)。本原型不包含矩阵平方根或原GSoP的行卷积激励，不称完整复现这些网络。

库内104c.py虽出现协方差字样，实际是地理字段模拟相关性热图并有Windows字体/显示副作用，不是网络模块，仅读源码、未执行，也未将模拟数值当CLE证据。其余bilinear命中多数为插值，不当成双线性池化来源。
