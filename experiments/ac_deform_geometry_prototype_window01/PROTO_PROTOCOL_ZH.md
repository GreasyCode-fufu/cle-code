# 保留方向矩后的局部几何采样：CPU原型预注册

本原型只做合成张量/公开预训练网络数值核对；CLE图像、test概率、新CLE训练均为0。当前跨尺度结构轮照原配置完成全部控制，最终决策依完整结果。本轮不是结构有效性证明。

依据：训练图可见弯曲亮边界与大小不一暗区。固定train阈值的val纹理诊断已审计：高std组107张/21患者，B0平均macroF1约92.974%，全val98.103%，但类别混合/背景/少量错误及多重分组均影响解释，不把相关性当因果。全局跨层相关目前无可靠增量，下一假设是保持已训练局部纹理卷积权重、用受约束采样调整局部感受野，可能优于额外全局统计。此处“可能”需后续受控重训。

原型算子保留ConvNeXt stage3最后一个7×7 depthwise原Conv对象和参数。3×3卷积预测六个位置参数，零初始化W和bias；0.75tanh约束平移，角度±pi/8，log尺度±log1.5，剪切±0.25，令A=R(theta)*[[exp(sy),shear],[0,exp(sx)]]。对每个输出位置的7×7采样格，用A*r+t−r产生98个交错(y,x)offset。det(A)=exp(sy+sx)>0，只约束每个局部采样基的方向，**不保证全局位移场无折叠、整网等变或临床拓扑正确**。

输出=原Conv(x)+D(x,offset,W)−D(x,zero,W)，D为已安装Torchvision官方DCNv1零填充算子。两次D均保留梯度，offset0时它们的前向严格相消，避免公开预训练配对起点受不同卷积实现数值差异影响；不靠detach修改encoder有效梯度。残差FP32（double检查保留double），原分支保持原AMP；双D额外计算/插值/访存成本必须实测并披露，不称免费。

合成控制：permuted_affine对相同六参数生成的49个offset点用固定seed排列，完全同参数/offset边际范数，破坏局部仿射对应；free以98通道无约束offset保持DCNv1成熟算子参照，参数更多要明确列预算。若正式研究获准自动继续（用户已总体授权），应新独立源码/config，保留原M方向矩，重训full/去方向矩/等参数打乱/成熟DCN参照，全部至少三seed，不修改历史prototype。

CPU核对：NumPy直接双线性零边界、独立grid_sample输出与输入/offset/权重梯度、double gradcheck、局部det>0、zero精确前向及原参数ID/RNG、全公开Tiny224+方向矩初始logit完全一致、有限反向、两个合成优化步骤。共享LDConv/AKConv与作者官方model.py AST对齐仅作源码审计，不能默认为标准DCN语义；检查offset bias实际是否零、LR hook实际效果、边界重复权重。FADC官方/共享源码和依赖差异记录，环境mmcv当前缺失，不装旧Torch替换历史环境，也不将自写DCN称FADC完全复现。

成熟先验：[DCNv1](https://arxiv.org/abs/1703.06211)、[LDConv](https://arxiv.org/abs/2311.11587)、[FADC](https://arxiv.org/abs/2403.05369)、[AffineFormer](https://openaccess.thecvf.com/content/CVPR2024W/IMW/papers/Chen_Affine-based_Deformable_Attention_and_Selective_Fusion_for_Semi-dense_Matching_CVPRW_2024_paper.pdf)等已涉及可变形、可调感受野或局部仿射采样；不能把这些概念本身声明为新颖。当前只是CLE适配的待验证设计，论文新颖性需更充分检索/实验，不保证SCI创新或录用。
