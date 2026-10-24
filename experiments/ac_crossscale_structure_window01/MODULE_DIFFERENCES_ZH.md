# 参考方法与改动

[B-CNN论文](https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html)与[官方代码](https://github.com/cvl-umass/bilinear-cnn)的TensorProduct保存原件/许可/blob/SHA；mom_bilinear对同投影特征的乘积/位置均值和BCNNModule表达式sqrt(relu(z)+1e-5)-sqrt(relu(-z)+1e-5)、默认L2逐值及梯度核验。它是忠实算子参照；backbone改ConvNeXt、低rank跨stage及零残差头、固定scale16均为预算适配，不能称完整原版双VGG实验。

[HBP ECCV2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Chaojian_Yu_Hierarchical_Bilinear_Pooling_ECCV_2018_paper.html)及[作者官方源码](https://github.com/ChaojianYu/Hierarchical-Bilinear-Pooling)已检查。其prototxt用VGG conv5_1/2/3三个8192投影，三个Hadamard乘积+sum pooling+signedsqrt+L2，再concat，非本轮16×16跨层外积；已有跨层相互作用和多层融合，不把这些概念声明为新颖。Caffe SignedSqrt前向未加eps、反向有eps，L2平方范数加1e-7；本轮成熟参照明确采用官方PyTorch B-CNN的平滑表达式，不能混称HBP完全复现。HBP源码和BSD/文件版权原文保存upstream，只检查不导入Caffe训练。

iSQRT-COV/MPNCOV的中心化协方差官方代码原件沿用CPU原型存档。本轮不做矩阵平方根/Newton–Schulz，不叫iSQRT-COV。投影跨层中心化标准相关加有界归一与其不同，需要独立对照验证，不自动构成创新。

共享425文件库保持只读；之前发现104c只是示意协方差热图，无NN训练实现，不执行带展示副作用脚本、不作为CLE依据。方向矩文件从已冻结M源码逐字复制，保留public权重和原参数ID，新模块独立目录。与旧CPU原型差异只有明确正式scale16和预注册对照拓展，旧原型及失败记录不覆盖。
