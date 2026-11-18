# 模型与共享模块核查

全量静态清单由prepare生成module_inventory.json，记录models和只读共享module_library全部Python文件SHA、类、函数、imports、解析失败。不会执行共享库代码或修改它。会议/年份标签是文件名元数据，未经原文证实的条目不当作真实论文、官方实现或新颖性依据。

现有骨干覆盖ResNet/DenseNet/EfficientNet/ConvNeXt/V2/MogaNet、Swin/ViT/PVT/MobileViT等。Convnext.py标注来自Meta官方，主Block为7×7 DWConv、LN、4倍MLP/GELU、LayerScale、残差，层级下采样和GAP；属于成熟ConvNeXt。MogaNet.py依赖旧mmcv/mmcv.runner，不能只凭源码存在当作当前环境可用。V2依赖本地GRN/LayerNorm，没有已验证预训练迁移不能直接公平比较。部分模型优化支路已经被旧研究测试，应先查旧账本，不能重做有效训练。

本轮主参照选完整DINO，依据开发均值.982784，而CNN为.973445；这是强开发参照，test历史差异不用于选择。现代CNN挑战参照选已有torchvision ConvNeXt-Base ImageNet1K权重，严格原1000类模型load后仅替换分类头，无网络下载或随机主干假装预训练。实际runtime/权重SHA写config，stochastic depth=.1公开冻结；约4倍参数与额外计算单独报告。同一食谱的第一轮结果不等同穷尽调优。

重点源码差异（2026-10-24官方源核对）：

1. 共享44号CBAM：共享通道支路1×1Conv+bias与官方共享Linear+bias在1×1描述子上可等价；共享SpatialAttention的7×7Conv默认bias=True且无BN，而作者代码SpatialGate使用bias=False的7×7Conv+BatchNorm(eps1e-5,momentum.01)，无ReLU。故不把共享文件称官方精确CBAM。本轮没有使用这一未经修正共享分支。作者公式参照：https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py 。
2. 共享43号SE：GAP→降维FC→ReLU→升维FC→sigmoid→乘输入，是SE基本机制的无bias版本；文件内多项“魔改”不因为名字即具有独立论文贡献，本轮没有盲目加入。
3. 共享100号Mona：可见三种DWConv核3/5/7平均、残差与64维投影，try/except吞掉mmcv/mmengine导入失败。没有核验作者版本和完整依赖前，不将其当成忠实Mona参照，暂不用于正式比较。
4. 共享61号WTConv：包含pywt依赖、多级小波/逆小波和小核分支。旧统计/小波聚合已产生阴性，不在缺乏新机制证据时重复堆此模块；本轮没有声称其与原作者代码逐位一致。
5. 已有DINO迁移：官方无register ViT-S/14公开权重SHA由旧研究锁定，strict port只丢不用mask_token，位置编码从预训练网格到224插值；timm使用明确输出尺寸，与官方历史offset=.1插值存在差异，沿用披露，不称官方逐位模型前向。官方定义：https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py 。
6. 本轮official4参照：官方eval/linear.py的create_linear_input将最后n个归一化CLS拼接，启用avgpool时加末层patch均值；n=4则1920维。模型get_intermediate_layers对每层使用同一最终norm。本轮该读出公式忠实，采用共同完整微调与默认Linear初始化，故区别于作者冻结特征、初始化normal(.01)/零bias、SGD搜索的完整线性probe流程。来源：https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py 。
7. ConvNeXt现代参照使用安装版torchvision严格加载其官方权重格式，不混载本地Convnext.py的Meta权重命名。Meta Block与torchvisionCNBlock均含DW7、LN、4倍MLP/GELU、LayerScale、残差；实现组织和状态key不同。来源：https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py 与 https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py 。

自主relational读出属于待检验结构：多深度、邻域通道乘积统计和零起点残差组合，不能单靠组合称“已证明新颖”。必须超过冻结强参照和重训机制消融，报告参数/计数FLOPs/训练及推理时间。它没有更改输入、分级或标签，也不把特征响应叫病灶定位。独立预检将核验baseline等价、配对初始化、边界邻域均值解析参考、真实图像有限梯度；off诊断仅用于检查实际作用。
