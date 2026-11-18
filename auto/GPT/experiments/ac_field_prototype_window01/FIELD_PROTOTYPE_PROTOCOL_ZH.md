# 光学视野支持约束：数值原型协议

本目录只做CPU算子与配对预处理验证，不训练/筛选CLE分类器，不更换尚在资格比较中的基础baseline。正式结构试验必须等四家族公平资格比较全部完成并独立审计、selected_baseline冻结后，再单独预注册配置/任务/三seed与必要预检。这里的算子原型通过不能被当成性能收益或目标完成。

## 开发证据与可证伪机制

冻结AC train/val2504张图RGB三通道完全相同；黑像素比例中位约23%。固定单位圆在256缓存中覆盖78.534%面积，仅在2077张train上核验，圆外平均亮度中位约3.68e-5，圆外亮度>.02的比例中位为0。这支持近似圆光学视野的几何先验，不是医生区域标注。圆内仍可能有大量暗组织，故不能以“亮度低”作为无效掩膜，不能筛图或改标签。原始固定AC清单和患者split不变。

假设：成像视野外区域与视野边界使局部卷积/窗口attention及全局池化形成不适合CLE的统计。利用和图像同步增强的固定几何支持，约束空间混合与聚合，可能比继续堆末层读出更有效。该假设不同于旧亮度/对比度路由和跨深度二阶聚合，但仍待训练与消融证据，不声称掩膜归一化/注意力/残差本身新颖。

## 原型定义与参照

- 几何支持为像素中心落在单位圆内，中心(0.5,0.5)、归一化半径1，面积近π/4；不依赖图像强度或标签。支持下采样用area平均，故边界为分数支持。
- 卷积参照为NVIDIA原PartialConv的偏置校正和有效样本数归一化：对输入乘mask后卷积，去bias、乘kernel_area/(count+1e-8)、加bias、乘clamp(count,0,1)。这是原算子公式参照，不是整套inpainting网络复现。共享21号FasterNet通道部分卷积是另一种算法，已独立核对，不混用。
- 实验卷积把计数比上限暂固定4，使用tanh(alpha)的零起点残差与原卷积融合；候选基线初始前向完全一致。完整比率参照和去局部归一化、去真实圆视野将作为未来重训控制；不能只靠关闭支路代替重训。
- Swin-v1原型沿用其窗口划分、移位、相对位置bias和既有QKV/projection；对有效key按支持权重重新归一化softmax，完全空窗口用有限退化并把无效query输出置零。无mask应复现官方原注意力；显式全1图像支持仍会排除人工padding，因此只作为无光学圆视野的padding控制，不声称和原模型完全相同。
- 窗口attention仅支持资格比较中的Swin-v1且内部dropout为0，避免额外随机抽样改变paired训练。Swin-v2余弦注意力不在本原型适用范围。
- 支持加权pool取sum(x*m)/sum(m)，全空支持输出0；未来比较pool-only、无pool和完整局部支持适配。

预计正式机制对照包含：原冻结基础baseline、仅支持pool、成熟算子方程参照、零起点残差完整设计、同结构无圆视野、去局部归一化、去pool。最终适用的层与计算预算在基础主干冻结后确定；现在不提前跑结构训练，也不根据新test选择任何机制。

## 数值检查任务

CPU FP64：partial卷积与独立标量NumPy公式及实际NVIDIA算子比对，覆盖偏置、group/depthwise、stride和空/满/分数支持；梯度检查；残差alpha=0与原算子逐位一致，alpha梯度非零且两步更新有限；weighted pool独立公式和空支持；Swin多个奇数/小于窗口尺寸、shift/非shift，无mask前向/梯度与安装版torchvision一致，支持归一化与官方注意力加空间key bias的独立参考一致，空窗口前向/梯度有限。

真实冻结train图：RGBA携带几何mask后，完全相同Torch seed下RGB增强结果必须与原RGB流程逐位一致；两个食谱均检查，不能因增加mask顺便换输入或增强。所有用到的图片索引显式限制AC train；不访问test。若RGBA导致不等价，则改为显式共享同一组增强参数，并记录修订，不能放宽等价检查。

通过后保留源码SHA、协议、结果及许可；以后正式模型仍需真实batch32 CUDA AMP、完整基线val复现、参数/计算和训练索引预检，不能拿CPU原型替代这些门槛。

来源：NVIDIA PartialConv作者代码和BSD3许可已归档在上一资格比较目录references_official；torchvision安装版源码与版本/文件SHA将记录在预检。官方路径为 https://github.com/NVIDIA/partialconv/blob/master/models/partialconv2d.py 和 https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py 。
