# 患者差异机制证据：仅train/val，CPU分析

本阶段不更新网络、不改变H1训练，不读test图像。W0先前按test反馈选择，故此分析并非完全没有test筛选依赖。已拟合encoder见过全部train患者；这里没有患者OOF或独立验证。原型只采用固定train类均值和距离公式作诊断，不是原始Prototypical Networks完整episodic训练，也不声称创新。

修订记录：v1复用了test helper，抽样本身使用实际39个val患者，但返回元数据将患者数写死为41。v2用独立混淆计数重建39患者bootstrap，逐项确认原区间数值不变；原v1源码/报告/审计另存，不重新提取特征、不训练模型。

## 清单证据

train：2077图/182患者；每患者1–102图，最多10患者占579图；71患者含多个类别。图像均匀权重的患者集中度等效数69.734，只是权重集中度，不是独立样本量。
val：427图/39患者；每患者1–71图，最多10患者占296图；11患者含多个类别。图像均匀权重的患者集中度等效数14.700，只是权重集中度，不是独立样本量。

因此不能把每患者当一个固定疾病标签，也不能将其全部图像直接平均后赋单一标签。这里只在每个类别内部构造患者均值，再按患者等权汇总；图像清单、标签、A–C等级完全保留。

|固定诊断|seed45 val F1|seed46|seed47|均值|患者等权accuracy|
|---|---:|---:|---:|---:|---:|
|frozen_linear_head_cpu|97.6955%|97.6955%|97.9009%|97.7640%|96.6170%|
|image_mean_euclidean|97.4162%|97.6955%|97.3615%|97.4910%|95.9888%|
|patient_class_mean_euclidean|97.4162%|97.1194%|97.6460%|97.3939%|95.8819%|

各类别同类方差中患者间部分占比、逐seed混淆矩阵、CPU/GPU数值差异与条件患者bootstrap见JSON。若患者等权原型没有稳定收益，不能把患者原型机制写成有效；此证据只用于决定下一机制，不自动启动近似组合搜索。

## 成熟方法参照

均值原型距离是已有方法：[Prototypical Networks，NIPS2017](https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html)。这里仅借用固定embedding下的类均值距离，不复制其完整学习流程。
有监督同类聚合已有[SupCon，NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)；群组稳健优化已有[Group DRO官方代码](https://github.com/kohpangwei/group_DRO)。未来若采用相关机制，须保留忠实参照、预算匹配与消融；不能将成熟思想重新命名为已证明新颖。
