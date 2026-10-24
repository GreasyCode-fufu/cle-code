# 本次患者诊断与成熟方法的边界

本文件补充方法来源，不改变已冻结配置、源码、诊断或判定规则。

|参照|原方法的相关内容|本次实际操作|不应作出的结论|
|---|---|---|---|
|Prototypical Networks|式(1)类支持集特征均值，式(2)负距离softmax；使用episodic训练学习embedding|固定已有W0的CE训练embedding，仅train拟合均值，固定平方欧氏距离和温度1；另一诊断把估计器换为患者×类别均值的等权平均|没有重现完整episodic训练；不能称新的神经网络或原论文完整复现|
|SupCon|利用同类标签构造有监督对比学习|本阶段没有实现或训练SupCon，仅将其列为未来若采用类聚合机制时需保留的成熟参照|不能把类聚合/对比思想本身称为新颖；也不能隐藏额外增强视图或训练阶段的预算|
|Group DRO|对预定义群组的风险进行稳健优化；官方代码和论文强调正则化的重要性|本阶段没有DRO梯度更新，只审计患者权重和类内特征方差|患者偏斜不证明DRO有效；多类别患者也不能强行单标签化|

公式已核对[Prototypical Networks原论文，第2.2–2.4节](https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf)。固定embedding上的平方欧氏原型分类可写成受约束线性分类，不能仅凭“原型头”名称包装结构创新。

本次两种原型使用相同固定embedding、相同距离和相同隐含类别先验；差别仅是train类均值的患者权重。与原学习线性头比较，还涉及参数化约束和先验，不能把全部差值归因于患者不变性。原型统计用raw特征；方差分解单独使用L2归一化特征，二者没有混用。

train中71/182患者组、val中11/39组含多个类别。患者等权准确率仍按每患者内部逐图准确率求均值，不对患者强制多数类标签。患者×类别原型不跨类别平均；所有图像标签、等级、患者split、阈值及清单保持原样。

只在固定train/val进行本次计算，但W0先前使用过test反馈选择；encoder见过所有train患者，不能将该分析称为患者OOF、完全无test筛选依赖或独立验证。未完成全部三seed不报告最高seed，描述性方差不证明因果。没有稳定收益就保留阴性，不据此自动扩展近似原型组合。

其余主要来源：[SupCon原论文](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)、[Group DRO作者代码](https://github.com/kohpangwei/group_DRO)、[Group DRO作者论文](https://arxiv.org/abs/1911.08731)。
