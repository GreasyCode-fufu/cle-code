# 患者差异证据预注册：CPU、固定train/val、无新网络训练

主GPU正在进行独立冻结的H1几何支持结构实验。为后续机制收集可判定证据，本目录仅用空闲CPU、原W0三个已审计checkpoint提取train/val中心裁剪特征；不修改H1或任何旧源码/权重，不接触test图像，不换baseline。

已只读确认：train2077图/182患者，每患者1–102图，最多10患者579图；val427图/39患者，最多10患者296图。train有71个多类别患者组，val有11个。因此患者不等于单一标签，不能把所有图像平均为一个疾病标签，也不据此改标签、分级、清单或等级。

机制问题：图像均匀训练可能偏重高样本患者；在同一类别内，表征是否仍显著按患者变化？患者等权而非图像等权的类原型是否在固定val提供更合理方向？这是待检验假设；样本数量偏斜本身不能证明表征捷径或泛化因果。

执行固定：W0按正式H1已冻结工作起点复用（当前swin_t_B），三个seed45/46/47，同一完整训练checkpoint SHA；CPU float32 eval，batch16、2 loader workers、固定中心裁剪，不使用训练随机增强。CPU线程仅测2/4/6并选总吞吐97%最佳以内的最少线程，自己的进程nice10，CUDA_VISIBLE_DEVICES为空。预注册/任务/来源先Git；独立raw-SHA/resize/crop张量、test拒绝、混合标签患者的原型加权算术预检通过并提交后提取。没有新网络优化、梯度或额外训练epoch。

提取每图最终分类前特征、原head logits、标签和原global index；全部2504图顺序和身份精确。保存源/配置/checkpoint/feature SHA、有限值、网络state不变、吞吐/CPU/资源，完成seed优先复用有效特征，不重复提取。特征大数组服务器保留、SHA和证明入Git。

诊断严格固定两种原型：A为每类train特征图像均值；B先按train的患者×类别求均值，再按患者等权求该类均值。使用raw特征和平方欧氏距离、温度1，无超参搜索、无val估计原型、无test。固定线性head CPU推理为参照。记录全部三seed val macroF1、accuracy、恶性召回、患者等权accuracy、混淆矩阵，及39患者配对条件bootstrap。它是已有原型距离公式的受控诊断，并非原始episodic Prototypical Networks完整复现，也不是新的网络创新。

另在L2归一化train特征上按类别计算总类内方差=患者×类别内部方差+同类患者均值间方差，做数值等式校验。记录image均值/患者等权均值漂移。将N²/Σn_g²解释为图像权重的患者集中度等效数，不称独立样本量。

独立计算审计用保存特征和checkpoint线性权重重建logits、用反患者×类别计数权重重建等权原型、复算方差分解/指标/预测身份和SHA。CPU/GPU已有val概率差异只记录精度影响，不用CPU替换冻结比较结果；预测一致率若低于99%先诊断。若等权原型没有稳定收益，保留阴性、不把其包装为已有效机制，不自动枚举变体。

局限：W0先前由test反馈选择，故即使此脚本只读train/val也不能称完全无test筛选依赖；encoder见过全部train患者，本分析不是OOF或独立验证。现有A–C选择和历史test标签依赖照旧披露。

成熟参照：[Prototypical Networks NIPS2017](https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html)、[SupCon NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)、[Group DRO官方代码](https://github.com/kohpangwei/group_DRO)。后续若依据证据采用相关思想，需忠实参照、参数/计算预算与重训消融；不因论文需要2–3点就认定本诊断是创新。
