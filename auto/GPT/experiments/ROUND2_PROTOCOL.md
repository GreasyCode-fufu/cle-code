# CLE结构研究第二轮：上下文约束的局部证据汇聚

## 首轮结论与机制变化

第一轮18次训练全部完成，提交b32e225。A–C策略下，普通卷积、LSConv参考、CLE改造的val均值分别0.873465、0.875350、0.868542，相对同种子原网络分别+0.105、+0.294、-0.387个百分点；等级加权下分别0.874385、0.881937、0.873220，相对原网络-0.757、-0.002、-0.874个百分点。所有患者配对区间跨零，没有可靠结构收益，不宣布成功。

支路特征扰动约42%～64%，而同checkpoint关闭模块通常只改变约1%～3%的预测。不能将失败继续解释为“零门控未打开”；表征改变大也不等于任务收益。

随后使用6个冻结基线checkpoint，仅对完整val做无训练诊断。大多数错误图片仍有部分位置响应现有标签（如加权seed42为65/69），但直接改用GeM或按类别取最强20%响应一般降低val F1。位置的感受野互相重叠，并非独立病理局部真值。这个结果不证明GAP有错，也不能证明重训练GeM无效；它支持下一轮检验“受全局上下文约束的学习式汇聚”，避免盲目强化极值。

## 冻结设计

数据策略固定为等级加权，依据此前开发集均值0.8820高于A–C的0.8724，而不是依据test排序。所有标签、图片、患者分组、权重与原数据研究不变。输入224，原ResNet34、AdamW与60epoch协议、种子42/43/44、checkpoint选择均不变；本轮仅对完整538张val评估，不对test推理。

6个候选各3种子，共18次训练：

1. gem_control：从p=3开始学习有界GeM指数，作为简单池化对照，不称新机制。
2. attention_control：门控attention汇聚，采用与主候选近似匹配的参数量，并有相同的有界GAP混合。
3. cle_evidence：layer4语义、对齐的layer3形态特征、图像全局上下文共同决定空间权重；增加Haar细节能量作为可学习正/负提示。输出始终保留至少50%的原GAP描述，每通道混合系数受限，防止无约束极值代替全局证据。
4. cle_no_frequency：去掉频带提示，其他机制相同。
5. cle_fixed_blend：GAP与区域汇聚混合系数固定为0.25，检验自适应混合。
6. cle_no_context：去掉图像全局query条件，检验全局上下文的作用。

Haar细节不是质量标签，高频可能是形态也可能是噪声，因此提示系数可正可负。所有推理信号仅由输入图像生成，不读取A–E、病理或真实类别。局部feature grid也不是监督的病理区域。

## 模块库关联与对照来源

- shared/module_library/61的WTConv提示对频带分别建模；本轮只独立实现固定正交Haar分解作为形态提示，不称完整WTConv复现，也不把小波基础运算称原创。
- Wavelet Convolutions for Large Receptive Fields，Finder等，ECCV2024：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07137.pdf ；官方代码https://github.com/BGU-CS-VIL/WTConv（先前已核验MIT许可）。
- attention控制的成熟机制参考Ilse等，Attention-based Deep Multiple Instance Learning，ICML2018：https://proceedings.mlr.press/v80/ilse18a.html 。不将attention pooling本身包装为新方法。
- 前述两项不构成对新组合有效性或原创性的证明。主候选必须优于匹配attention/GeM并通过消融、患者确认，才能讨论CLE特异贡献。

## 工程与验证

新模型models/optimize/cle_evidence_pool.py；新入口experiments/run_pooling_study.py；输出experiments/pooling_study/与weights/pooling_study/。原数据研究与第一轮脚本、模型、配置不修改。复用之前验证过的开发集基线预测，记录它们的SHA256。

先验证Haar偶数/奇数尺寸能量保持和梯度、AMP稳定、attention归一化、GAP占比下界、频率系数非零梯度、同seed骨干/分类头初始化及关闭模块等价。每次训练后保存完整val逐图预测、混合系数和关闭汇聚模块的同checkpoint诊断；后者不能替代从头训练消融。

所有完成标记带原训练脚本、第一轮框架、新驱动与模型源码、冻结数据协议与分级清单的哈希。每项结果Git提交。并发2、CPU线程2；不改其他Agent工作区或进程。

## 决策

报告所有候选、同种子差与患者分组区间，同时比较主候选对GAP、GeM、匹配attention及三个消融。它们仍是开发集选优后的探索性结果，不能用未校正区间宣称普适收益。

若有机制一致的正信号，下一步进行严格患者级确认与数据×结构交叉；外层确认中的数据分级须在外层训练折内部重做。若没有正信号，保留负结果并检查分辨率/标签粒度/患者分布等替代解释，不继续凭最高单点叠模块。最终冻结候选后才可单次审计完整test，保留历史test已使用的限制。
