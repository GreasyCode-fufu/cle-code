# 配对几何支持加载器原型审计

本任务不训练分类器、不评分候选、不改变尚在资格比较中的baseline。原RGB加载器保持不可改，支持层包装q.GuardedData：取得当前Torch CPU RNG快照，调用原RGB读取/增强，再从同一快照重放相同几何增强到固定单位圆mask。必须核验两次增强消耗完全相同的随机数，并将RNG恢复到原RGB读取后的状态。标签、权重、全局索引和图像顺序原样传递。

这个包装只增加第四个几何支持通道，原RGB三通道不得变化。不能以像素明暗生成mask、不能重筛A–C、不改变等级或患者split。训练与验证允许索引继续由原GuardedData严格约束。随机性检查不能只验证输出形状，须逐张比较原RGB tensor和读取后的RNG状态。单位圆模板产生不消耗随机数。

CPU实际检查全部2077 AC train和427 AC val，两个食谱A(256/224)、B(448/384)，共5008次完整tensor/标签/权重/索引/RNG配对比较；另用4张AC train×20seed×2食谱的RGBA整体增强作为独立空间对齐参照，验证第四通道mask逐位相同。不得访问test或神经分类器，不能把通过当结构性能收益。

今后正式阶段使用相同种子的DataLoader generator和worker数；新增模型分支没有额外随机算子，Swin内部dropout必须为0，分类器原dropout和stochastic depth仍按被冻结食谱。完整原baseline的427张概率、真batch32 CUDA AMP和多worker配对仍待最终baseline冻结后验证。若任何当前检查失败，保留失败源码/配置/日志，不放宽标准，不覆盖原缓存。
