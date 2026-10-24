# 保留方向矩的患者条件表征研究：正式冻结

原方向矩在固定A–C上提供尚未达+1.5百分点的工作信号。局部几何完整12模型、所有对照与独立审计已完成，完整几何低于方向矩；保留方向矩，改从训练表征而非继续近似空间模块枚举。具体假设：同患者多帧正对可能强化病例特征；同类不同患者对齐可能减少此依赖，也可能抹除有用个体差异，须正反机制对照。

四arm×seed45/46/47全部从相同公开ImageNet初始化，60epoch、batch32、原HV/15度增强、AdamW.05、encoder1e-5、头及新模块1e-4、cosine到0、AMP、clip1、worker2/thread1。固定B0不可更换或削弱，所有有效参考按源码/任务/权重SHA复用。每epoch仍2077唯一train索引一次，不改标签/等级/阈值/患者split，不做类别或患者再采样。3595物理缓存与2077训练使用明确区分。

mom_cross=原M方向矩+训练辅助同类不同患者对齐；cross_only去方向矩；mom_supcon=同方向矩/投影的成熟标准SupCon；mom_within=同患者同类正对的反机制参照。辅助MLP768→128→64（GELU、无偏、L2归一），106496参数，推理仅原分类logit，不需要患者ID。单视图CE+0.05L，T=baseT=.1，超参本轮固定，完整三seed不挑最高。cross/within分母为相应正对+全部异类对，被排除的同类对不变成伪负；多标签同患者的异类仍是负对。空正对anchor贡献0，全anchor均值；原随机batch下有效anchor比例及正对数逐epoch记录，不隐藏within稀疏性。标准SupCon与官方单视图loss/梯度独立对齐，不称PaCL完整复现。

只在独立新源码引擎更换训练loss回调和增加objective日志，其他训练函数保留原AST。CPU先核独立NumPy、官方loss/梯度、doublegradcheck、退化边界、公共初始化/RNG/原参数；CUDA核真实batch32两步有限及projection梯度、6个完整427val概率复建、test训练加载拒绝和物理cacheSHA。单/并发实测必须包含完整辅助目标，按吞吐/显存/CPU选择，保留6GiB余量，不为填GPU造无价值任务。

原0度val最早最高macroF1选权重。每方法三seed完成后两独立概率路径统一推理val/test的single/C4；所有seed的macroF1、accuracy、恶性召回、患者等权accuracy、混淆矩阵、成本与SHA保留。直接比较mom_cross对原M、去一、成熟SupCon、患者内反机制、固定B0，两推理预算分列。独立整数Fraction检查平均差严格>3/200、所有seed正、41患者20000次配对bootstrap下界正。

用户明确test反馈下一设计，且历史test被查看、AC构造使用标签；所有test为自适应内部开发，条件bootstrap不校正选择偏差，不能称独立泛化/新外部验证。等级只是模型一致性代理。患者条件对比学习、SupCon和投影均有成熟先验；不预先声明网络新颖、可靠提升或SCI录用。一般阴性保留后依据对照继续，整体目标不因此轮结束。
