# CLE局部几何适配正式轮：冻结假设与比较

前轮跨尺度12模型及全部直接控制已完成并独立审计；原M方向矩仍是工作结构，固定B0不变。新轮从公开ImageNet同seed初始化重训，不复训历史有效模型或热启动已训练权重。纹理诊断只访问原train/val，发现高std组错误更多但不足以因果归因；训练图可见弯曲边界和大小不一暗区。提出可检验假设：保留原stem方向离散度线索，在stage3末已预训练局部7×7纹理卷积上调整采样形状，或许比额外全局跨层统计更适合这些局部几何变化。

仅替换stage3最后块depthwise的包装形式，保留原Conv对象和公开参数ID。原型affine_geometry.py逐字复用：3×3预测六个位置参数，zero W/bias；平移0.75tanh、theta±pi/8、log尺度±log1.5、剪切±.25，A=R*上三角正对角矩阵，生成49位置的交错(y,x)offset。det(A)>0只保证每个局部采样基的方向，不保证全局位移场无折叠/整网等变/临床拓扑。输出原Conv+官方DCNv1(x,offset,W)−DCNv1(x,zero,W)，两次D均保留梯度，FP32残差，原AMP分支不变。双D开销、插值与访存不能藏进“零参数模块”。

四arm×seed45/46/47=12新模型：mom_affine完整；affine_only去方向矩；mom_permuted把同六参数产生的49个offset固定打乱，完全同参数/边际offset范数，检验局部仿射对应；mom_dcn保留98自由offset的成熟DCNv1参照，参数更多显式列预算。三类几何控制均在同base+双D残差框架，mature DCNv1是忠实算子参照及公开权重适配，不称完整原论文backbone实验。原B0/M方向矩/成熟C4强参照复用有效三seed。所有完整三seed单视图及C4都报告，直接累计/去一/打乱/成熟参照比较在结果前锁定。

固定2077train/182患者、427val/39、576test/41，患者原split互斥，A–C标签/等级/阈值/清单不变，3595物理cache不等于训练3595，每epoch核验实际2077唯一索引及SHA。60epoch、batch32、原encoder1e-5、新预测器/方向矩/头1e-4，AdamW.05/cosine0/clip1/AMP，原相同H/V/rotation15，workers2/thread1。最早最高原0度val macroF1选权重，test不选本轮epoch。

CPU原型已通过NumPy直接双线性与独立grid_sample输出/三梯度、double gradcheck、局部det区间、public Tiny224+方向矩初始logit严格一致、原参数ID/RNG与有限反向；仅合成张量、CLE图0，不能称CLE有效。正式源/config/tasks/必要预检查先Git提交，再做batch32两步CUDA有限/所有预测器梯度、完整B0和M方向矩六个427概率复现，才测并发并开训。独立实测1/2/4起步，若GPU不足80%、吞吐确实改善且CPU仍有余量则按显存继续扩展，不继承6/8硬上限。保留6GiB余量，选择实测最佳97%的最小并发。

每epoch及独立连续采样记录GPU/显存/CPU总与单核/I/O wait/RAM/磁盘/DataLoader等待/sps/耗时。仅检查已核验自身worker/子进程，不干预别的Agent。失败先看日志/活进程，完整权重后处理失败只补推理。所有seed、阴性、源码/配置/权重/预测SHA、有限值/val最佳/患者隔离/预算独立审计并Git保存。

用户明确train→val选权重→test→test反馈下一结构。故test为自适应内部开发；历史test已查看且A–C构造用过标签。20k患者配对bootstrap仅条件区间，不校正选择偏差，不称独立泛化或外部验证。固定平均macroF1提升严格>0.015、三配对seed稳定正且区间下界正才进入达标复核，Fraction整数混淆矩阵严格3/200再独立核算。accuracy/恶性召回/患者等权accuracy/全部混淆矩阵/计算代价同时报告。整体研究不因本轮阴性结束，不保证提升或SCI新颖/录用。
