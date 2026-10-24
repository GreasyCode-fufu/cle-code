# Swin B原方向矩迁移与首merge空间对比：正式预注册

POWER全部15×60epoch与直接控制已独立通过且10数值门槛均false。原方向矩保留；q、rawH2、重复σ及患者辅助loss均无已确认增量，不带入。CPU原型passed只证明实现，不能证明CLE效果。固定B0仍Q按预提交val规则的ConvNeXt-Tiny_A224，不能换成较弱Swin A；工作Swin B根据历史test选择，必须披露。

采用原Q Swin_t_B：原图SHA核过RGB resize448、crop384、公开ImageNet、stochastic depth .1。原Swin B三seed已有效60epoch，按任务/源码/checkpointSHA复用不重训。新五arm×seed45/46/47：mom_only仅原方向矩（主干迁移去local参考）；mom_local原Mom+首merge local_std；local_only去Mom；mom_mean同参数均值控制；mom_global相同local_std先空间均值广播、移除空间定位。四种extra分支参数同18624，Mom另18528；shape信息并非自动创新。成熟Swin合并层为主要同主干参照，均值/全局控制检验标准统计/残差参数与空间定位；Haar仅方差数值oracle，非HWD整法复现。

merge保持原concat→learnedLN→Linear参数对象。统计用同4C拼接张量的无仿射LN eps1e-5；h=sqrt(mean_q((u_q−mean_q u)^2)+1e-6)−.001，零C→2C投影残差。选择无仿射避免各象限gamma/beta破坏统计置换性质，原learnedLN继续原样主分支。平坦h=0，不声称整网等变、临床结构正确或eps下严格强度不变。仅first merge dim96，所有新分支零初始、RNG/原参数不变，不能热启动已训练权重。

原固定AC train2077/182、val427/39、test576/41，标签/等级/清单/split/采样不动，恶性1。每epoch2077唯一索引SHA。训练原Q60epoch,batch32,encoder1e-5/new+head1e-4,AdamW.05,cosine0,AMP/clip1,原HV/rotation15/randomcrop384,workers2,thread1。B train/val2504×448原图缓存及T test576×448缓存分别只读验证，不从256上采样，不将物理cache数量称训练数量。

原0度val最早最高macroF1选checkpoint，三seed全部训练完才注册/test；test反馈用于下一方法，无独立泛化声明。每模型完整427val与576test的single/C4、两个独立像素加载路线、权重有限/source/job/checkpointSHA、整数混淆矩阵/Fraction、41患者20k联合bootstrap。对mom_local预注册mom_only/local_only/mom_mean/mom_global/SwinB/固定B0两mode直接比较。主门槛平均Δ严格>3/200、三seed全正及条件CI下界正，所有原始阴性保留；多轮适应性选择偏差未被bootstrap校正。

预算plain27521661/28.597871616G；Mom27540189/29.192414208G；local_only27540285/28.682806272G；Mom+merge27558813/29.277348864G。相较固定B0 8.909531136G，主干成本显著更高，必须同时报告同Swin父模型/同C4预算及实测吞吐，逻辑MAC不含统计/LN/点运算/访存。正式每arm测1/2/4并发，97%最小有益选择和6GiB余量；最多仅3正式seed，cap4不能称4正式进程。GPU满载不为CPU数字增无益任务。

中文结构、使用与实际训练回执后补独立文件；冻结正式source/config不能漂移。后处理失败补CPU/推理不重训有效模型。不保证提点、SCI新颖或录用；整体研究active。
