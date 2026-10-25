# 用户指定evaluate.py全指标结果后补审

用户明确最终性能不限宏F1，且指标以/home/login/data/cle/auto/GPT/evaluate.py为准。原字节SHA e5213828f81e04baf2aa025c35dfa3a1b585efbdcf8a7cfbd9033510592b0ee7。这里不执行该脚本的load_model/目录扫描/图像变换/推理入口，仅受限AST原compute_metrics函数，对已完成、逐图概率已独立验收的同576测试样本原完整精度CSV计算全指标。已有研究用独立冻结评估，不能冒称此前直接跑了evaluate.py全程序。现有新模型不在其注册表，且其strict=False/过滤key的加载方式不宜盲用本研究自定义权重，未改原脚本、原训练或推理结果。

覆盖Macro-F1、Accuracy、OvR Macro-AUROC、Macro-PR/AP、weighted F1、balanced accuracy、宏precision/recall/sensitivity/specificity、weighted precision/recall、Kappa、MCC；各类precision/recall/sensitivity/specificity/F1/NPV/AUROC/AUC-PR和混淆矩阵均保留。患者等权accuracy不是evaluate.py指标，只为原附加诊断，不能替代用户指定的达标指标。参数/FLOPs/FPS是单独成本，不能用速度增加1.5FPS等伪充分类性能1.5个百分点；Kappa/MCC作为系数变化完整报告，不把系数+0.015自动当作1.5个百分点分类性能。

指标扩展发生在用户查看历史结果之后，属于明确用户授权的结果后补审，不伪称事前注册、不重新选择test epoch。原固定B0及其三seed、相同标签/等级/原患者split、val最早最佳权重规则和训练预算均不变，不重训或恢复任何有效终态。全部既有完整CNN结构研究都纳入，保留所有指标和阴性，不能只挑单个最好数值隐去取舍。C4单列对B0 single累计与B0_C4同预算，原Mom同视图作为父模型，候选/去一/成熟控制同时比较。成熟组合不自动成为SCI创新。

每条原CSV核完整576图/41患者索引、sample顺序/标签/患者组、概率范围/和、绑定group/registration/config/source与实际checkpoint SHA；原权重从服务器字节再次独立重算，不下载本地权重。每个点指标调用原evaluate.compute_metrics，并独立整数Fraction重算混淆矩阵派生指标；AUROC用正负成对顺序含tie1/2、AP按相同score阈值分组的精确precision增量做独立Fraction oracle，对原函数误差<1e-12。

不确定性使用原41患者配对20k bootstrap、同一患者抽样跨三seed共用，并保持原20261024抽样种子。CM派生指标全比较均计算条件CI；AUROC/AP的条件CI只在至少+1.5pp且三seed同向的潜在达标比较中补算。rank指标缺某类的抽样不参与区间，报告valid_draws，最多缺1%。这是患者聚类条件描述区间，与evaluate.py可选的逐图1000次bootstrap不同，不混称同一个区间；不校正自适应模型/指标选择偏差，不称独立或外部泛化。

每个相关比例指标的相对固定B0均值至少+1.5pp、配对三seed均正、条件CI下界正，仅记录为多指标条件数值资格；仍需审视2–3互补网络设计的真实累计/去一/成熟控制与成本/其它指标取舍，不能仅资格表行数或重复指标就宣布整体完成。weighted recall等于accuracy、balanced accuracy等于macro recall/sensitivity为同源指标，不冒称多个独立改进。
