# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|mom_salience_single|94.7329%|94.6007%|94.6014%|94.6450%|94.6759%|92.0455%|92.2223%|
|mom_salience_c4|96.5097%|95.4752%|94.9293%|95.6380%|95.6597%|93.7500%|92.8535%|
|salience_only_single|95.6217%|94.2528%|94.7540%|94.8762%|94.9074%|93.9394%|92.1153%|
|salience_only_c4|95.9779%|95.8168%|95.6434%|95.8127%|95.8333%|95.4545%|91.6483%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_salience_single_vs_qualification_convnext_tiny_A|-0.3365|[-1.2717, 0.7788]|
|mom_salience_single_vs_B0_C4|-1.2814|[-2.1952, -0.2785]|
|mom_salience_single_vs_dino_baseline|-0.5323|[-2.7725, 1.4150]|
|mom_salience_single_vs_DINO_C4|-1.5101|[-3.9476, 0.0048]|
|mom_salience_single_vs_Swin_C4|-1.9320|[-3.9045, -0.4483]|
|mom_salience_c4_vs_qualification_convnext_tiny_A|0.6566|[-0.8947, 2.0449]|
|mom_salience_c4_vs_B0_C4|-0.2884|[-1.6834, 0.7843]|
|mom_salience_c4_vs_dino_baseline|0.4607|[-2.3208, 2.4844]|
|mom_salience_c4_vs_DINO_C4|-0.5170|[-3.5982, 1.2118]|
|mom_salience_c4_vs_Swin_C4|-0.9390|[-3.3754, 0.5517]|
|salience_only_single_vs_qualification_convnext_tiny_A|-0.1053|[-1.2990, 1.0063]|
|salience_only_single_vs_B0_C4|-1.0502|[-2.2559, 0.0611]|
|salience_only_single_vs_dino_baseline|-0.3011|[-2.6634, 1.4960]|
|salience_only_single_vs_DINO_C4|-1.2789|[-3.8330, 0.1682]|
|salience_only_single_vs_Swin_C4|-1.7008|[-3.8595, -0.2334]|
|salience_only_c4_vs_qualification_convnext_tiny_A|0.8312|[-0.9170, 2.0302]|
|salience_only_c4_vs_B0_C4|-0.1137|[-1.8422, 0.8932]|
|salience_only_c4_vs_dino_baseline|0.6354|[-2.4101, 2.6658]|
|salience_only_c4_vs_DINO_C4|-0.3424|[-3.7043, 1.4461]|
|salience_only_c4_vs_Swin_C4|-0.7643|[-3.5447, 0.8402]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原几何与Swin局部研究另有完整账本，不混为本轮结果。
