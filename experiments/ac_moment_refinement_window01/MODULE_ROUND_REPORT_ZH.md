# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|mean_only_single|95.4265%|94.3697%|95.3054%|95.0339%|95.0810%|92.0455%|92.2975%|
|mean_only_c4|96.4844%|94.9284%|96.1842%|95.8657%|95.8912%|93.5606%|92.7582%|
|random_moments_single|95.6488%|95.6389%|95.4710%|95.5863%|95.6019%|94.3182%|92.4603%|
|random_moments_c4|95.9906%|95.8094%|96.0057%|95.9352%|95.9491%|93.9394%|94.1314%|
|moments_eca_single|96.3332%|95.8133%|95.8180%|95.9882%|96.0069%|93.5606%|94.0453%|
|moments_eca_c4|95.6357%|96.1438%|95.9964%|95.9253%|95.9491%|92.6136%|94.9729%|
|aug_moments_single|96.1471%|94.9288%|95.4835%|95.5198%|95.5440%|93.9394%|93.7030%|
|aug_moments_c4|96.5084%|95.6393%|95.8372%|95.9950%|96.0069%|94.8864%|93.4440%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mean_only_single_vs_qualification_convnext_tiny_A|0.0524|[-0.8278, 1.0014]|
|mean_only_single_vs_B0_C4|-0.8926|[-1.7884, 0.0755]|
|mean_only_single_vs_dino_baseline|-0.1435|[-2.3400, 1.8464]|
|mean_only_single_vs_DINO_C4|-1.1212|[-3.5218, 0.4558]|
|mean_only_single_vs_Swin_C4|-1.5431|[-3.6011, 0.0736]|
|mean_only_c4_vs_qualification_convnext_tiny_A|0.8842|[0.1935, 1.7001]|
|mean_only_c4_vs_B0_C4|-0.0607|[-0.3580, 0.1774]|
|mean_only_c4_vs_dino_baseline|0.6884|[-1.5109, 2.5673]|
|mean_only_c4_vs_DINO_C4|-0.2894|[-2.7368, 1.2482]|
|mean_only_c4_vs_Swin_C4|-0.7113|[-2.4646, 0.3838]|
|random_moments_single_vs_qualification_convnext_tiny_A|0.6048|[-0.3288, 1.8657]|
|random_moments_single_vs_B0_C4|-0.3402|[-1.3631, 0.8657]|
|random_moments_single_vs_dino_baseline|0.4089|[-1.6875, 2.2501]|
|random_moments_single_vs_DINO_C4|-0.5688|[-2.8371, 0.8209]|
|random_moments_single_vs_Swin_C4|-0.9908|[-2.5897, 0.2893]|
|random_moments_c4_vs_qualification_convnext_tiny_A|0.9538|[0.0117, 2.6409]|
|random_moments_c4_vs_B0_C4|0.0088|[-0.9618, 1.5665]|
|random_moments_c4_vs_dino_baseline|0.7579|[-1.2416, 3.0143]|
|random_moments_c4_vs_DINO_C4|-0.2198|[-2.3386, 1.4770]|
|random_moments_c4_vs_Swin_C4|-0.6418|[-2.1104, 0.8938]|
|moments_eca_single_vs_qualification_convnext_tiny_A|1.0067|[0.3293, 2.4444]|
|moments_eca_single_vs_B0_C4|0.0618|[-0.6674, 1.4234]|
|moments_eca_single_vs_dino_baseline|0.8109|[-0.8568, 3.1331]|
|moments_eca_single_vs_DINO_C4|-0.1669|[-1.9247, 1.5485]|
|moments_eca_single_vs_Swin_C4|-0.5888|[-1.9044, 0.9240]|
|moments_eca_c4_vs_qualification_convnext_tiny_A|0.9439|[-0.0481, 2.7186]|
|moments_eca_c4_vs_B0_C4|-0.0011|[-0.9064, 1.4646]|
|moments_eca_c4_vs_dino_baseline|0.7480|[-1.0961, 3.0568]|
|moments_eca_c4_vs_DINO_C4|-0.2298|[-2.2520, 1.5409]|
|moments_eca_c4_vs_Swin_C4|-0.6517|[-2.2969, 1.1283]|
|aug_moments_single_vs_qualification_convnext_tiny_A|0.5383|[-0.7410, 2.2703]|
|aug_moments_single_vs_B0_C4|-0.4066|[-1.7109, 1.2052]|
|aug_moments_single_vs_dino_baseline|0.3425|[-1.8687, 2.3999]|
|aug_moments_single_vs_DINO_C4|-0.6353|[-3.0171, 0.9711]|
|aug_moments_single_vs_Swin_C4|-1.0572|[-2.7483, 0.4384]|
|aug_moments_c4_vs_qualification_convnext_tiny_A|1.0135|[-0.2226, 2.4389]|
|aug_moments_c4_vs_B0_C4|0.0685|[-1.1883, 1.3201]|
|aug_moments_c4_vs_dino_baseline|0.8176|[-1.6058, 2.7676]|
|aug_moments_c4_vs_DINO_C4|-0.1601|[-2.8029, 1.4365]|
|aug_moments_c4_vs_Swin_C4|-0.5820|[-2.4731, 0.8236]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原18项几何结构研究另有完整账本，不混为本轮结果。
