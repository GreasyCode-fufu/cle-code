# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|mom_cross_single|95.9641%|94.0244%|96.1700%|95.3861%|95.4282%|91.6667%|92.9179%|
|mom_cross_c4|96.4504%|95.1196%|96.0036%|95.8579%|95.8912%|92.8030%|93.9586%|
|cross_only_single|96.1067%|94.2184%|95.2879%|95.2043%|95.2546%|91.8561%|93.0975%|
|cross_only_c4|96.6517%|94.7661%|96.0003%|95.8061%|95.8333%|93.1818%|93.2823%|
|mom_supcon_single|95.8005%|94.0244%|96.1700%|95.3316%|95.3704%|92.2348%|92.7700%|
|mom_supcon_c4|96.1165%|95.1196%|95.8338%|95.6900%|95.7176%|93.1818%|93.0794%|
|mom_within_single|95.4569%|94.1956%|96.1587%|95.2704%|95.3125%|92.0455%|92.7655%|
|mom_within_c4|96.2714%|94.9320%|96.3363%|95.8466%|95.8912%|92.0455%|94.5859%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_cross_single_vs_qualification_convnext_tiny_A|0.4047|[-0.2712, 1.5431]|
|mom_cross_single_vs_B0_C4|-0.5403|[-1.3317, 0.6246]|
|mom_cross_single_vs_dino_baseline|0.2088|[-1.5430, 2.2310]|
|mom_cross_single_vs_DINO_C4|-0.7689|[-2.6588, 0.6941]|
|mom_cross_single_vs_Swin_C4|-1.1909|[-2.6891, 0.2239]|
|mom_cross_c4_vs_qualification_convnext_tiny_A|0.8765|[0.0997, 2.1167]|
|mom_cross_c4_vs_B0_C4|-0.0685|[-0.5756, 0.7368]|
|mom_cross_c4_vs_dino_baseline|0.6806|[-1.3076, 2.7146]|
|mom_cross_c4_vs_DINO_C4|-0.2972|[-2.5581, 1.3011]|
|mom_cross_c4_vs_Swin_C4|-0.7191|[-2.4043, 0.7635]|
|cross_only_single_vs_qualification_convnext_tiny_A|0.2229|[-0.2289, 1.2071]|
|cross_only_single_vs_B0_C4|-0.7221|[-1.5199, 0.5261]|
|cross_only_single_vs_dino_baseline|0.0270|[-1.9827, 2.2870]|
|cross_only_single_vs_DINO_C4|-0.9508|[-3.0687, 0.7309]|
|cross_only_single_vs_Swin_C4|-1.3727|[-2.9740, 0.1191]|
|cross_only_c4_vs_qualification_convnext_tiny_A|0.8246|[0.2656, 1.8175]|
|cross_only_c4_vs_B0_C4|-0.1204|[-0.6063, 0.6629]|
|cross_only_c4_vs_dino_baseline|0.6287|[-1.4303, 2.7747]|
|cross_only_c4_vs_DINO_C4|-0.3490|[-2.6156, 1.3020]|
|cross_only_c4_vs_Swin_C4|-0.7710|[-2.4082, 0.6365]|
|mom_supcon_single_vs_qualification_convnext_tiny_A|0.3502|[-0.2768, 1.4543]|
|mom_supcon_single_vs_B0_C4|-0.5948|[-1.3660, 0.5955]|
|mom_supcon_single_vs_dino_baseline|0.1543|[-1.6875, 2.2504]|
|mom_supcon_single_vs_DINO_C4|-0.8235|[-2.8091, 0.7570]|
|mom_supcon_single_vs_Swin_C4|-1.2454|[-2.7949, 0.2962]|
|mom_supcon_c4_vs_qualification_convnext_tiny_A|0.7085|[-0.0505, 1.8128]|
|mom_supcon_c4_vs_B0_C4|-0.2364|[-0.7850, 0.4926]|
|mom_supcon_c4_vs_dino_baseline|0.5127|[-1.5523, 2.4914]|
|mom_supcon_c4_vs_DINO_C4|-0.4651|[-2.7795, 1.1061]|
|mom_supcon_c4_vs_Swin_C4|-0.8870|[-2.6722, 0.6058]|
|mom_within_single_vs_qualification_convnext_tiny_A|0.2889|[-0.4197, 1.3111]|
|mom_within_single_vs_B0_C4|-0.6560|[-1.4682, 0.4052]|
|mom_within_single_vs_dino_baseline|0.0931|[-2.0732, 2.2740]|
|mom_within_single_vs_DINO_C4|-0.8847|[-3.2350, 0.8070]|
|mom_within_single_vs_Swin_C4|-1.3066|[-3.1516, 0.2196]|
|mom_within_c4_vs_qualification_convnext_tiny_A|0.8651|[0.1840, 2.1661]|
|mom_within_c4_vs_B0_C4|-0.0798|[-0.5544, 0.7805]|
|mom_within_c4_vs_dino_baseline|0.6693|[-1.2637, 2.8520]|
|mom_within_c4_vs_DINO_C4|-0.3085|[-2.4715, 1.3855]|
|mom_within_c4_vs_Swin_C4|-0.7304|[-2.2817, 0.6937]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原18项几何结构研究另有完整账本，不混为本轮结果。
