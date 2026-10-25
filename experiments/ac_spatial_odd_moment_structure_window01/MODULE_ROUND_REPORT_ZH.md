# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|mom_odd_single|95.7674%|94.2042%|95.6497%|95.2071%|95.2546%|91.4773%|92.9244%|
|mom_odd_c4|96.3310%|95.2704%|96.1365%|95.9126%|95.9491%|92.4242%|93.7586%|
|odd_only_single|95.8245%|94.3558%|94.0087%|94.7296%|94.7917%|90.1515%|92.5536%|
|odd_only_c4|95.9990%|94.7535%|95.0955%|95.2827%|95.3125%|91.6667%|92.2278%|
|mom_raw3_single|95.9562%|95.2768%|95.2948%|95.5092%|95.5440%|93.5606%|92.5814%|
|mom_raw3_c4|96.3013%|95.7896%|95.8008%|95.9639%|96.0069%|93.7500%|93.4553%|
|mom_std_single|96.1716%|94.0187%|95.3072%|95.1658%|95.1968%|92.2348%|93.5154%|
|mom_std_c4|95.9794%|95.4490%|95.9961%|95.8082%|95.8333%|92.6136%|93.8126%|
|mom_even_single|95.2526%|94.1837%|95.4745%|94.9703%|95.0231%|91.8561%|91.9611%|
|mom_even_c4|96.3250%|94.9284%|95.9961%|95.7499%|95.7755%|93.1818%|92.7311%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_odd_single_vs_qualification_convnext_tiny_A|0.2257|[-0.6643, 1.4967]|
|mom_odd_single_vs_B0_C4|-0.7193|[-1.5489, 0.3063]|
|mom_odd_single_vs_dino_baseline|0.0298|[-2.0201, 2.1623]|
|mom_odd_single_vs_DINO_C4|-0.9480|[-3.2505, 0.7390]|
|mom_odd_single_vs_Swin_C4|-1.3699|[-3.3753, 0.3940]|
|mom_odd_c4_vs_qualification_convnext_tiny_A|0.9312|[0.0388, 2.1605]|
|mom_odd_c4_vs_B0_C4|-0.0138|[-0.5618, 0.7340]|
|mom_odd_c4_vs_dino_baseline|0.7353|[-1.4788, 2.8142]|
|mom_odd_c4_vs_DINO_C4|-0.2425|[-2.7615, 1.4316]|
|mom_odd_c4_vs_Swin_C4|-0.6644|[-2.7101, 0.9796]|
|odd_only_single_vs_qualification_convnext_tiny_A|-0.2518|[-1.1066, 0.7320]|
|odd_only_single_vs_B0_C4|-1.1968|[-2.0293, -0.3391]|
|odd_only_single_vs_dino_baseline|-0.4477|[-2.8624, 1.6128]|
|odd_only_single_vs_DINO_C4|-1.4254|[-4.0940, 0.2876]|
|odd_only_single_vs_Swin_C4|-1.8474|[-3.9709, -0.3012]|
|odd_only_c4_vs_qualification_convnext_tiny_A|0.3012|[-0.8355, 1.3835]|
|odd_only_c4_vs_B0_C4|-0.6437|[-1.4276, 0.0412]|
|odd_only_c4_vs_dino_baseline|0.1053|[-2.3522, 2.1825]|
|odd_only_c4_vs_DINO_C4|-0.8724|[-3.6349, 0.9199]|
|odd_only_c4_vs_Swin_C4|-1.2943|[-3.3940, 0.2399]|
|mom_raw3_single_vs_qualification_convnext_tiny_A|0.5278|[-0.5142, 1.5969]|
|mom_raw3_single_vs_B0_C4|-0.4172|[-1.4891, 0.5576]|
|mom_raw3_single_vs_dino_baseline|0.3319|[-2.0563, 2.4411]|
|mom_raw3_single_vs_DINO_C4|-0.6458|[-3.2436, 1.0036]|
|mom_raw3_single_vs_Swin_C4|-1.0678|[-2.8741, 0.2135]|
|mom_raw3_c4_vs_qualification_convnext_tiny_A|0.9825|[-0.0766, 2.2844]|
|mom_raw3_c4_vs_B0_C4|0.0375|[-0.9591, 1.1941]|
|mom_raw3_c4_vs_dino_baseline|0.7866|[-1.5925, 2.8763]|
|mom_raw3_c4_vs_DINO_C4|-0.1912|[-2.8355, 1.4445]|
|mom_raw3_c4_vs_Swin_C4|-0.6131|[-2.7233, 1.0275]|
|mom_std_single_vs_qualification_convnext_tiny_A|0.1844|[-0.6473, 1.4802]|
|mom_std_single_vs_B0_C4|-0.7606|[-1.6493, 0.5724]|
|mom_std_single_vs_dino_baseline|-0.0115|[-1.7903, 1.9817]|
|mom_std_single_vs_DINO_C4|-0.9892|[-2.8855, 0.4463]|
|mom_std_single_vs_Swin_C4|-1.4112|[-3.0182, 0.0855]|
|mom_std_c4_vs_qualification_convnext_tiny_A|0.8267|[0.0821, 2.0972]|
|mom_std_c4_vs_B0_C4|-0.1183|[-0.7093, 0.7981]|
|mom_std_c4_vs_dino_baseline|0.6308|[-1.2978, 2.7668]|
|mom_std_c4_vs_DINO_C4|-0.3469|[-2.4787, 1.2773]|
|mom_std_c4_vs_Swin_C4|-0.7688|[-2.3338, 0.6604]|
|mom_even_single_vs_qualification_convnext_tiny_A|-0.0112|[-0.6127, 0.6282]|
|mom_even_single_vs_B0_C4|-0.9561|[-1.7404, -0.1892]|
|mom_even_single_vs_dino_baseline|-0.2070|[-2.4315, 1.7594]|
|mom_even_single_vs_DINO_C4|-1.1848|[-3.5819, 0.3666]|
|mom_even_single_vs_Swin_C4|-1.6067|[-3.5287, -0.2359]|
|mom_even_c4_vs_qualification_convnext_tiny_A|0.7684|[0.0551, 1.5280]|
|mom_even_c4_vs_B0_C4|-0.1765|[-0.5254, 0.0796]|
|mom_even_c4_vs_dino_baseline|0.5725|[-1.7822, 2.5582]|
|mom_even_c4_vs_DINO_C4|-0.4052|[-2.9859, 1.2390]|
|mom_even_c4_vs_Swin_C4|-0.8271|[-2.6266, 0.2539]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原几何与Swin局部研究另有完整账本，不混为本轮结果。
