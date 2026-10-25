# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|mom_focalcontrast_single|94.6678%|95.6208%|94.1821%|94.8236%|94.9074%|89.7727%|92.3226%|
|mom_focalcontrast_c4|96.1134%|96.5241%|94.8969%|95.8448%|95.8912%|92.2348%|92.4902%|
|focalcontrast_only_single|96.1532%|95.2349%|94.3579%|95.2487%|95.3125%|90.5303%|93.1496%|
|focalcontrast_only_c4|96.0029%|95.4377%|94.7335%|95.3914%|95.4282%|91.4773%|92.5107%|
|mom_focalraw_single|94.8805%|95.4739%|95.8049%|95.3864%|95.4282%|93.3712%|91.8490%|
|mom_focalraw_c4|95.6057%|95.6453%|96.3295%|95.8602%|95.8912%|93.5606%|93.2393%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_focalcontrast_single_vs_qualification_convnext_tiny_A|-0.1579|[-1.3652, 1.0496]|
|mom_focalcontrast_single_vs_B0_C4|-1.1029|[-2.0951, -0.1394]|
|mom_focalcontrast_single_vs_dino_baseline|-0.3538|[-2.9285, 1.7456]|
|mom_focalcontrast_single_vs_DINO_C4|-1.3315|[-4.2261, 0.3988]|
|mom_focalcontrast_single_vs_Swin_C4|-1.7535|[-4.2312, 0.0183]|
|mom_focalcontrast_c4_vs_qualification_convnext_tiny_A|0.8633|[-0.3756, 2.1487]|
|mom_focalcontrast_c4_vs_B0_C4|-0.0816|[-1.0744, 0.8840]|
|mom_focalcontrast_c4_vs_dino_baseline|0.6675|[-1.5402, 2.5325]|
|mom_focalcontrast_c4_vs_DINO_C4|-0.3103|[-2.8516, 1.2226]|
|mom_focalcontrast_c4_vs_Swin_C4|-0.7322|[-2.9536, 1.0089]|
|focalcontrast_only_single_vs_qualification_convnext_tiny_A|0.2672|[-0.8035, 1.5212]|
|focalcontrast_only_single_vs_B0_C4|-0.6778|[-1.6485, 0.4263]|
|focalcontrast_only_single_vs_dino_baseline|0.0713|[-2.4407, 2.2186]|
|focalcontrast_only_single_vs_DINO_C4|-0.9064|[-3.7030, 0.8266]|
|focalcontrast_only_single_vs_Swin_C4|-1.3284|[-3.6897, 0.4236]|
|focalcontrast_only_c4_vs_qualification_convnext_tiny_A|0.4099|[-0.6733, 1.5089]|
|focalcontrast_only_c4_vs_B0_C4|-0.5350|[-1.3553, 0.2300]|
|focalcontrast_only_c4_vs_dino_baseline|0.2141|[-2.2591, 2.2657]|
|focalcontrast_only_c4_vs_DINO_C4|-0.7637|[-3.5336, 0.9431]|
|focalcontrast_only_c4_vs_Swin_C4|-1.1856|[-3.5099, 0.4375]|
|mom_focalraw_single_vs_qualification_convnext_tiny_A|0.4050|[-0.3926, 1.3690]|
|mom_focalraw_single_vs_B0_C4|-0.5400|[-1.3267, 0.3458]|
|mom_focalraw_single_vs_dino_baseline|0.2091|[-2.0595, 2.3130]|
|mom_focalraw_single_vs_DINO_C4|-0.7687|[-3.2643, 0.9054]|
|mom_focalraw_single_vs_Swin_C4|-1.1906|[-3.1662, 0.3564]|
|mom_focalraw_c4_vs_qualification_convnext_tiny_A|0.8787|[0.0826, 2.0307]|
|mom_focalraw_c4_vs_B0_C4|-0.0662|[-0.7372, 0.8343]|
|mom_focalraw_c4_vs_dino_baseline|0.6828|[-1.2136, 2.5943]|
|mom_focalraw_c4_vs_DINO_C4|-0.2949|[-2.4067, 1.1633]|
|mom_focalraw_c4_vs_Swin_C4|-0.7168|[-2.4459, 0.7698]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原几何与Swin局部研究另有完整账本，不混为本轮结果。
