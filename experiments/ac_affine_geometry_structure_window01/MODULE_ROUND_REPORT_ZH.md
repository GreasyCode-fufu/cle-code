# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|mom_affine_single|95.8158%|94.0244%|95.6439%|95.1613%|95.1968%|92.2348%|93.2117%|
|mom_affine_c4|95.7797%|94.9557%|96.0039%|95.5798%|95.6019%|92.6136%|93.5490%|
|affine_only_single|95.4265%|93.8569%|95.1232%|94.8022%|94.8495%|92.0455%|91.1713%|
|affine_only_c4|96.6760%|94.9414%|95.8225%|95.8133%|95.8333%|93.3712%|92.7939%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_affine_single_vs_qualification_convnext_tiny_A|0.1799|[-0.6264, 1.3495]|
|mom_affine_single_vs_B0_C4|-0.7651|[-1.6047, 0.3787]|
|mom_affine_single_vs_dino_baseline|-0.0160|[-1.9266, 2.0569]|
|mom_affine_single_vs_DINO_C4|-0.9937|[-3.0809, 0.5778]|
|mom_affine_single_vs_Swin_C4|-1.4157|[-3.2241, 0.1728]|
|mom_affine_c4_vs_qualification_convnext_tiny_A|0.5983|[-0.1902, 1.8403]|
|mom_affine_c4_vs_B0_C4|-0.3466|[-0.9017, 0.4661]|
|mom_affine_c4_vs_dino_baseline|0.4024|[-1.6638, 2.5111]|
|mom_affine_c4_vs_DINO_C4|-0.5753|[-2.8903, 1.0909]|
|mom_affine_c4_vs_Swin_C4|-0.9972|[-2.7381, 0.4932]|
|affine_only_single_vs_qualification_convnext_tiny_A|-0.1792|[-0.7482, 0.2561]|
|affine_only_single_vs_B0_C4|-1.1242|[-2.0551, -0.3191]|
|affine_only_single_vs_dino_baseline|-0.3751|[-2.9093, 1.7299]|
|affine_only_single_vs_DINO_C4|-1.3529|[-4.0534, 0.3047]|
|affine_only_single_vs_Swin_C4|-1.7748|[-3.8949, -0.4848]|
|affine_only_c4_vs_qualification_convnext_tiny_A|0.8319|[0.1293, 1.5897]|
|affine_only_c4_vs_B0_C4|-0.1131|[-0.3152, 0.0000]|
|affine_only_c4_vs_dino_baseline|0.6360|[-1.6475, 2.6235]|
|affine_only_c4_vs_DINO_C4|-0.3418|[-2.8599, 1.2736]|
|affine_only_c4_vs_Swin_C4|-0.7637|[-2.5615, 0.3645]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原18项几何结构研究另有完整账本，不混为本轮结果。
