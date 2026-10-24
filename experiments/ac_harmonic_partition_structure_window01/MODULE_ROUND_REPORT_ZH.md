# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|mom_shape_single|95.6235%|94.7691%|95.3000%|95.2309%|95.2546%|94.5076%|92.9793%|
|mom_shape_c4|96.1517%|95.9814%|95.4879%|95.8737%|95.8912%|94.6970%|93.3376%|
|shape_only_single|95.1374%|95.2843%|96.6832%|95.7016%|95.7176%|93.9394%|93.8921%|
|shape_only_c4|96.6971%|96.1740%|96.0086%|96.2932%|96.2963%|95.2652%|94.0831%|
|mom_permuted_single|95.9625%|95.4554%|95.6534%|95.6904%|95.7176%|94.3182%|93.8038%|
|mom_permuted_c4|96.3128%|95.9827%|96.0020%|96.0991%|96.1227%|94.8864%|93.9369%|
|mom_duplicate_single|95.8076%|94.5631%|95.9901%|95.4536%|95.4861%|92.8030%|93.4073%|
|mom_duplicate_c4|96.6886%|95.6317%|96.3253%|96.2152%|96.2384%|93.9394%|93.9691%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_shape_single_vs_qualification_convnext_tiny_A|0.2494|[-1.4554, 1.8911]|
|mom_shape_single_vs_B0_C4|-0.6956|[-2.3869, 0.7602]|
|mom_shape_single_vs_dino_baseline|0.0535|[-2.2107, 1.7365]|
|mom_shape_single_vs_DINO_C4|-0.9242|[-3.3942, 0.4493]|
|mom_shape_single_vs_Swin_C4|-1.3462|[-3.3844, 0.1926]|
|mom_shape_c4_vs_qualification_convnext_tiny_A|0.8922|[-0.5869, 2.4243]|
|mom_shape_c4_vs_B0_C4|-0.0527|[-1.4494, 1.1566]|
|mom_shape_c4_vs_dino_baseline|0.6964|[-1.6656, 2.5251]|
|mom_shape_c4_vs_DINO_C4|-0.2814|[-2.8974, 1.2291]|
|mom_shape_c4_vs_Swin_C4|-0.7033|[-2.7090, 0.7364]|
|shape_only_single_vs_qualification_convnext_tiny_A|0.7202|[-0.6409, 2.2566]|
|shape_only_single_vs_B0_C4|-0.2248|[-1.4965, 1.1291]|
|shape_only_single_vs_dino_baseline|0.5243|[-1.4848, 2.4130]|
|shape_only_single_vs_DINO_C4|-0.4535|[-2.6642, 0.9834]|
|shape_only_single_vs_Swin_C4|-0.8754|[-2.8403, 0.7288]|
|shape_only_c4_vs_qualification_convnext_tiny_A|1.3118|[-0.0254, 2.8277]|
|shape_only_c4_vs_B0_C4|0.3668|[-0.7978, 1.5569]|
|shape_only_c4_vs_dino_baseline|1.1159|[-1.0524, 2.9564]|
|shape_only_c4_vs_DINO_C4|0.1381|[-2.3231, 1.6372]|
|shape_only_c4_vs_Swin_C4|-0.2838|[-2.3457, 1.3542]|
|mom_permuted_single_vs_qualification_convnext_tiny_A|0.7090|[-1.0957, 2.4235]|
|mom_permuted_single_vs_B0_C4|-0.2360|[-1.8895, 1.2555]|
|mom_permuted_single_vs_dino_baseline|0.5131|[-1.7476, 2.2379]|
|mom_permuted_single_vs_DINO_C4|-0.4647|[-2.9797, 0.9663]|
|mom_permuted_single_vs_Swin_C4|-0.8866|[-3.0944, 0.7327]|
|mom_permuted_c4_vs_qualification_convnext_tiny_A|1.1177|[-0.2088, 2.5920]|
|mom_permuted_c4_vs_B0_C4|0.1727|[-1.0329, 1.4020]|
|mom_permuted_c4_vs_dino_baseline|0.9218|[-1.4475, 2.9254]|
|mom_permuted_c4_vs_DINO_C4|-0.0559|[-2.6499, 1.5740]|
|mom_permuted_c4_vs_Swin_C4|-0.4779|[-2.4915, 1.0311]|
|mom_duplicate_single_vs_qualification_convnext_tiny_A|0.4722|[-0.4328, 1.7706]|
|mom_duplicate_single_vs_B0_C4|-0.4728|[-1.4203, 0.7911]|
|mom_duplicate_single_vs_dino_baseline|0.2763|[-1.6231, 2.2941]|
|mom_duplicate_single_vs_DINO_C4|-0.7015|[-2.7756, 0.7948]|
|mom_duplicate_single_vs_Swin_C4|-1.1234|[-2.9121, 0.5096]|
|mom_duplicate_c4_vs_qualification_convnext_tiny_A|1.2338|[0.2454, 2.5097]|
|mom_duplicate_c4_vs_B0_C4|0.2888|[-0.4849, 1.2626]|
|mom_duplicate_c4_vs_dino_baseline|1.0379|[-0.9041, 2.9936]|
|mom_duplicate_c4_vs_DINO_C4|0.0601|[-2.0976, 1.5732]|
|mom_duplicate_c4_vs_Swin_C4|-0.3618|[-2.0017, 1.0511]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原18项几何结构研究另有完整账本，不混为本轮结果。
