# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|mom_detailcontrast_single|95.4315%|94.3578%|95.3018%|95.0303%|95.0810%|91.8561%|91.4996%|
|mom_detailcontrast_c4|96.6617%|95.2743%|95.9983%|95.9781%|96.0069%|93.1818%|93.2515%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_detailcontrast_single_vs_qualification_convnext_tiny_A|0.0489|[-0.9241, 0.8699]|
|mom_detailcontrast_single_vs_B0_C4|-0.8961|[-1.8645, -0.1111]|
|mom_detailcontrast_single_vs_dino_baseline|-0.1470|[-2.5266, 1.7754]|
|mom_detailcontrast_single_vs_DINO_C4|-1.1248|[-3.7545, 0.4311]|
|mom_detailcontrast_single_vs_Swin_C4|-1.5467|[-3.7441, -0.0378]|
|mom_detailcontrast_c4_vs_qualification_convnext_tiny_A|0.9967|[0.0531, 2.1517]|
|mom_detailcontrast_c4_vs_B0_C4|0.0517|[-0.5279, 0.6826]|
|mom_detailcontrast_c4_vs_dino_baseline|0.8008|[-1.2421, 2.6044]|
|mom_detailcontrast_c4_vs_DINO_C4|-0.1770|[-2.5147, 1.2831]|
|mom_detailcontrast_c4_vs_Swin_C4|-0.5989|[-2.3896, 0.7603]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原几何与Swin局部研究另有完整账本，不混为本轮结果。
