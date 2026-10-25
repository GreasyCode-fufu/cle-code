# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|mom_covsqrt_single|96.3265%|95.8076%|95.2901%|95.8080%|95.8333%|93.5606%|94.1421%|
|mom_covsqrt_c4|96.1564%|96.3359%|95.6438%|96.0454%|96.0648%|93.1818%|94.3481%|
|covsqrt_only_single|95.7889%|94.2258%|95.8083%|95.2743%|95.3125%|92.2348%|92.4791%|
|covsqrt_only_c4|96.6760%|94.9284%|96.1496%|95.9180%|95.9491%|92.9924%|93.2056%|
|mom_rawcov_single|95.9641%|93.8590%|95.6438%|95.1556%|95.1968%|91.8561%|92.5250%|
|mom_rawcov_c4|96.3019%|95.0916%|96.0018%|95.7984%|95.8333%|92.6136%|93.1978%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_covsqrt_single_vs_qualification_convnext_tiny_A|0.8266|[-0.0885, 2.3741]|
|mom_covsqrt_single_vs_B0_C4|-0.1184|[-1.0787, 1.3823]|
|mom_covsqrt_single_vs_dino_baseline|0.6307|[-1.2940, 2.9269]|
|mom_covsqrt_single_vs_DINO_C4|-0.3471|[-2.3598, 1.3749]|
|mom_covsqrt_single_vs_Swin_C4|-0.7690|[-2.2415, 0.7182]|
|mom_covsqrt_c4_vs_qualification_convnext_tiny_A|1.0639|[0.0357, 2.8128]|
|mom_covsqrt_c4_vs_B0_C4|0.1190|[-0.8305, 1.5996]|
|mom_covsqrt_c4_vs_dino_baseline|0.8681|[-0.9048, 3.0841]|
|mom_covsqrt_c4_vs_DINO_C4|-0.1097|[-2.0663, 1.5406]|
|mom_covsqrt_c4_vs_Swin_C4|-0.5316|[-2.0944, 1.1531]|
|covsqrt_only_single_vs_qualification_convnext_tiny_A|0.2929|[-0.0620, 0.8249]|
|covsqrt_only_single_vs_B0_C4|-0.6521|[-1.1396, -0.1145]|
|covsqrt_only_single_vs_dino_baseline|0.0970|[-2.1383, 2.1291]|
|covsqrt_only_single_vs_DINO_C4|-0.8807|[-3.2985, 0.7071]|
|covsqrt_only_single_vs_Swin_C4|-1.3027|[-3.0624, -0.1290]|
|covsqrt_only_c4_vs_qualification_convnext_tiny_A|0.9366|[0.2034, 1.7594]|
|covsqrt_only_c4_vs_B0_C4|-0.0084|[-0.3050, 0.2602]|
|covsqrt_only_c4_vs_dino_baseline|0.7407|[-1.6106, 2.7764]|
|covsqrt_only_c4_vs_DINO_C4|-0.2371|[-2.8421, 1.3981]|
|covsqrt_only_c4_vs_Swin_C4|-0.6590|[-2.5583, 0.4471]|
|mom_rawcov_single_vs_qualification_convnext_tiny_A|0.1742|[-0.4472, 1.0577]|
|mom_rawcov_single_vs_B0_C4|-0.7708|[-1.4244, 0.0524]|
|mom_rawcov_single_vs_dino_baseline|-0.0217|[-2.0472, 1.9775]|
|mom_rawcov_single_vs_DINO_C4|-0.9995|[-3.2420, 0.5323]|
|mom_rawcov_single_vs_Swin_C4|-1.4214|[-3.2369, 0.0512]|
|mom_rawcov_c4_vs_qualification_convnext_tiny_A|0.8170|[0.1670, 1.7027]|
|mom_rawcov_c4_vs_B0_C4|-0.1280|[-0.3973, 0.2286]|
|mom_rawcov_c4_vs_dino_baseline|0.6211|[-1.5273, 2.5833]|
|mom_rawcov_c4_vs_DINO_C4|-0.3567|[-2.7398, 1.1896]|
|mom_rawcov_c4_vs_Swin_C4|-0.7786|[-2.4382, 0.3480]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原几何与Swin局部研究另有完整账本，不混为本轮结果。
