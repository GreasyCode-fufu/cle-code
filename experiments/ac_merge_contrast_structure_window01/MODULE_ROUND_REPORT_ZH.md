# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|qualification_swin_t_B|95.9891%|95.6657%|96.1759%|95.9436%|95.9491%|95.0758%|93.6555%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|mom_only_single|95.8627%|95.3271%|96.3337%|95.8412%|95.8333%|95.0758%|93.6163%|
|mom_only_c4|97.5597%|96.1968%|96.6772%|96.8112%|96.8171%|95.6439%|94.6239%|
|mom_local_single|95.8496%|95.4253%|95.9831%|95.7526%|95.7755%|92.4242%|93.0688%|
|mom_local_c4|96.8502%|96.1255%|96.6672%|96.5477%|96.5856%|92.8030%|94.3718%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_only_single_vs_qualification_convnext_tiny_A|0.8597|[-0.9733, 3.2888]|
|mom_only_single_vs_qualification_swin_t_B|-0.1024|[-0.6477, 0.4339]|
|mom_only_single_vs_B0_C4|-0.0852|[-1.7757, 2.0894]|
|mom_only_single_vs_dino_baseline|0.6638|[-1.4819, 2.6844]|
|mom_only_single_vs_DINO_C4|-0.3139|[-2.4184, 1.2808]|
|mom_only_single_vs_Swin_C4|-0.7358|[-1.6170, -0.0478]|
|mom_only_c4_vs_qualification_convnext_tiny_A|1.8298|[0.5487, 3.9728]|
|mom_only_c4_vs_qualification_swin_t_B|0.8677|[0.0742, 1.8938]|
|mom_only_c4_vs_B0_C4|0.8848|[-0.2848, 2.7841]|
|mom_only_c4_vs_dino_baseline|1.6339|[-0.0315, 3.5474]|
|mom_only_c4_vs_DINO_C4|0.6561|[-0.9700, 2.0155]|
|mom_only_c4_vs_Swin_C4|0.2342|[-0.4009, 1.0553]|
|mom_local_single_vs_qualification_convnext_tiny_A|0.7712|[-0.3741, 2.9883]|
|mom_local_single_vs_qualification_swin_t_B|-0.1909|[-1.2415, 1.2288]|
|mom_local_single_vs_B0_C4|-0.1738|[-1.3815, 1.9327]|
|mom_local_single_vs_dino_baseline|0.5753|[-1.0707, 2.8725]|
|mom_local_single_vs_DINO_C4|-0.4024|[-1.9563, 1.2356]|
|mom_local_single_vs_Swin_C4|-0.8244|[-1.7705, 0.2628]|
|mom_local_c4_vs_qualification_convnext_tiny_A|1.5662|[0.5165, 3.6970]|
|mom_local_c4_vs_qualification_swin_t_B|0.6041|[-0.6116, 2.3143]|
|mom_local_c4_vs_B0_C4|0.6212|[-0.5341, 2.6368]|
|mom_local_c4_vs_dino_baseline|1.3703|[-0.0861, 3.6889]|
|mom_local_c4_vs_DINO_C4|0.3926|[-0.9792, 2.0322]|
|mom_local_c4_vs_Swin_C4|-0.0294|[-1.0895, 1.3286]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原18项几何结构研究另有完整账本，不混为本轮结果。
