# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|mom_corr_single|94.9394%|94.9029%|94.3558%|94.7327%|94.7917%|90.7197%|91.0853%|
|mom_corr_c4|95.0989%|95.2740%|94.2074%|94.8601%|94.9074%|91.2879%|91.2466%|
|corr_only_single|94.7369%|94.5759%|94.1925%|94.5017%|94.5602%|89.0152%|91.9631%|
|corr_only_c4|95.7875%|95.6438%|94.7551%|95.3955%|95.4282%|91.6667%|92.5737%|
|mom_shuffle_single|95.7534%|94.1898%|95.2993%|95.0808%|95.1389%|91.2879%|92.8799%|
|mom_shuffle_c4|96.4669%|94.5868%|95.8230%|95.6256%|95.6597%|92.0455%|93.9780%|
|mom_bilinear_single|95.9666%|95.0329%|94.1879%|95.0625%|95.1389%|90.3409%|93.9503%|
|mom_bilinear_c4|96.6767%|95.8081%|94.9296%|95.8048%|95.8333%|92.8030%|93.1242%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_corr_single_vs_qualification_convnext_tiny_A|-0.2487|[-1.7095, 0.7192]|
|mom_corr_single_vs_B0_C4|-1.1937|[-2.5102, -0.4474]|
|mom_corr_single_vs_dino_baseline|-0.4446|[-3.1065, 1.4078]|
|mom_corr_single_vs_DINO_C4|-1.4224|[-4.4470, 0.1746]|
|mom_corr_single_vs_Swin_C4|-1.8443|[-4.4221, -0.3180]|
|mom_corr_c4_vs_qualification_convnext_tiny_A|-0.1213|[-1.8363, 0.9488]|
|mom_corr_c4_vs_B0_C4|-1.0663|[-2.5559, -0.3711]|
|mom_corr_c4_vs_dino_baseline|-0.3172|[-3.2493, 1.6704]|
|mom_corr_c4_vs_DINO_C4|-1.2950|[-4.6050, 0.5170]|
|mom_corr_c4_vs_Swin_C4|-1.7169|[-4.5047, -0.1911]|
|corr_only_single_vs_qualification_convnext_tiny_A|-0.4797|[-1.3685, 0.7885]|
|corr_only_single_vs_B0_C4|-1.4247|[-2.4196, -0.3322]|
|corr_only_single_vs_dino_baseline|-0.6756|[-2.5015, 1.4986]|
|corr_only_single_vs_DINO_C4|-1.6533|[-3.6767, 0.0012]|
|corr_only_single_vs_Swin_C4|-2.0753|[-3.7600, -0.5879]|
|corr_only_c4_vs_qualification_convnext_tiny_A|0.4140|[-1.0406, 1.7930]|
|corr_only_c4_vs_B0_C4|-0.5309|[-1.6574, 0.4201]|
|corr_only_c4_vs_dino_baseline|0.2182|[-2.1963, 2.1778]|
|corr_only_c4_vs_DINO_C4|-0.7596|[-3.5249, 0.8888]|
|corr_only_c4_vs_Swin_C4|-1.1815|[-3.4632, 0.4923]|
|mom_shuffle_single_vs_qualification_convnext_tiny_A|0.0994|[-0.6260, 1.1486]|
|mom_shuffle_single_vs_B0_C4|-0.8456|[-1.7031, 0.1701]|
|mom_shuffle_single_vs_dino_baseline|-0.0965|[-2.2368, 1.9183]|
|mom_shuffle_single_vs_DINO_C4|-1.0743|[-3.4236, 0.4677]|
|mom_shuffle_single_vs_Swin_C4|-1.4962|[-3.4747, 0.1558]|
|mom_shuffle_c4_vs_qualification_convnext_tiny_A|0.6441|[-0.0790, 1.7743]|
|mom_shuffle_c4_vs_B0_C4|-0.3009|[-0.7204, 0.3701]|
|mom_shuffle_c4_vs_dino_baseline|0.4482|[-1.6337, 2.5565]|
|mom_shuffle_c4_vs_DINO_C4|-0.5295|[-2.8279, 1.1424]|
|mom_shuffle_c4_vs_Swin_C4|-0.9515|[-2.6082, 0.4014]|
|mom_bilinear_single_vs_qualification_convnext_tiny_A|0.0811|[-0.8073, 1.3688]|
|mom_bilinear_single_vs_B0_C4|-0.8639|[-1.6653, 0.1568]|
|mom_bilinear_single_vs_dino_baseline|-0.1148|[-2.0972, 1.8863]|
|mom_bilinear_single_vs_DINO_C4|-1.0926|[-3.3180, 0.4231]|
|mom_bilinear_single_vs_Swin_C4|-1.5145|[-3.3315, 0.0564]|
|mom_bilinear_c4_vs_qualification_convnext_tiny_A|0.8233|[-0.3213, 2.0571]|
|mom_bilinear_c4_vs_B0_C4|-0.1216|[-0.8720, 0.7066]|
|mom_bilinear_c4_vs_dino_baseline|0.6275|[-1.5192, 2.4833]|
|mom_bilinear_c4_vs_DINO_C4|-0.3503|[-2.8345, 1.1603]|
|mom_bilinear_c4_vs_Swin_C4|-0.7722|[-2.7754, 0.8284]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原18项几何结构研究另有完整账本，不混为本轮结果。
