# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|mom_lag_single|95.0983%|93.9964%|96.1749%|95.0899%|95.1389%|90.7197%|93.5718%|
|mom_lag_c4|96.6691%|95.0897%|95.8372%|95.8653%|95.8912%|92.6136%|93.7380%|
|lag_only_single|91.6498%|93.6317%|96.0058%|93.7625%|93.8657%|87.3106%|91.6546%|
|lag_only_c4|93.4550%|95.2685%|95.8291%|94.8509%|94.9074%|90.1515%|92.2774%|
|mom_permuted_single|95.0678%|94.5382%|95.1220%|94.9093%|94.9653%|91.4773%|91.9408%|
|mom_permuted_c4|95.9550%|95.1034%|96.0002%|95.6862%|95.7176%|92.9924%|92.3356%|
|mom_duplicate_single|95.0636%|94.9430%|93.8277%|94.6115%|94.6759%|91.0985%|90.7945%|
|mom_duplicate_c4|96.1263%|95.4727%|94.5231%|95.3740%|95.4282%|92.2348%|91.3744%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_lag_single_vs_qualification_convnext_tiny_A|0.1084|[-0.8044, 1.6793]|
|mom_lag_single_vs_B0_C4|-0.8366|[-1.8558, 0.7007]|
|mom_lag_single_vs_dino_baseline|-0.0875|[-1.7699, 2.2305]|
|mom_lag_single_vs_DINO_C4|-1.0652|[-2.8256, 0.6488]|
|mom_lag_single_vs_Swin_C4|-1.4871|[-3.0435, 0.1507]|
|mom_lag_c4_vs_qualification_convnext_tiny_A|0.8839|[0.2326, 2.1584]|
|mom_lag_c4_vs_B0_C4|-0.0611|[-0.5871, 0.8055]|
|mom_lag_c4_vs_dino_baseline|0.6880|[-1.0519, 2.6943]|
|mom_lag_c4_vs_DINO_C4|-0.2897|[-2.2064, 1.1922]|
|mom_lag_c4_vs_Swin_C4|-0.7117|[-2.1569, 0.6144]|
|lag_only_single_vs_qualification_convnext_tiny_A|-1.2190|[-2.3151, 0.0931]|
|lag_only_single_vs_B0_C4|-2.1640|[-3.4405, -0.7763]|
|lag_only_single_vs_dino_baseline|-1.4149|[-4.0563, 1.0939]|
|lag_only_single_vs_DINO_C4|-2.3926|[-5.1616, -0.3060]|
|lag_only_single_vs_Swin_C4|-2.8146|[-4.8550, -1.1098]|
|lag_only_c4_vs_qualification_convnext_tiny_A|-0.1306|[-1.0243, 0.7988]|
|lag_only_c4_vs_B0_C4|-1.0755|[-1.7299, -0.4948]|
|lag_only_c4_vs_dino_baseline|-0.3264|[-2.7180, 1.7719]|
|lag_only_c4_vs_DINO_C4|-1.3042|[-3.8981, 0.4558]|
|lag_only_c4_vs_Swin_C4|-1.7261|[-3.6198, -0.4457]|
|mom_permuted_single_vs_qualification_convnext_tiny_A|-0.0721|[-0.8003, 0.6394]|
|mom_permuted_single_vs_B0_C4|-1.0171|[-1.7567, -0.3627]|
|mom_permuted_single_vs_dino_baseline|-0.2680|[-2.7088, 1.7479]|
|mom_permuted_single_vs_DINO_C4|-1.2458|[-3.9253, 0.3832]|
|mom_permuted_single_vs_Swin_C4|-1.6677|[-3.8788, -0.2015]|
|mom_permuted_c4_vs_qualification_convnext_tiny_A|0.7048|[0.0216, 1.3833]|
|mom_permuted_c4_vs_B0_C4|-0.2402|[-0.5674, -0.0295]|
|mom_permuted_c4_vs_dino_baseline|0.5089|[-1.8241, 2.4458]|
|mom_permuted_c4_vs_DINO_C4|-0.4689|[-3.0387, 1.1340]|
|mom_permuted_c4_vs_Swin_C4|-0.8908|[-2.7452, 0.2504]|
|mom_duplicate_single_vs_qualification_convnext_tiny_A|-0.3700|[-1.9342, 0.4670]|
|mom_duplicate_single_vs_B0_C4|-1.3150|[-2.7820, -0.4871]|
|mom_duplicate_single_vs_dino_baseline|-0.5659|[-3.8211, 1.6128]|
|mom_duplicate_single_vs_DINO_C4|-1.5436|[-5.1533, 0.3542]|
|mom_duplicate_single_vs_Swin_C4|-1.9655|[-5.0168, -0.2636]|
|mom_duplicate_c4_vs_qualification_convnext_tiny_A|0.3926|[-1.0906, 1.3539]|
|mom_duplicate_c4_vs_B0_C4|-0.5524|[-1.8420, 0.1373]|
|mom_duplicate_c4_vs_dino_baseline|0.1967|[-2.9525, 2.3443]|
|mom_duplicate_c4_vs_DINO_C4|-0.7810|[-4.2816, 1.1162]|
|mom_duplicate_c4_vs_Swin_C4|-1.2030|[-4.0665, 0.4715]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原几何与Swin局部研究另有完整账本，不混为本轮结果。
