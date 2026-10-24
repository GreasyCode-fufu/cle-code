# 共享模块与方向机制正式实验（持续研究）

固定B0：ConvNeXt-Tiny /224。test用于适应性开发，非独立泛化确认。原始val选权重，全部方法保留三个seed及相同C4推理预算参照。

|方法|45 F1|46 F1|47 F1|平均F1|准确率|恶性召回|患者等权准确率|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|dino_baseline|93.8572%|96.8627%|94.8121%|95.1773%|95.1968%|94.5076%|92.8292%|
|B0_C4|96.6760%|95.1069%|95.9964%|95.9264%|95.9491%|93.3712%|92.8477%|
|DINO_C4|95.9477%|96.3259%|96.1917%|96.1551%|96.1806%|94.5076%|94.6958%|
|Swin_C4|97.0389%|96.3527%|96.3394%|96.5770%|96.5856%|95.4545%|94.0282%|
|wtconv_single|95.9562%|96.1452%|94.0083%|95.3699%|95.4282%|91.8561%|93.1969%|
|wtconv_c4|96.4943%|96.1438%|95.2540%|95.9640%|96.0069%|93.3712%|92.8706%|
|simam_single|95.6189%|93.8388%|96.3578%|95.2718%|95.3125%|91.8561%|92.7174%|
|simam_c4|96.6760%|94.7630%|96.1759%|95.8716%|95.8912%|93.3712%|93.4775%|
|eca_single|95.4387%|94.7410%|96.1700%|95.4499%|95.4861%|92.6136%|92.6486%|
|eca_c4|96.5005%|95.1069%|95.9888%|95.8654%|95.8912%|93.1818%|93.4898%|
|ema_single|95.4320%|95.9824%|95.3027%|95.5724%|95.6019%|93.1818%|91.4306%|
|ema_c4|96.8343%|96.3385%|95.7662%|96.3130%|96.3542%|93.3712%|94.0472%|
|c4_moments_single|95.6438%|95.9901%|95.9870%|95.8736%|95.8912%|93.7500%|93.9043%|
|c4_moments_c4|96.6886%|96.3462%|96.3390%|96.4579%|96.4699%|94.3182%|94.9322%|
|c4_aug_single|96.1484%|95.4361%|95.4672%|95.6839%|95.7176%|93.5606%|93.8183%|
|c4_aug_c4|97.0235%|95.9691%|95.9915%|96.3280%|96.3542%|95.0758%|93.6482%|

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|wtconv_single_vs_qualification_convnext_tiny_A|0.3884|[-0.4923, 1.4418]|
|wtconv_single_vs_B0_C4|-0.5565|[-1.2884, 0.3119]|
|wtconv_single_vs_dino_baseline|0.1926|[-2.1730, 2.1861]|
|wtconv_single_vs_DINO_C4|-0.7852|[-3.4154, 0.8352]|
|wtconv_single_vs_Swin_C4|-1.2071|[-3.3147, 0.3606]|
|wtconv_c4_vs_qualification_convnext_tiny_A|0.9826|[-0.1318, 2.0072]|
|wtconv_c4_vs_B0_C4|0.0376|[-0.8284, 0.7035]|
|wtconv_c4_vs_dino_baseline|0.7867|[-1.7861, 2.7330]|
|wtconv_c4_vs_DINO_C4|-0.1910|[-3.1234, 1.4702]|
|wtconv_c4_vs_Swin_C4|-0.6130|[-2.9183, 0.8863]|
|simam_single_vs_qualification_convnext_tiny_A|0.2904|[-0.1862, 1.2002]|
|simam_single_vs_B0_C4|-0.6546|[-1.2546, 0.2199]|
|simam_single_vs_dino_baseline|0.0945|[-1.9552, 2.2507]|
|simam_single_vs_DINO_C4|-0.8833|[-3.0874, 0.8027]|
|simam_single_vs_Swin_C4|-1.3052|[-2.9065, 0.0127]|
|simam_c4_vs_qualification_convnext_tiny_A|0.8902|[0.1350, 1.7816]|
|simam_c4_vs_B0_C4|-0.0548|[-0.4088, 0.3150]|
|simam_c4_vs_dino_baseline|0.6943|[-1.5449, 2.6673]|
|simam_c4_vs_DINO_C4|-0.2835|[-2.7521, 1.3140]|
|simam_c4_vs_Swin_C4|-0.7054|[-2.4822, 0.3571]|
|eca_single_vs_qualification_convnext_tiny_A|0.4685|[-0.1919, 1.3613]|
|eca_single_vs_B0_C4|-0.4765|[-1.1394, 0.3933]|
|eca_single_vs_dino_baseline|0.2726|[-1.7574, 2.2138]|
|eca_single_vs_DINO_C4|-0.7052|[-2.9233, 0.8064]|
|eca_single_vs_Swin_C4|-1.1271|[-2.8083, 0.1869]|
|eca_c4_vs_qualification_convnext_tiny_A|0.8839|[0.1231, 1.8164]|
|eca_c4_vs_B0_C4|-0.0610|[-0.3980, 0.3272]|
|eca_c4_vs_dino_baseline|0.6880|[-1.4146, 2.5666]|
|eca_c4_vs_DINO_C4|-0.2897|[-2.6316, 1.1959]|
|eca_c4_vs_Swin_C4|-0.7116|[-2.4121, 0.4131]|
|ema_single_vs_qualification_convnext_tiny_A|0.5909|[-0.0790, 1.6723]|
|ema_single_vs_B0_C4|-0.3540|[-1.1552, 0.6765]|
|ema_single_vs_dino_baseline|0.3951|[-1.3855, 2.3298]|
|ema_single_vs_DINO_C4|-0.5827|[-2.5773, 0.8636]|
|ema_single_vs_Swin_C4|-1.0046|[-2.5685, 0.5067]|
|ema_c4_vs_qualification_convnext_tiny_A|1.3316|[0.5415, 2.7860]|
|ema_c4_vs_B0_C4|0.3866|[-0.3647, 1.5969]|
|ema_c4_vs_dino_baseline|1.1357|[-0.8709, 3.2383]|
|ema_c4_vs_DINO_C4|0.1579|[-2.1139, 1.7575]|
|ema_c4_vs_Swin_C4|-0.2640|[-2.0761, 1.4363]|
|c4_moments_single_vs_qualification_convnext_tiny_A|0.8922|[0.1299, 2.3971]|
|c4_moments_single_vs_B0_C4|-0.0528|[-0.8219, 1.3176]|
|c4_moments_single_vs_dino_baseline|0.6963|[-0.9804, 2.9300]|
|c4_moments_single_vs_DINO_C4|-0.2814|[-2.0666, 1.3720]|
|c4_moments_single_vs_Swin_C4|-0.7034|[-2.0306, 0.7460]|
|c4_moments_single_vs_c4_aug_single|0.1898|[-0.4024, 1.1644]|
|c4_moments_c4_vs_qualification_convnext_tiny_A|1.4765|[0.4744, 3.3405]|
|c4_moments_c4_vs_B0_C4|0.5315|[-0.3925, 2.1732]|
|c4_moments_c4_vs_dino_baseline|1.2806|[-0.4443, 3.4367]|
|c4_moments_c4_vs_DINO_C4|0.3028|[-1.5557, 1.9119]|
|c4_moments_c4_vs_Swin_C4|-0.1191|[-1.5348, 1.5166]|
|c4_moments_c4_vs_c4_aug_c4|0.1299|[-0.5362, 1.3597]|
|c4_aug_single_vs_qualification_convnext_tiny_A|0.7025|[-0.3471, 2.1872]|
|c4_aug_single_vs_B0_C4|-0.2425|[-1.3117, 1.1785]|
|c4_aug_single_vs_dino_baseline|0.5066|[-1.4976, 2.5111]|
|c4_aug_single_vs_DINO_C4|-0.4712|[-2.6189, 1.0246]|
|c4_aug_single_vs_Swin_C4|-0.8931|[-2.4485, 0.5492]|
|c4_aug_c4_vs_qualification_convnext_tiny_A|1.3466|[0.2159, 2.7544]|
|c4_aug_c4_vs_B0_C4|0.4016|[-0.7574, 1.6107]|
|c4_aug_c4_vs_dino_baseline|1.1507|[-1.2278, 3.1301]|
|c4_aug_c4_vs_DINO_C4|0.1729|[-2.4438, 1.7722]|
|c4_aug_c4_vs_Swin_C4|-0.2490|[-2.0592, 1.1087]|

逐图预测、每seed混淆矩阵、训练预算、SHA见round_results.json及evaluations。库模块为成熟参照；组合不自动构成SCI创新。历史test已被查看，A–C构建用过test标签，等级为模型一致性代理。条件bootstrap不校正适应性筛选偏差。原18项几何结构研究另有完整账本，不混为本轮结果。
