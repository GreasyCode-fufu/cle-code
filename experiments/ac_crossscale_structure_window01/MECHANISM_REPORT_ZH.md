# 跨尺度对应统计：累计与消融对照

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_corr_single_vs_M_c4_moments|-1.1409|[-3.1727, -0.2809]|
|mom_corr_single_vs_M_c4_aug|-0.9512|[-2.8366, 0.0685]|
|mom_corr_single_vs_M_eca|-0.7172|[-2.0312, -0.1346]|
|mom_corr_c4_vs_M_c4_moments|-1.5978|[-3.8488, -0.4888]|
|mom_corr_c4_vs_M_c4_aug|-1.4679|[-3.2894, -0.3196]|
|mom_corr_c4_vs_M_eca|-1.0053|[-2.4517, -0.3270]|
|corr_only_single_vs_M_c4_moments|-1.3719|[-2.2832, -0.7546]|
|corr_only_single_vs_M_c4_aug|-1.1821|[-2.2585, 0.0647]|
|corr_only_single_vs_M_eca|-0.9482|[-1.7096, -0.1061]|
|corr_only_c4_vs_M_c4_moments|-1.0624|[-2.5890, -0.3602]|
|corr_only_c4_vs_M_c4_aug|-0.9325|[-2.2769, 0.3172]|
|corr_only_c4_vs_M_eca|-0.4699|[-1.5003, 0.3685]|
|mom_shuffle_single_vs_M_c4_moments|-0.7928|[-1.9638, -0.1349]|
|mom_shuffle_single_vs_M_c4_aug|-0.6031|[-1.7249, 0.2996]|
|mom_shuffle_single_vs_M_eca|-0.3691|[-1.1262, 0.4039]|
|mom_shuffle_c4_vs_M_c4_moments|-0.8324|[-2.1746, 0.0091]|
|mom_shuffle_c4_vs_M_c4_aug|-0.7025|[-1.7266, 0.5953]|
|mom_shuffle_c4_vs_M_eca|-0.2398|[-0.6664, 0.3839]|
|mom_bilinear_single_vs_M_c4_moments|-0.8112|[-1.9721, -0.1410]|
|mom_bilinear_single_vs_M_c4_aug|-0.6214|[-1.6393, 0.3313]|
|mom_bilinear_single_vs_M_eca|-0.3874|[-1.1125, 0.4245]|
|mom_bilinear_c4_vs_M_c4_moments|-0.6531|[-2.0578, -0.0159]|
|mom_bilinear_c4_vs_M_c4_aug|-0.5232|[-1.6526, 0.6261]|
|mom_bilinear_c4_vs_M_eca|-0.0606|[-0.7451, 0.6535]|

mom_corr与M_c4_moments检验跨尺度增量；corr_only检验去方向矩；mom_shuffle等参数打乱对应，mom_bilinear保留成熟非居中二阶算子。全部seed/阴性保留，不能据点估计称创新或独立确认。
