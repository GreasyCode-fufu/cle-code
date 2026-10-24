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

mom_corr与M_c4_moments检验跨尺度增量；corr_only检验去方向矩；mom_shuffle等参数打乱对应，mom_bilinear保留成熟非居中二阶算子。全部seed/阴性保留，不能据点估计称创新或独立确认。
