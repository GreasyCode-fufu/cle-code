# 方向矩细化：累计与消融对照

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mean_only_single_vs_M_c4_moments|-0.8398|[-2.0105, -0.2099]|
|mean_only_single_vs_M_c4_aug|-0.6500|[-1.7739, 0.1618]|
|mean_only_single_vs_M_eca|-0.4161|[-1.1998, 0.1231]|
|mean_only_c4_vs_M_c4_moments|-0.5922|[-2.1470, 0.2629]|
|mean_only_c4_vs_M_c4_aug|-0.4623|[-1.5975, 0.5970]|
|mean_only_c4_vs_M_eca|0.0003|[-0.3868, 0.2876]|
|random_moments_single_vs_M_c4_moments|-0.2874|[-1.3681, 0.2285]|
|random_moments_single_vs_M_c4_aug|-0.0976|[-0.9302, 0.4700]|
|random_moments_single_vs_M_eca|0.1363|[-0.9975, 1.1386]|
|random_moments_c4_vs_M_c4_moments|-0.5227|[-1.0333, -0.1488]|
|random_moments_c4_vs_M_c4_aug|-0.3928|[-0.9831, 0.5056]|
|random_moments_c4_vs_M_eca|0.0699|[-0.9191, 1.5427]|

mean_only/random_moments与M_c4_moments的反向差检验完整机制；moments_eca与M_moments检验ECA增量，aug_moments与M_c4_aug检验结构是否超出成熟旋转增强。全部seed/阴性保留，不能据点估计称创新或独立确认。
