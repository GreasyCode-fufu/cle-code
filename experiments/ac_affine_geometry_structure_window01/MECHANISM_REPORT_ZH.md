# 局部几何采样：累计与消融对照

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_affine_single_vs_M_c4_moments|-0.7123|[-1.5556, -0.1855]|
|mom_affine_single_vs_M_c4_aug|-0.5226|[-1.4487, 0.3966]|
|mom_affine_single_vs_M_eca|-0.2886|[-0.9466, 0.4969]|
|mom_affine_c4_vs_M_c4_moments|-0.8782|[-1.8592, -0.2987]|
|mom_affine_c4_vs_M_c4_aug|-0.7482|[-1.5748, 0.3957]|
|mom_affine_c4_vs_M_eca|-0.2856|[-0.8332, 0.4775]|
|affine_only_single_vs_M_c4_moments|-1.0715|[-2.5876, -0.3128]|
|affine_only_single_vs_M_c4_aug|-0.8817|[-2.4799, 0.2063]|
|affine_only_single_vs_M_eca|-0.6477|[-1.5788, -0.1476]|
|affine_only_c4_vs_M_c4_moments|-0.6446|[-2.3494, 0.2802]|
|affine_only_c4_vs_M_c4_aug|-0.5147|[-1.7955, 0.6263]|
|affine_only_c4_vs_M_eca|-0.0521|[-0.4622, 0.2726]|
|mom_permuted_single_vs_M_c4_moments|-0.3478|[-0.8785, -0.0042]|
|mom_permuted_single_vs_M_c4_aug|-0.1580|[-0.5847, 0.4666]|
|mom_permuted_single_vs_M_eca|0.0760|[-0.8576, 1.2156]|
|mom_permuted_c4_vs_M_c4_moments|-0.3560|[-0.9800, 0.0287]|
|mom_permuted_c4_vs_M_c4_aug|-0.2261|[-0.8998, 0.8211]|
|mom_permuted_c4_vs_M_eca|0.2365|[-0.5970, 1.4489]|
|mom_dcn_single_vs_M_c4_moments|-0.7732|[-1.8464, 0.0105]|
|mom_dcn_single_vs_M_c4_aug|-0.5834|[-1.7201, 0.6640]|
|mom_dcn_single_vs_M_eca|-0.3494|[-0.9008, 0.3449]|
|mom_dcn_c4_vs_M_c4_moments|-1.1705|[-2.5344, -0.2872]|
|mom_dcn_c4_vs_M_c4_aug|-1.0405|[-2.1462, 0.3559]|
|mom_dcn_c4_vs_M_eca|-0.5779|[-1.0976, 0.1218]|

mom_affine与原M方向矩检验新增几何贡献，affine_only去一，mom_permuted同参数打乱，mom_dcn保留自由DCN成熟算子。全部seed/阴性保留，不能据点估计称创新或独立确认。
