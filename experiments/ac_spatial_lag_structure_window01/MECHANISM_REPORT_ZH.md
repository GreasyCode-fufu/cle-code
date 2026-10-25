# 末层空间间隔关联：累计与消融对照

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_lag_single_vs_M_c4_moments|-0.7838|[-1.4925, 0.0604]|
|mom_lag_single_vs_M_c4_aug|-0.5940|[-1.4917, 0.8883]|
|mom_lag_single_vs_M_eca|-0.3600|[-0.9485, 0.5700]|
|mom_lag_c4_vs_M_c4_moments|-0.5926|[-1.5251, 0.0972]|
|mom_lag_c4_vs_M_c4_aug|-0.4627|[-1.3101, 0.9354]|
|mom_lag_c4_vs_M_eca|-0.0000|[-0.4343, 0.7037]|
|lag_only_single_vs_M_c4_moments|-2.1112|[-3.9134, -0.8862]|
|lag_only_single_vs_M_c4_aug|-1.9214|[-3.8445, -0.1754]|
|lag_only_single_vs_M_eca|-1.6875|[-2.8586, -0.7093]|
|lag_only_c4_vs_M_c4_moments|-1.6070|[-3.4574, -0.5249]|
|lag_only_c4_vs_M_c4_aug|-1.4771|[-2.8686, -0.1788]|
|lag_only_c4_vs_M_eca|-1.0145|[-1.7459, -0.4205]|
|mom_permuted_single_vs_M_c4_moments|-0.9643|[-2.3798, -0.3187]|
|mom_permuted_single_vs_M_c4_aug|-0.7746|[-2.2721, 0.1590]|
|mom_permuted_single_vs_M_eca|-0.5406|[-1.4326, 0.0660]|
|mom_permuted_c4_vs_M_c4_moments|-0.7717|[-2.4706, 0.1283]|
|mom_permuted_c4_vs_M_c4_aug|-0.6418|[-1.8797, 0.3555]|
|mom_permuted_c4_vs_M_eca|-0.1792|[-0.7279, 0.1867]|
|mom_duplicate_single_vs_M_c4_moments|-1.2622|[-3.7035, -0.1723]|
|mom_duplicate_single_vs_M_c4_aug|-1.0724|[-3.5071, 0.2640]|
|mom_duplicate_single_vs_M_eca|-0.8384|[-2.5012, -0.0450]|
|mom_duplicate_c4_vs_M_c4_moments|-1.0839|[-3.4970, 0.1285]|
|mom_duplicate_c4_vs_M_c4_aug|-0.9540|[-2.8787, 0.2676]|
|mom_duplicate_c4_vs_M_eca|-0.4913|[-1.9278, 0.3091]|

mom_lag对原Mom检验空间间隔增量；lag_only真去Mom；permuted改邻接、duplicate去lag3、acf同间隔成熟全能量归一控制。全部seed/阴性保留，不能据点估计称创新或独立确认。
