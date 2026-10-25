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

mom_lag对原Mom检验空间间隔增量；lag_only真去Mom；permuted改邻接、duplicate去lag3、acf同间隔成熟全能量归一控制。全部seed/阴性保留，不能据点估计称创新或独立确认。
