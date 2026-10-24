# C4功率形状：累计与消融对照

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_shape_single_vs_M_c4_moments|-0.6428|[-2.4035, 0.2636]|
|mom_shape_single_vs_M_c4_aug|-0.4530|[-1.7556, 0.3504]|
|mom_shape_single_vs_M_eca|-0.2191|[-1.8169, 0.9157]|
|mom_shape_c4_vs_M_c4_moments|-0.5842|[-1.9058, 0.0391]|
|mom_shape_c4_vs_M_c4_aug|-0.4543|[-1.2082, 0.0960]|
|mom_shape_c4_vs_M_eca|0.0083|[-1.3539, 1.1126]|

mom_shape对原Mom检验功率增量；shape_only真去Mom；mom_permuted错循环；mom_duplicate重复σ控制；mom_rawh2保留成熟DFT幅值。全部seed/阴性保留，不能据点估计称创新或独立确认。
