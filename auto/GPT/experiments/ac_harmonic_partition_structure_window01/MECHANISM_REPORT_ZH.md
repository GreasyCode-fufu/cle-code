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
|shape_only_single_vs_M_c4_moments|-0.1720|[-1.4306, 0.4486]|
|shape_only_single_vs_M_c4_aug|0.0177|[-1.0073, 0.7317]|
|shape_only_single_vs_M_eca|0.2517|[-0.9725, 1.3068]|
|shape_only_c4_vs_M_c4_moments|-0.1647|[-1.5104, 0.5007]|
|shape_only_c4_vs_M_c4_aug|-0.0348|[-0.9104, 0.8128]|
|shape_only_c4_vs_M_eca|0.4279|[-0.7698, 1.5050]|
|mom_permuted_single_vs_M_c4_moments|-0.1832|[-1.9109, 0.6875]|
|mom_permuted_single_vs_M_c4_aug|0.0065|[-1.3861, 0.8873]|
|mom_permuted_single_vs_M_eca|0.2405|[-1.3260, 1.3946]|
|mom_permuted_c4_vs_M_c4_moments|-0.3588|[-1.5045, 0.2211]|
|mom_permuted_c4_vs_M_c4_aug|-0.2289|[-0.8704, 0.4016]|
|mom_permuted_c4_vs_M_eca|0.2338|[-0.9953, 1.3504]|
|mom_duplicate_single_vs_M_c4_moments|-0.4200|[-1.2600, 0.1019]|
|mom_duplicate_single_vs_M_c4_aug|-0.2303|[-1.0155, 0.5447]|
|mom_duplicate_single_vs_M_eca|0.0037|[-0.7124, 0.8502]|
|mom_duplicate_c4_vs_M_c4_moments|-0.2427|[-1.1454, 0.3149]|
|mom_duplicate_c4_vs_M_c4_aug|-0.1128|[-0.8504, 0.9150]|
|mom_duplicate_c4_vs_M_eca|0.3498|[-0.3867, 1.2555]|
|mom_rawh2_single_vs_M_c4_moments|-0.2424|[-1.2716, 0.3061]|
|mom_rawh2_single_vs_M_c4_aug|-0.0527|[-0.9211, 0.5138]|
|mom_rawh2_single_vs_M_eca|0.1813|[-0.8431, 0.9328]|
|mom_rawh2_c4_vs_M_c4_moments|-0.0037|[-1.0743, 0.6760]|
|mom_rawh2_c4_vs_M_c4_aug|0.1262|[-0.6134, 1.0229]|
|mom_rawh2_c4_vs_M_eca|0.5888|[-0.0049, 1.2588]|

mom_shape对原Mom检验功率增量；shape_only真去Mom；mom_permuted错循环；mom_duplicate重复σ控制；mom_rawh2保留成熟DFT幅值。全部seed/阴性保留，不能据点估计称创新或独立确认。
