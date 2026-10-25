# 末层空间奇统计：累计与消融对照

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_odd_single_vs_M_c4_moments|-0.6665|[-1.8631, 0.1683]|
|mom_odd_single_vs_M_c4_aug|-0.4768|[-1.8267, 0.8499]|
|mom_odd_single_vs_M_eca|-0.2428|[-0.8854, 0.5652]|
|mom_odd_c4_vs_M_c4_moments|-0.5453|[-2.0464, 0.3707]|
|mom_odd_c4_vs_M_c4_aug|-0.4154|[-1.6454, 1.0641]|
|mom_odd_c4_vs_M_eca|0.0473|[-0.4844, 0.7611]|
|odd_only_single_vs_M_c4_moments|-1.1440|[-2.8015, -0.1627]|
|odd_only_single_vs_M_c4_aug|-0.9542|[-2.5914, 0.4261]|
|odd_only_single_vs_M_eca|-0.7203|[-1.5471, -0.1083]|
|odd_only_c4_vs_M_c4_moments|-1.1753|[-3.0363, -0.0863]|
|odd_only_c4_vs_M_c4_aug|-1.0453|[-2.5691, 0.4000]|
|odd_only_c4_vs_M_eca|-0.5827|[-1.3780, 0.1047]|
|mom_raw3_single_vs_M_c4_moments|-0.3644|[-1.6033, 0.1591]|
|mom_raw3_single_vs_M_c4_aug|-0.1747|[-1.0121, 0.2991]|
|mom_raw3_single_vs_M_eca|0.0593|[-1.2329, 0.9326]|
|mom_raw3_c4_vs_M_c4_moments|-0.4940|[-1.6747, 0.0385]|
|mom_raw3_c4_vs_M_c4_aug|-0.3641|[-1.3519, 0.7130]|
|mom_raw3_c4_vs_M_eca|0.0986|[-0.8698, 1.1885]|

mom_odd对原Mom检验末层奇统计增量；odd_only真去Mom；raw3/std/even为同参数成熟矩与去符号控制。全部seed/阴性保留，不能据点估计称创新或独立确认。
