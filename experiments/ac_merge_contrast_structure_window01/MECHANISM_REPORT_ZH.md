# Swin局部空间对比：累计与消融对照

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_only_single_vs_M_c4_moments|-0.0325|[-1.8953, 1.6247]|
|mom_only_c4_vs_M_c4_moments|0.3533|[-1.0373, 1.7735]|
|mom_local_single_vs_M_c4_moments|-0.1210|[-1.2804, 1.2762]|
|mom_local_c4_vs_M_c4_moments|0.0897|[-1.0925, 1.6361]|
|local_only_single_vs_M_c4_moments|-0.0414|[-1.8156, 1.6019]|
|local_only_c4_vs_M_c4_moments|0.0632|[-1.3206, 1.6102]|
|mom_mean_single_vs_M_c4_moments|-0.5425|[-1.9389, 0.9502]|
|mom_mean_c4_vs_M_c4_moments|-0.1659|[-1.1958, 1.4078]|
|mom_global_single_vs_M_c4_moments|0.0720|[-1.2733, 1.6817]|
|mom_global_c4_vs_M_c4_moments|-0.0506|[-1.1008, 1.6638]|

mom_only检验向Swin迁移；mom_local对mom_only检验累计局部增量；local_only去Mom；mom_mean控制均值与参数；mom_global移除局部空间定位。全部seed/阴性保留，不能据点估计称创新或独立确认。
