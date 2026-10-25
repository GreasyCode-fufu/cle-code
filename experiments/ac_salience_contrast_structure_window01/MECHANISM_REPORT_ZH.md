# 内容注意力对比：累计与消融对照

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_salience_single_vs_M_c4_moments|-1.2287|[-2.4105, -0.7347]|
|mom_salience_single_vs_M_c4_aug|-1.0389|[-2.1703, -0.2371]|
|mom_salience_single_vs_M_eca|-0.8049|[-1.6745, -0.0788]|
|mom_salience_c4_vs_M_c4_moments|-0.8199|[-2.4929, -0.0725]|
|mom_salience_c4_vs_M_c4_aug|-0.6900|[-1.9926, 0.1523]|
|mom_salience_c4_vs_M_eca|-0.2273|[-1.6140, 0.7697]|

mom_salience对原Mom检验内容注意力对比增量；salience_only真去Mom；weightedmean去减均值；ungated更少参数非等容量；misaligned改变权重和值配对。全部seed/阴性保留，不能据点估计称创新或独立确认。
