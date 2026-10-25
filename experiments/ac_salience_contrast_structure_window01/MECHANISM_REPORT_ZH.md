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
|salience_only_single_vs_M_c4_moments|-0.9975|[-2.5001, -0.2637]|
|salience_only_single_vs_M_c4_aug|-0.8077|[-2.2956, 0.1193]|
|salience_only_single_vs_M_eca|-0.5737|[-1.8600, 0.2699]|
|salience_only_c4_vs_M_c4_moments|-0.6452|[-2.7207, 0.3721]|
|salience_only_c4_vs_M_c4_aug|-0.5153|[-1.8402, 0.1501]|
|salience_only_c4_vs_M_eca|-0.0527|[-1.8732, 0.9578]|
|mom_weightedmean_single_vs_M_c4_moments|-0.9604|[-2.2418, -0.3295]|
|mom_weightedmean_single_vs_M_c4_aug|-0.7706|[-1.9335, 0.2291]|
|mom_weightedmean_single_vs_M_eca|-0.5366|[-1.1310, -0.0657]|
|mom_weightedmean_c4_vs_M_c4_moments|-0.9511|[-2.6530, 0.0272]|
|mom_weightedmean_c4_vs_M_c4_aug|-0.8212|[-2.2450, 0.6679]|
|mom_weightedmean_c4_vs_M_eca|-0.3586|[-1.0592, 0.2106]|
|mom_misaligned_single_vs_M_c4_moments|-0.4857|[-1.7763, 0.1362]|
|mom_misaligned_single_vs_M_c4_aug|-0.2959|[-1.6016, 0.8286]|
|mom_misaligned_single_vs_M_eca|-0.0619|[-0.8041, 0.6659]|
|mom_misaligned_c4_vs_M_c4_moments|-0.8810|[-2.4227, -0.0756]|
|mom_misaligned_c4_vs_M_c4_aug|-0.7511|[-1.9255, 0.2933]|
|mom_misaligned_c4_vs_M_eca|-0.2885|[-1.0648, 0.3719]|

mom_salience对原Mom检验内容注意力对比增量；salience_only真去Mom；weightedmean去减均值；ungated更少参数非等容量；misaligned改变权重和值配对。全部seed/阴性保留，不能据点估计称创新或独立确认。
