# 患者条件表征：累计与消融对照

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_cross_single_vs_M_c4_moments|-0.4875|[-1.2910, 0.0873]|
|mom_cross_single_vs_M_c4_aug|-0.2977|[-1.1247, 0.7757]|
|mom_cross_single_vs_M_eca|-0.0638|[-0.4981, 0.5715]|
|mom_cross_c4_vs_M_c4_moments|-0.6000|[-1.6947, 0.1196]|
|mom_cross_c4_vs_M_c4_aug|-0.4701|[-1.2722, 0.7456]|
|mom_cross_c4_vs_M_eca|-0.0075|[-0.4542, 0.6581]|

mom_cross与原M检验辅助目标增量，cross_only去方向矩，mom_supcon是成熟标准目标，mom_within检验相反患者内机制。全部seed/阴性保留，不能据点估计称创新或独立确认。
