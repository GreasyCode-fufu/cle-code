# 局部细节强度对比：累计与消融对照

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_detailcontrast_single_vs_M_c4_moments|-0.8433|[-2.3956, -0.1424]|
|mom_detailcontrast_single_vs_M_c4_aug|-0.6536|[-2.0623, 0.2873]|
|mom_detailcontrast_single_vs_M_eca|-0.4196|[-1.1592, -0.0473]|
|mom_detailcontrast_c4_vs_M_c4_moments|-0.4798|[-1.8221, 0.2415]|
|mom_detailcontrast_c4_vs_M_c4_aug|-0.3499|[-1.3371, 0.7470]|
|mom_detailcontrast_c4_vs_M_eca|0.1127|[-0.3975, 0.6338]|
|detailcontrast_only_single_vs_M_c4_moments|-0.5927|[-1.6026, 0.2044]|
|detailcontrast_only_single_vs_M_c4_aug|-0.4029|[-1.3110, 0.9012]|
|detailcontrast_only_single_vs_M_eca|-0.1689|[-0.7032, 0.6609]|
|detailcontrast_only_c4_vs_M_c4_moments|-0.7430|[-1.6204, -0.2105]|
|detailcontrast_only_c4_vs_M_c4_aug|-0.6131|[-1.4534, 0.6141]|
|detailcontrast_only_c4_vs_M_eca|-0.1505|[-0.6738, 0.6789]|

mom_detailcontrast对原Mom检验第一stage新增贡献；detailcontrast_only真去18528Mom；detailrawgate去q中心化；highpass仅D；misaligned固定错配q与D。最终2–3网络设计累计+1.5，单点以对父模型贡献评估。全部seed/阴性保留，不能据点估计称创新或独立确认。
