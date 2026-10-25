# 真实固定错配去Mom控制：原宏F1辅助报告，最终按截图全指标补审

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|misaligned_only_single_vs_M_c4_moments|-0.5966|[-1.8313, 0.1662]|
|misaligned_only_single_vs_M_c4_aug|-0.4069|[-1.6626, 0.8333]|
|misaligned_only_single_vs_M_eca|-0.1729|[-0.6497, 0.3689]|
|misaligned_only_c4_vs_M_c4_moments|-0.6481|[-2.2175, 0.3404]|
|misaligned_only_c4_vs_M_c4_aug|-0.5182|[-1.7487, 0.9575]|
|misaligned_only_c4_vs_M_eca|-0.0555|[-0.3955, 0.4301]|

mom_focalcontrast对原Mom检验中间特征新增贡献；focalcontrast_only真去18528Mom；focalraw去调制器中心化；additive用加法代乘法；misaligned固定错配调制器和query。最终2–3网络设计累计+1.5，单点以对父模型贡献评估。全部seed/阴性保留，不能据点估计称创新或独立确认。
