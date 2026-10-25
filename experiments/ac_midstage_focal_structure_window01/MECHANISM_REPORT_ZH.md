# 中间层聚焦对比：累计与消融对照

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_focalcontrast_single_vs_M_c4_moments|-1.0501|[-2.9067, -0.0343]|
|mom_focalcontrast_single_vs_M_c4_aug|-0.8603|[-2.7287, 0.5680]|
|mom_focalcontrast_single_vs_M_eca|-0.6264|[-1.7355, 0.1817]|
|mom_focalcontrast_c4_vs_M_c4_moments|-0.6131|[-2.2829, 0.3239]|
|mom_focalcontrast_c4_vs_M_c4_aug|-0.4832|[-1.8839, 0.9714]|
|mom_focalcontrast_c4_vs_M_eca|-0.0206|[-0.9394, 0.8272]|

mom_focalcontrast对原Mom检验中间特征新增贡献；focalcontrast_only真去18528Mom；focalraw去调制器中心化；additive用加法代乘法；misaligned固定错配调制器和query。最终2–3网络设计累计+1.5，单点以对父模型贡献评估。全部seed/阴性保留，不能据点估计称创新或独立确认。
