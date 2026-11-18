# 同层通道协方差：累计与消融对照

配对参考复用原M轮同seed有效模型，不重复训练。所有原B0/强参照指标仍见MODULE_ROUND_REPORT_ZH.md。

|比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|mom_covsqrt_single_vs_M_c4_moments|-0.0656|[-0.5546, 0.2600]|
|mom_covsqrt_single_vs_M_c4_aug|0.1241|[-0.3034, 0.7982]|
|mom_covsqrt_single_vs_M_eca|0.3581|[-0.5880, 1.5691]|
|mom_covsqrt_c4_vs_M_c4_moments|-0.4125|[-0.9035, 0.0369]|
|mom_covsqrt_c4_vs_M_c4_aug|-0.2826|[-1.2006, 1.1236]|
|mom_covsqrt_c4_vs_M_eca|0.1800|[-0.6846, 1.5312]|
|covsqrt_only_single_vs_M_c4_moments|-0.5993|[-1.8672, 0.0701]|
|covsqrt_only_single_vs_M_c4_aug|-0.4096|[-1.6962, 0.5808]|
|covsqrt_only_single_vs_M_eca|-0.1756|[-0.9534, 0.5206]|
|covsqrt_only_c4_vs_M_c4_moments|-0.5399|[-2.2867, 0.4620]|
|covsqrt_only_c4_vs_M_c4_aug|-0.4100|[-1.7226, 0.8321]|
|covsqrt_only_c4_vs_M_eca|0.0526|[-0.4121, 0.4441]|
|mom_rawcov_single_vs_M_c4_moments|-0.7180|[-1.7950, -0.0918]|
|mom_rawcov_single_vs_M_c4_aug|-0.5283|[-1.7076, 0.4987]|
|mom_rawcov_single_vs_M_eca|-0.2943|[-0.8084, 0.2167]|
|mom_rawcov_c4_vs_M_c4_moments|-0.6595|[-2.0273, 0.0790]|
|mom_rawcov_c4_vs_M_c4_aug|-0.5296|[-1.5020, 0.5688]|
|mom_rawcov_c4_vs_M_eca|-0.0669|[-0.3557, 0.2459]|
|mom_diag_single_vs_M_c4_moments|-0.8348|[-1.3619, -0.2990]|
|mom_diag_single_vs_M_c4_aug|-0.6450|[-1.3199, 0.4099]|
|mom_diag_single_vs_M_eca|-0.4111|[-1.0347, 0.5817]|
|mom_diag_c4_vs_M_c4_moments|-0.7015|[-1.5744, 0.0067]|
|mom_diag_c4_vs_M_c4_aug|-0.5716|[-1.3703, 0.8550]|
|mom_diag_c4_vs_M_eca|-0.1090|[-0.5878, 0.8157]|
|mom_channelperm_single_vs_M_c4_moments|-0.0687|[-0.8702, 0.3881]|
|mom_channelperm_single_vs_M_c4_aug|0.1211|[-0.7536, 0.8151]|
|mom_channelperm_single_vs_M_eca|0.3551|[-0.4507, 1.1641]|
|mom_channelperm_c4_vs_M_c4_moments|-0.1824|[-0.8835, 0.2443]|
|mom_channelperm_c4_vs_M_c4_aug|-0.0525|[-0.6340, 0.8964]|
|mom_channelperm_c4_vs_M_eca|0.4101|[-0.2748, 1.3903]|

mom_covsqrt对原Mom检验同层协方差读出增量；covsqrt_only真去Mom；rawcov去矩阵幂、diag仅16有效对角坐标、channelperm改变联合配对。全部seed/阴性保留，不能据点估计称创新或独立确认。
