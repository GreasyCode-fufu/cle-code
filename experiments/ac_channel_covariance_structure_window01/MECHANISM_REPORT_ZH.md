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

mom_covsqrt对原Mom检验同层协方差读出增量；covsqrt_only真去Mom；rawcov去矩阵幂、diag仅16有效对角坐标、channelperm改变联合配对。全部seed/阴性保留，不能据点估计称创新或独立确认。
