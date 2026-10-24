# test反馈的成本与下一轮依据

此处为用户授权的适应性开发分析。所有test反馈均留痕，不能作为独立泛化确认。患者错误只用于设计分析，不改标签、分级、阈值、清单或样本。

|方法|参数(M)|计算(GFLOPs)|crop|
|---|---:|---:|---:|
|dino_baseline|21.63111|12.24713|224|
|round1_relational|21.81649|12.30405|224|
|round1_no_neighbor|21.81649|12.30405|224|
|round1_no_depth|21.81649|12.30405|224|
|round1_final_only|21.81649|12.30405|224|
|round1_official4|21.63456|12.24714|224|
|round1_convnext|87.56954|30.70742|224|
|qualification_convnext_tiny_A|27.82243|8.90953|224|
|qualification_convnext_tiny_B|27.82243|26.18310|384|
|qualification_efficientnet_v2_s_A|20.18133|5.69724|224|
|qualification_efficientnet_v2_s_B|20.18133|16.72868|384|
|qualification_swin_t_A|27.52166|8.97960|224|
|qualification_swin_t_B|27.52166|28.59787|384|
|qualification_convnext_base_A|87.56954|30.70742|224|
|qualification_convnext_base_B|87.56954|90.24219|384|

FLOPs按2 FLOP/MAC，包含显式注意力，其他排除项见JSON。不同原训练并发下的总耗时不能直接当独立速度基准比较。完整逐seed混淆矩阵、原始训练耗时、此次推理/独立复算耗时在成本JSON中。

|同主干384相对224|seed45差|seed46差|seed47差|平均差|患者条件95%区间|计算倍数|
|---|---:|---:|---:|---:|---|---:|
|convnext_tiny|1.3972|1.2576|-1.4419|0.4043|[-0.5875, 1.9833]|2.939|
|efficientnet_v2_s|-0.9032|0.0198|1.1065|0.0743|[-2.5340, 2.8484]|2.936|
|swin_t|1.4254|0.9325|1.6387|1.3322|[-0.2549, 3.5918]|3.185|
|convnext_base|-0.0102|0.7061|0.8477|0.5146|[-0.8452, 2.4879]|2.939|

差值单位为百分点。分辨率变化是成熟训练设置，不冒称网络创新。条件区间不校正反复test选择。下一轮采用NEXT_MECHANISM_PREREGISTRATION_ZH.md中的固定B0/工作W0规则和几何视场支持H1，先正式CUDA数值与吞吐预检并提交再训练。

资格实验还未完成时，不锁定W0，不用暂时分数替换主要baseline；后续全部资格方法完成后需重新生成本附表。
