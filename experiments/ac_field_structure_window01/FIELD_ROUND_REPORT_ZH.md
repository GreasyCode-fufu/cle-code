# 几何视场支持正式结构研究

固定B0：convnext_tiny_A；同主干预算参照W0：swin_t_B。W0按完整资格方案test均值选择，明确为适应性开发，不能称独立确认。

|方法|seed45 F1|seed46 F1|seed47 F1|平均F1|accuracy|恶性召回|患者等权accuracy|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_swin_t_B|95.9891%|95.6657%|96.1759%|95.9436%|95.9491%|95.0758%|93.6555%|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|fov_residual|95.4279%|96.6702%|90.8811%|94.3264%|94.5023%|86.9318%|91.8583%|
|field_pool|96.1807%|96.6580%|95.9971%|96.2786%|96.2963%|93.3712%|93.5150%|

|配对比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|fov_residual_vs_qualification_convnext_tiny_A|-0.6550|[-3.0832, 1.0390]|
|fov_residual_vs_qualification_swin_t_B|-1.6172|[-4.5213, 0.0493]|
|field_pool_vs_qualification_convnext_tiny_A|1.2972|[0.0328, 3.6230]|
|field_pool_vs_fov_residual|1.9522|[0.4263, 4.9750]|
|field_pool_vs_qualification_swin_t_B|0.3350|[-0.5767, 1.6312]|

逐seed混淆矩阵、门值、checkpoint SHA、逐图预测与预算见round_results.json及evaluations。历史test已被查看，A–C构建用过test标签；等级是模型一致性代理。现test参与方法选择，条件bootstrap不消除选择偏差。成熟算子组合不自动构成SCI创新；普通阴性后继续有证据的机制调整。
