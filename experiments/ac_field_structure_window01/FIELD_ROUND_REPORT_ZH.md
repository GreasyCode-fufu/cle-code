# 几何视场支持正式结构研究

固定B0：convnext_tiny_A；同主干预算参照W0：swin_t_B。W0按完整资格方案test均值选择，明确为适应性开发，不能称独立确认。

|方法|seed45 F1|seed46 F1|seed47 F1|平均F1|accuracy|恶性召回|患者等权accuracy|
|---|---:|---:|---:|---:|---:|---:|---:|
|qualification_swin_t_B|95.9891%|95.6657%|96.1759%|95.9436%|95.9491%|95.0758%|93.6555%|
|qualification_convnext_tiny_A|95.2515%|94.2258%|95.4670%|94.9814%|95.0231%|92.2348%|91.1468%|
|fov_residual|95.4279%|96.6702%|90.8811%|94.3264%|94.5023%|86.9318%|91.8583%|
|field_pool|96.1807%|96.6580%|95.9971%|96.2786%|96.2963%|93.3712%|93.5150%|
|reference|95.9728%|95.2208%|96.3177%|95.8371%|95.8912%|91.0985%|94.8284%|
|no_field|96.1452%|95.3111%|95.4225%|95.6263%|95.6597%|92.4242%|93.7053%|
|no_local_norm|95.8354%|95.8390%|91.2839%|94.3194%|94.4444%|89.0152%|91.7533%|

|配对比较|平均差（百分点）|患者条件95%区间（百分点）|
|---|---:|---|
|fov_residual_vs_qualification_convnext_tiny_A|-0.6550|[-3.0832, 1.0390]|
|fov_residual_vs_qualification_swin_t_B|-1.6172|[-4.5213, 0.0493]|
|field_pool_vs_qualification_convnext_tiny_A|1.2972|[0.0328, 3.6230]|
|field_pool_vs_fov_residual|1.9522|[0.4263, 4.9750]|
|field_pool_vs_qualification_swin_t_B|0.3350|[-0.5767, 1.6312]|
|reference_vs_qualification_convnext_tiny_A|0.8557|[-0.3362, 3.3539]|
|reference_vs_fov_residual|1.5107|[0.0650, 4.7670]|
|reference_vs_qualification_swin_t_B|-0.1065|[-1.1420, 1.5798]|
|no_field_vs_qualification_convnext_tiny_A|0.6448|[-0.7989, 2.6523]|
|no_field_vs_fov_residual|1.2999|[0.1156, 3.7510]|
|no_field_vs_qualification_swin_t_B|-0.3173|[-1.1702, 0.6479]|
|no_local_norm_vs_qualification_convnext_tiny_A|-0.6620|[-3.1316, 1.0086]|
|no_local_norm_vs_fov_residual|-0.0070|[-0.9450, 0.8482]|
|no_local_norm_vs_qualification_swin_t_B|-1.6241|[-4.0488, -0.3998]|

逐seed混淆矩阵、门值、checkpoint SHA、逐图预测与预算见round_results.json及evaluations。历史test已被查看，A–C构建用过test标签；等级是模型一致性代理。现test参与方法选择，条件bootstrap不消除选择偏差。成熟算子组合不自动构成SCI创新；普通阴性后继续有证据的机制调整。
