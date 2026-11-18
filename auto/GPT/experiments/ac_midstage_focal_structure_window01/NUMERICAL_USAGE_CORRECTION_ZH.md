# 单图预测数值修正（结果后分析修复）

首真实mom_focalcontrast45、最早valbest2、固定首val2386、原权重SHA及1e-4阈值均不变。原默认cuDNN FP16 B1概率最大差0.00012111663818359375，超差记录及原CLI输出保留；未把原默认CLI称为通过。RGB256/中心224像素与缓存严格一致，FP32 B1也未解决。

显式关闭cuDNN卷积后端，仍使用原冻结预测源码、AMP FP16及物理B1（不填充到B32），native CUDA卷积单图复核误差见usage_smoke/independent_cli_audit.json，仍严格小于事前1e-4且类别一致。包装器逐forward记录B1形状及实际后端设置；m.verify、原checkpoint字节SHA和全部源cfg保持。该后端是在相同固定val图数值诊断后选择，明确是结果后使用修复，不伪称事前注册。旧失败文件及其SHA、诊断和新源码由cli_resolution_v2.json绑定。训练与全概率审计后端未改变，未重新训练，也无新test图；不推广到所有模型/图像数值差均小于阈值。

实际预测请使用S/predict_module_stable.py，参数与原predict_module_model.py相同，包含--arm/--seed/--image/--views/--output；仍拒绝未完成、权重SHA错误或源cfg漂移。native单图路径未重新测吞吐，不把训练cuDNN profile当作其性能。旧STRUCTURE_AND_USE及启动198包记录原始时点，原包不改；本修正覆盖其默认预测命令。
