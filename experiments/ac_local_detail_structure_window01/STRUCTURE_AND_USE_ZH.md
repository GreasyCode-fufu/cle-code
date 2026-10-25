# CLE局部细节强度对比：结构、训练与实际权重使用

保留原CNN方向矩M；224图像→原stem方向矩→第一stage96×56×56→局部细节零残差→原首次下采样→原余下stage→原GAP/norm/head。逐位置channelLN后固定replicate二项3×3高通D，用局部RMS每通道q减空间均值后调制D，再channelLN和零无bias96×96投影。仅9216新增可学习参数，固定核不训练。候选累计27,850,179参数、9.180481536G计数；highpass对照少一次blur。所有计数含固定F.conv2d，不计LN/sqrt/逐元素/pad/访存，真实吞吐见profile和完整资源审计。

五组×45/46/47；从原公开ImageNet同seed head起点独立训练，无CLE热启动。Q/M A224 CE60epoch/B32/encoder1e-5、新模块head1e-4、AdamW.05/cosine0/clip1/AMP FP16、HV翻转/rotation15/random224/stochasticdepth.1/workers2/thread1。每epoch固定2077唯一train图；val427按最早最大宏F1选实际权重，再test576和C4开发反馈。原B0和原Mom有效权重只复用。原标签/等级/患者split不变。完整权重文件/最早best/SHA记录在runs每job complete.json及最终weights_and_tasks.json；未完成的模型预测入口会拒绝，不静默生成随机权重结果。

## 服务器命令

在/home/login/data/cle/auto/GPT中，以login运行：

```bash
/home/login/miniconda3/envs/sunfuxiang/bin/python experiments/ac_local_detail_structure_window01/predict_module_model.py --arm mom_detailcontrast --seed 45 --image /绝对路径/实际图像.png --views single --output /绝对路径/预测.json
```

--views c4为四次前向平均概率，算力另列，不算额外网络创新。首权重固定首val2386实际CLI验证由check_prediction_cli.py自动执行。此轮事前冻结CLI nativeCUDA卷积后端（cudnnDisabled，原AMP FP16、物理B1），对原训练默认后端B32概率预设1e-4容差且类别同；不是承诺所有图/所有模型的误差上限，不以该单图耗时代表训练profile。禁止为获得通过而更换固定图/权重/阈值；任何失败保留、独立诊断，不重训已有效模型。

## 科学边界与来源

成熟binomial/BlurPool来源：Richard Zhang ICML2019 https://proceedings.mlr.press/v97/zhang19a.html ，Adobe官方源码与CC BY-NC-SA4.0许可、归属在CPU原型references，本任务为非商业研究适配。受限算子oracle只核replicate/3/stride1输出与输入梯度，不是官方整网复现。非线性高通残差不称抗混叠/整网等变/临床定位/稳健，成熟模块组合不自动是SCI创新。最终目标为2–3互补网络设计累计相对固定B0+1.5个百分点；单点以相对保留父模型新增贡献、seed稳定性、条件不确定性与成本评估。test均自适应内部开发，历史test已查看且AC构建用其标签；患者bootstrap为未校正设计选择偏差条件区间，不称独立或外部验证。全部普通阴性保留、整体active。
