# 支持机制原型研究账本

# 光学视野机制：原型实现和独立数值检查

更新：2026-10-24T19:40:10.394386+08:00。全部为实现预检查，尚无此机制的CLE分类器训练或性能结论。

## 当前主研究

基础主干公平资格比较仍在运行：新增14/21完成，另复用3个有效Base224；当前4个真实任务，GPU100%，无记录故障。最终基础baseline尚未选定，不能提前挑有利参照。

## 机制依据

固定AC train/val的RGB三通道完全相同。预先固定的单位圆覆盖约78.53%面积；仅在train2077上检查，圆外平均亮度中位约3.68e-5、圆外非黑像素比例中位0。圆内仍存在暗组织，因此使用几何视野支持，不用亮度判无效，更不改患者、标签、等级或AC清单。
待验证假设是边界/视野外特征参与空间混合与全局池化可能影响CLE泛化；用支持约束残差保持预训练路径，再由实验检验。卷积、attention和pool是同一机制的实现/消融，不能凑成三个创新点。

## 已执行的严格检查

- v1真实对照失败：无bias、分数support条件下多乘了一次有效mask，示例最大误差约.364477；二值mask和有bias分支没有暴露它。原代码、配置和失败日志完整保存。
- v2按作者实际无bias分支修正，不放宽误差门槛。20项CPU数值检查通过：独立NumPy公式、实际NVIDIA算子、group/depthwise/stride/dilation/偏置/分数和空支持、梯度、Swin奇数或小于窗口尺寸/shift、空窗口、零起点残差等；160组配对增强的原RGB完全相同。
- 四家族×七接口共28项完整CPU集成通过。全部公共源/head张量逐位保留；适用的零起点接口与原logits逐位相同；新gate梯度非零且有限；EfficientNet的30个内部SE pool保持原样；context缓存清理；只反向，不更新已拟合分类器。
- 全部2504张允许AC train/val在两个食谱上共5008次RGB、标签、权重、索引和读取后RNG状态逐位相同；额外160例RGBA整体增强与重放mask逐位空间对齐。没有任何test图像或神经分类器参与此数据检查。

## 224输入的计数预算

| 主干 | 原参数 | 新参数 | 原计数FLOPs | 支持残差计数FLOPs | 增幅 |
|---|---|---|---|---|---|
|convnext_tiny|27822435|27822455|8.910G|9.150G|2.70%|
|efficientnet_v2_s|20181331|20181363|5.697G|5.773G|1.34%|
|swin_t|27521661|27521675|8.980G|12.063G|34.34%|
|convnext_base|87569539|87569577|30.707G|31.205G|1.62%|

计数包括Conv/Linear/attention及支持计数卷积，两FLOP/MAC；显式补计F.conv2d/F.linear绕过子模块hook的第二支路。未计norm、插值、激活、softmax、逐点修正、pool和数据搬运。Swin额外attention计算明显更高，不能称零成本。这里不是实际GPU延迟/显存测量。

## 正式实验前仍须完成

等待公平资格比较全部24模型完成与独立审计，冻结有竞争力的B0、食谱和三个checkpoint。复制已通过原型到新正式目录并保留历史源码；将RGB切片显式contiguous以匹配原加载器内存布局，再执行真实batch32 CUDA AMP、完整427 baseline概率复现、多worker配对输入/RNG、全部参数与索引/预算及实际资源预检。预注册提交后才做结构训练。
接口已实现plain、field_pool、reference、fov_residual、no_field、no_local_norm、no_pool；没有提前用其性能筛选模型。reference是成熟算子方程参照，不声称整套原网络复现。

## 用户新增累计创新策略

按用户要求，以2–3条互补可验证机制累计推进：第一项有有效证据后保留，在其基础上加1–2项；保留逐步叠加与drop-one重新训练。固定B0持续作为主要参照，新增组合使用同起点和公平预算，不把数值修复/工程复现或成熟模块数量当论文创新。当前component_registry仍没有已确认有效组件。
每次改动均在train/val评估；最终统一test可以包含所有预冻结的逐步模型和消融。但看到test后不据此改模型并反复测试。最终平均macroF1严格增益>.015、跨seed稳定且患者区间下界>0才可靠达标。历史test和AC清单使用标签的局限持续披露，不保证论文创新/录用。

## 服务器复现命令

```bash
cd /home/login/data/cle/auto/GPT
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
P=experiments/ac_field_prototype_window01
$PY $P/check_field_ops_v2.py
$PY $P/check_field_model_cpu.py
$PY $P/check_paired_field_data.py
```
field_ops.py/check_field_ops.py是保留的v1失败版本；有效算子使用field_ops_v2.py。上述命令只做CPU实现检查，不训练CLE分类器。正式权重及原始医疗图像均留服务器。