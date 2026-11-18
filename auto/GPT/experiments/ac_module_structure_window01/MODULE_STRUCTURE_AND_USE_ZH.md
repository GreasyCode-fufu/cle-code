# CLE共享模块正式轮：结构、复现与使用

本轮18项训练已预注册，真实CPU及CUDA数值预检通过；是否有效仍须等待全部配对seed、逐图预测与审计。这里的模型名指实验假设，不代表已经证实的创新点。

固定B0：公开预训练ConvNeXt-Tiny，输入224×224，四阶段96/192/384/768通道、空间56/28/14/7；全局平均池化、原LayerNorm、3类线性头。原B0三个seed已有完整有效训练，复用不重训。

|分支|插入位置/结构|训练参数|计入的逻辑GFLOPs/单视图|
|---|---|---:|---:|
|原B0/普通方向增强|原网络；方向增强只改变train输入|27,822,435|8.909531136|
|WTConv|阶段2、3末块DW7→原DW7＋两级Haar子带卷积/逆变换|28,053,411|8.942363136|
|SimAM|阶段3之后：x×sigmoid(空间能量)|27,822,435|8.909531136|
|ECA|阶段4之后：空间均值→通道Conv1d(k5)→sigmoid→逐通道缩放|27,822,440|8.909538816|
|EMA|阶段3之后：32组，轴向池化/GroupNorm与局部3×3双支，交叉空间权重|27,823,923|8.926347264|
|方向矩适配|stem原0度特征＋4旋转共享核响应的均值/标准差→零1×1投影|27,840,963|9.111840768|

FLOPs采用2 FLOP/MAC；覆盖Conv/Linear、WT DWT/IWT、ECA Conv1d、EMA矩阵乘、方向分支函数式卷积，排除归一化、池化、激活、逐元素运算、softmax、访存。SimAM的相同计数不代表没有实际耗时。四视图推理对应四倍前向计数；实际吞吐/显存/CPU/加载等待另由profile与训练日志报告。WT固定滤波器另有18,432个buffer元素，不混为可训练参数。

共享原件、官方原件和差异见MODULE_DIFFERENCES_ZH.md。模型新增模块仅进入独立models/optimize/ac_module_window01，未修改共享模块库。完整预注册见MODULE_PROTOCOL_ZH.md。

## 已完成的数值检查

CPU审计module_source_audit.json通过：共享/官方公式、梯度、Haar独立NumPy对照、原参数保持、RNG保持、各模型有限反传及基础图像/旋转RNG。原ECA逐位相等断言因double 4.44e-16舍入失败，保留v1；v2对输出和梯度使用明确1e-12容差，没有改网络、权重或训练配方。

CUDA preflight.json通过：所有分支真实batch32、两次AMP有限梯度更新、优化器覆盖、新模块梯度、全部3个原B0完整427图val概率重建、完整缓存SHA核验及test训练加载器拒绝。CUDA训练峰值约2.25–2.48GiB（另有框架上下文等开销）；正式并发仍须实际profile决定。这些是计算正确性证据，不是提点证据。

## 复现命令（服务器）

```bash
cd /home/login/data/cle/auto/GPT
PY=/home/login/miniconda3/envs/sunfuxiang/bin/python
# 读取当前真实入口和状态，先核对/proc，禁止重复启动存活的协调器
cat experiments/ac_module_structure_window01/bootstrap_identity_v2.json
cat experiments/ac_module_structure_window01/bootstrap_status_v2.json
cat experiments/ac_module_structure_window01/preflight.json
# 仅在确认同身份进程已终止且无存活子任务时，继续同一冻结研究
$PY experiments/ac_module_structure_window01/bootstrap_module_structure_v2.py
```

有效完整训练通过签名和checkpoint SHA复用。部分训练文件存在时训练引擎拒绝覆盖；应先查具体日志、存活worker和权重完成标记。CPU审计有效时不重复执行；preflight有效时不重复做；独立结果有效时不再训练。原bootstrap_failure.json属于已经诊断和修复的CPU断言v1，不能据它重复启动。

当前自动协调器先保留旧几何轮全部有效存活模型，自然结束并完成审计后再做本轮隔离并发profile及正式训练。每类实测1/2/4及资源允许的更多并发，取达到最佳97%的最小数，显存余量6GiB；活跃GPU已满时不因CPU数字增加无用任务。

若需要单独复现某个未完成任务（先核对同名任务不存在存活进程）：

```bash
$PY experiments/ac_module_structure_window01/run_module_structure.py worker --job module_wtconv_s45
```

## 完成后的模型使用

```bash
$PY experiments/ac_module_structure_window01/predict_module_model.py \
  --arm wtconv --seed 45 --image /absolute/path/image.png \
  --views single --output /absolute/path/prediction.json
```

`--arm`支持wtconv/simam/eca/ema/c4_moments/c4_aug；`--views c4`使用固定4方向概率均值。每个seed是一个模型，研究报告中的三seed均值不是三模型集成。脚本只接受已有完整训练、val选定且SHA正确的checkpoint；未完成时明确拒绝推理，不返回随机权重结果。类别保留原ID0/1/2，恶性为1，其余临床名称不猜测。模型属于研究用途，尚无独立临床验证结论。

训练结果将写runs及weights/ac_module_structure_window01；全量逐图预测、角度数组、混淆矩阵、对照差、患者条件区间及哈希位于evaluations和round_results.json。test已参与开发；报告将明确披露，不能称独立泛化确认。
