# 代码架构分析报告

## 1. 基础信息
- 代码来源：远端本地目录 `/data/cle/sunfuxiang-2/ultralytics-main`
- 本地包校验值：`f6a2fc2a3d5a5f785f7ddad7d91cc4814f952aa2ebb7f556135a79e9da03685d`
- 分析日期：2026-07-31T23:44:38.650822+08:00
- 完成状态：完整完成
- 整体分析置信度：高（动态Ultralytics内部边界为中）

## 2. 项目目录分层总览
- `ultralytics_ext/`：特征钩子、Gram损失、复用训练器、MTRD损失/子类
- `scripts/`：Gram与MTRD训练入口、temperature/权重/梯度诊断
- `tests/`：合成损失测试与真实pipeline测试
- `runs/analysis/mtrd_implementation_check/`：诊断和完整性报告

## 3. 可执行入口与主入口判断
| 入口路径 | 类型 | 主要用途 | 判断依据 | 置信度 |
|---|---|---|---|---|
| `scripts/markov_distill_train.py` | argparse | MTRD inspect/smoke/train | main与trainer factory | 高 |
| `scripts/calibrate_markov_temperature.py` | argparse | 训练batch熵标定 | main与训练loader | 高 |
| `scripts/calibrate_markov_kd_weight.py` | argparse | 训练batch损失比例标定 | main与compute_kd_step | 高 |
| `scripts/diagnose_markov_gradients.py` | argparse | 三尺度梯度与Gram余弦 | autograd.grad | 高 |
| `scripts/distill_train.py` | argparse | 原Gram RelKD | 默认KDDetectionTrainer | 高 |

## 4. 核心入口完整执行数据流
- 数据加载：RelKD `args.yaml`中的data配置进入DetectionTrainer训练loader。
- 数据预处理：`preprocess_batch`，Teacher前向允许AMP。
- 模型前向：Teacher eval/no_grad；Student训练前向产生检测损失。
- 特征交互：P3/P4/P5钩子各触发一次，MTRD在float32构造`[B,C,C]`转移。
- 损失：逐行`KL(P_T^k || P_S^k)`，按步和尺度加权。
- 反向与更新：`det_loss + kd_weight*ramp*mtrd_raw*batch_size`进入GradScaler；只更新Student和EMA。
- 保存：Ultralytics结果与`mtrd_metrics.csv`；Teacher不进入checkpoint。
- 动态边界：基础循环来自锁定的Ultralytics 8.4.84本地实现，置信度高；其他版本未外推。

## 5. 模块功能明细表
| 文件 | 核心符号 | 作用 | 证据 | 置信度 |
|---|---|---|---|---|
| `feature_hooks.py` | FeatureHookBank | batch级特征捕获和计数/清理 | 源码与pipeline测试 | 高 |
| `kd_trainer.py` | KDDetectionTrainer | Teacher隔离与复用训练循环 | 源码与11项回归 | 高 |
| `markov_relation_kd.py` | MultiStepTransitionKDLoss | MTRD转移/KL目标 | 30项单元测试 | 高 |
| `markov_kd_trainer.py` | MarkovKDDetectionTrainer | 损失替换及MTRD日志 | 继承关系与smoke | 高 |
| `markov_distill_train.py` | main | 独立CLI | inspect/smoke实测 | 高 |

## 6. 模块分类清单
### 6.1 通用基础模块
`feature_hooks.py`、`kd_trainer.py`。

### 6.2 项目定制候选模块
Gram/MTRD损失、MTRD trainer、训练/标定/诊断入口及相应测试。

### 6.3 不确定模块
未展开分析的Ultralytics动态注册与本任务无关模块。

## 7. 风险与边界说明
- P5梯度比触发WARNING；不自动改权重。
- 五batch余弦仅说明短程梯度差异，不代表精度收益。
- 权重与数据集未做静态反序列化分析；运行测试仅使用用户明确指定的可信本地权重。
- 没有使用git diff；使用SHA256。

## 8. 后续调用信息
- analysis_id：rca-20260731-mtrd8c4e84
- 已生成ProjectAnalysisPackage：是
