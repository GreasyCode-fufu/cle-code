# SR1-P4Only RelKD 严格配对 NoKD 对照报告

## 运行结论

两项训练与独立 test split 评估均成功完成。配对核验通过：Student 初始参数哈希完全一致，seed、数据配置、Student YAML、训练调用链和全部训练超参数一致。除用于隔离输出的 `name`/`save_dir` 外，唯一优化变量是 `kd_weight`：NoKD 为 0，RelKD 为 0.0431658384。

NoKD 的 300 条 `kd_metrics.csv` 记录中，`kd_weight_base`、`kd_scaled_mean` 和 `kd_to_det_ratio` 均始终为 0。因此虽然保留了 Teacher 前向与关系损失统计，KD 项没有影响 Student 梯度。

## 独立测试集结果

| 指标 | Paired NoKD | RelKD | RelKD - NoKD |
|---|---:|---:|---:|
| Precision | 0.76923 | 0.74985 | -0.01938 |
| Recall | 0.66821 | 0.67544 | +0.00722 |
| mAP50 | 0.69747 | 0.70447 | +0.00700 |
| mAP75 | 0.43523 | 0.45963 | +0.02440 |
| mAP50-95 | 0.42120 | 0.43568 | +0.01448 |
| spalling AP50-95 | 0.56010 | 0.58200 | +0.02190 |
| crack AP50-95 | 0.41213 | 0.41068 | -0.00144 |
| moisture AP50-95 | 0.29136 | 0.31436 | +0.02299 |

RelKD 相比严格配对 NoKD 的 test mAP50-95 提高 0.01448（1.448 个百分点），mAP75 提高 0.02440（2.440 个百分点）。主要收益来自 spalling 和 moisture；crack AP50-95 基本持平并轻微下降 0.00144。RelKD 的召回率提高 0.00722，但精确率下降 0.01938。

## 训练与模型复杂度

| 项目 | Paired NoKD | RelKD | 差值 |
|---|---:|---:|---:|
| best epoch | 287 | 300 | +13 |
| `results.csv` 训练时间 | 3489.85 s | 3504.71 s | +14.86 s |
| 启动器总耗时 | 3499.04 s | 3513.83 s | +14.79 s |
| fused Params | 2,603,129 | 2,603,129 | 0 |
| GFLOPs | 7.044096 | 7.044096 | 0 |

RelKD 的训练时间比 NoKD 增加约 0.426%。由于两项实验均保留 Teacher 前向和关系损失计算，这个差值主要反映KD反向传播带来的边际开销及正常计时波动，而不是完整 Teacher 监督管线相对于纯检测训练的总开销。

## 严格配对与完整性证据

- 两次 Student 随机初始化哈希均为 `b5c987921cb2eae9d9536c8c632c619145e7d56a992f37d9badff8a368ae25b6`。
- 两个 `args.yaml` 仅 `name` 和派生 `save_dir` 不同。
- Student YAML、数据 YAML、`distill_train.py`、`KDDetectionTrainer` 源码哈希一致。
- NoKD 日志未发现 `train-9/weights/best.pt`、`yolov8n.pt`、`C2f_SPP`、`C2f_DSM` 或非零 transferred-weight 载入迹象。
- Teacher 参数在两次训练前后哈希均一致，且未进入 optimizer、EMA 或 Student checkpoint。
- 两个 best.pt 均为 fused Params 2,603,129、GFLOPs 7.044096；推理结构复杂度没有增加。
- 两次评估均调用相同 default cfg 和 test split，仅权重路径不同。

## 判断

严格配对结果支持 RelKD 产生了真实、明显的总体检测收益，而不是由初始化、seed、数据或训练超参数差异造成。收益集中在更严格的定位质量（mAP75）以及 spalling、moisture 两类；代价是整体 Precision 降低，且 crack 未获得收益。
