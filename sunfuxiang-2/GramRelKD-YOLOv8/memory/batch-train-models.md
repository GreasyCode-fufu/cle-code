---
name: batch-train-models
description: 批量训练多模型的脚本 run_models.sh 与改造后的 main.py 的位置和用法
metadata:
  type: project
---

批量训练多个 YOLO 模型（对比实验）用的脚本：
- `ultralytics-main/run_models.sh` —— 循环训练 11 个模型（yolo11/12/26/10/9/6/8 的 m/l），用 `.yaml` 架构、文件名带 m/l 后缀自动套用尺度。在 `ultralytics-main/` 下运行（见 [[two-ultralytics-packages]]）。默认输出到 `runs/detect/`（train/train2/...），每个模型日志在 `run_logs/<模型名>.log`，最后打印「权重位置(best.pt) -> 模型名」对照表。
- `ultralytics-main/main.py` —— 已改造为支持命令行参数：`python main.py <模型.yaml> <数据集.yaml> <cfg.yaml>`；`data` 显式传入会覆盖 cfg 里的 data 字段（get_cfg 中 overrides 优先级高于 cfg 文件）。不传参则保持原默认行为。

数据集：`ours/ultralytics/cfg/datasets/coco8.yaml`（番茄 jptomato，6类）；其它超参：`ours/ultralytics/cfg/default.yaml`（epochs=300, batch=32）。
