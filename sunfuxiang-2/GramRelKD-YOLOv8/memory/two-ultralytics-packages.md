---
name: two-ultralytics-packages
description: 项目里有两套 ultralytics 包；ours 的太旧不支持 yolo11/12/26，训练新模型必须用 ultralytics-main
metadata:
  type: project
---

项目 `/home/login/cle/sunfuxiang-1` 下有两套 ultralytics 包：
- `ours/ultralytics` —— 较旧的 fork（yolov8 时代，自定义了 spd_CPCA.yaml）。`parse_model` 没有 `C3k2` 等模块，**无法**构建 yolo11/12/26。`guess_model_scale` 正则 `yolov\d+([nslmx])` 要求带 `v`，对 `yolo11`(无v) 解析不出 scale。
- `ultralytics-main/ultralytics` —— 新版，支持 yolo11/12/26，scale 解析正常。

**训练 yolo11/12/26/10/9/6/8 等模型时必须用 ultralytics-main 包**：在 `ultralytics-main/` 目录下运行脚本（让 `import ultralytics` 解析到它），或设 `PYTHONPATH=/home/login/cle/sunfuxiang-1/ultralytics-main`。

**Why:** 从目录结构看不出哪个能用，要实际构建模型才会报 `KeyError: 'C3k2'`。
**How to apply:** 用户的批量训练脚本 [[batch-train-models]] 在 `ultralytics-main/` 下运行 `main.py`。ours/ 里保留了番茄数据集配置(coco8.yaml 含 jptomato)和 default.yaml，但 default.yaml 的 `data:` 字段指向的 `ours/cfg/datasets/coco8.yaml` 不存在，需显式传 data 覆盖。
