# Gram RelKD 复现与核验

实际副本根目录：`/data/cle/sunfuxiang-2/GramRelKD-YOLOv8`。Python：`/home/login/miniconda3/envs/yolo/bin/python`。数据集仍位于 `/data/cle/sunfuxiang-2/datasets/hongkongScale`。

## 重要路径兼容说明

历史 `scripts/distill_train.py`、`scripts/test_relkd.py` 和多份分析脚本含源项目绝对路径。为保持证据链，它们未被修改。副本新增：

- `scripts/distill_train_clone.py`：与正式入口逻辑相同，仅将 `ROOT` 改为由脚本位置解析。
- `scripts/run_gram_relkd_from_clone.sh`：需要显式 `--train` 才会启动训练。
- `scripts/test_relkd_clone.py`：副本路径的 test split 入口。
- `scripts/clone_smoke_test.py`：无训练随机前向和隔离检查。

完整硬编码路径清单见 `HARD_CODED_SOURCE_PATHS.txt`。

## 正式训练命令（仅记录，本次未执行）

```bash
cd /data/cle/sunfuxiang-2/GramRelKD-YOLOv8
bash scripts/run_gram_relkd_from_clone.sh --train
```

该 launcher 使用 YAML 随机初始化 Student，固定 `kd_weight=0.0431658384`、epochs=300、batch=32、imgsz=640、workers=8、device=0、seed=0、deterministic、AMP、optimizer=auto、patience=0、mosaic=1、mixup=0、close_mosaic=10、10-epoch KD warmup、pool=20、P3/P4/P5 等权。不要加载历史 Student、Teacher 到 Student 或通用 yolov8n.pt。

## 模型加载与随机前向

```bash
cd /data/cle/sunfuxiang-2/GramRelKD-YOLOv8
/home/login/miniconda3/envs/yolo/bin/python scripts/clone_smoke_test.py
```

## 独立 test split（不要用于边看 test 边调参）

```bash
cd /data/cle/sunfuxiang-2/GramRelKD-YOLOv8
/home/login/miniconda3/envs/yolo/bin/python scripts/test_relkd_clone.py
```

## 参数量和 GFLOPs

`clone_smoke_test.py` 同时统计 checkpoint 的 unfused Params、深拷贝 fuse 后的 Params，以及 `get_flops(..., imgsz=640)`。

## 单 batch 特征形状

烟雾脚本在 15/18/21 层注册只读 hook，并以 `torch.randn(1,3,640,640)` 做一次 `eval/no_grad` 前向，预期得到 `[1,64,80,80]`、`[1,96,40,40]`、`[1,256,20,20]`。

## Teacher/checkpoint 隔离

检查 checkpoint 顶层键、state_dict 键、模块类名中是否出现 teacher、KD、distill、MTRD、GAN/discriminator 等标识；确认 `ultralytics_ext` 在独立加载与前向前后都不在 `sys.modules`。正式证据另见 `runs/analysis/kd_formal_result/model_integrity_report.json`。

