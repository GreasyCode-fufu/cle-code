# Ultralytics 8.4.84 知识蒸馏第一阶段检查报告

> 项目：`/data/cle/sunfuxiang-2/ultralytics-main`  
> 分析日期：2026-07-31（Asia/Shanghai）  
> 完成状态：部分完成  
> 整体置信度：高（模型、batch、loss 和调用链有源码/运行证据）；Git 修改来源与历史脚本快照为 UNKNOWN。

## A. Teacher 路径与结构

- 权重：`/data/cle/sunfuxiang-2/ultralytics-main/runs/detect/runs/detect/hongkong_L1-96/weights/best.pt`
- YAML：`/data/cle/sunfuxiang-2/ultralytics-main/ultralytics/cfg/models/hongkong/L1-96.yaml`
- checkpoint 保存了模型 YAML，核心 `nc/scales/backbone/head` 与磁盘 YAML 完全一致。
- 类：`DetectionModel`；层容器：`DetectionModel.model`（实际访问为 `teacher.model[15]` 等）。
- 精确参数量：2,646,185；640 输入 GFLOPs：7.2706048。
- Layer 12：`C2f`，输出 96，hidden 48。
- Layer 18：`C2f`，输出 96，hidden 48。
- Layer 21：`C2f`，输出 256，hidden 128。
- checkpoint best mAP50-95：0.42737。

## B. Student YAML、best.pt 与结构

- 身份：SR1-P4Only。
- YAML：`/data/cle/sunfuxiang-2/ultralytics-main/ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml`
- best.pt：`/data/cle/sunfuxiang-2/ultralytics-main/runs/detect/train-9/weights/best.pt`
- args：`/data/cle/sunfuxiang-2/ultralytics-main/runs/detect/train-9/args.yaml`
- checkpoint 保存了模型 YAML；核心内容与磁盘 YAML 完全一致。
- 所有现有 `runs/detect/**/weights/best.pt` 中，只有 train-9 同时满足：
  - Layer 12：`C2f`，输出 96，hidden 40；
  - Layer 18：`C2f`，输出 96，hidden 40；
  - Layer 21：`C2f`，输出 256，hidden 128。
- 精确参数量：2,607,913；640 输入 GFLOPs：7.1477248。
- checkpoint best mAP50-95：0.42166（第 287 epoch 行）；最后 epoch 为 0.41943。
- 与用户先验存在差异：参数量不是 2,603,129，best mAP 也不是约 0.427。结构和内嵌 YAML 提供了唯一身份确认，因此仍确认 train-9 是 SR1-P4Only；差异不能静默抹平。

## C. 训练入口与完整训练参数

当前 Python 入口：

`/data/cle/sunfuxiang-2/ultralytics-main/main.py`

当前内容等价于：

```python
model = YOLO("/data/cle/sunfuxiang-2/ultralytics-main/ultralytics/cfg/models/hongkong/C2f_DSM.yaml")
model.train(cfg="ultralytics/cfg/default.yaml")
```

另有 launcher：

`/data/cle/sunfuxiang-2/ultralytics-main/train_hongkong.sh`

它声明通过 `main.py --model ... --name ...` 训练 L1-B/L1-N/L1-96/C2f_SAPA，但当前 `main.py` 已不再解析这些参数，因此历史时点的 `main.py` 内容为 UNKNOWN。SR1、C2f_SPP、C2f_DSM 的 args 时间与顺序符合反复编辑 `main.py` 模型路径后调用相同 cfg 的模式，但历史源码快照同样不存在。

实际运行参数以每个 run 的 `args.yaml` 为准，完整字典见 `kd_training_args.json`。四个目标 run 的关键参数一致：

| 参数 | 实际值 |
| --- | --- |
| data | 解析后为 `/data/cle/sunfuxiang-2/ultralytics-main/ultralytics/cfg/datasets/coco8.yaml` |
| 数据目录 | `/data/cle/sunfuxiang-2/datasets/hongkongScale` |
| epochs | 300 |
| batch | 32 |
| imgsz | 640 |
| optimizer | `auto` |
| lr0 / lrf | 0.01 / 0.01 |
| momentum / weight_decay | 0.937 / 0.0005 |
| warmup_epochs | 3.0 |
| pretrained | true |
| seed / deterministic | 0 / true |
| AMP | true |
| mosaic / mixup / close_mosaic | 1.0 / 0.0 / 10 |
| patience | 0 |
| workers | 8 |
| device | null，运行时自动选择 CUDA:0 |
| resume | false |
| split | test |

`optimizer=auto` 的有效行为不能只看 args 字面值：

- 数据集 3315 张，`nbs=64`，batch=32，accumulate=2；
- `ceil(3315/max(32,64))*300 = 15600 > 10000`；
- `trainer.py:1060-1069` 因此选择 `MuSGD(lr=0.01, momentum=0.9)`，配置的 momentum=0.937 被忽略；
- 有效 weight decay 仍为 `0.0005 × 32 × 2 / 64 = 0.0005`。

初始化策略为 A：从 YAML 随机初始化。证据：

- train-9 的 `model` 是 YAML，`resume=false`；
- `Model._new` 直接构建 `DetectionModel`；
- `Model.train` 仅在 `self.ckpt` 存在时向 trainer 传已加载权重；
- `pretrained=true` 只是布尔值，不是 yolov8n.pt/L1-96 checkpoint 路径。

因此 baseline 没有 `Transferred x/y items`；历史控制台日志不存在，精确日志文本为 UNKNOWN。公平复现 KD 时应继续用同一 Student YAML 随机初始化、seed=0 和完全相同超参数，不应从 train-9 best.pt 继续训练，也不应加载 Teacher 权重。

## D. 本地源码修改状态

`git status`、`git diff --stat` 和 `git diff` 均无法执行：项目根目录没有 `.git`，Git 返回“不是 git 仓库”。Teacher/Student checkpoint 的 Git 元数据也均为 `root=None, branch=None, commit=None`。因此相对上游 Ultralytics 8.4.84 的精确 diff 为 UNKNOWN。

当前源码中可直接确认的项目定制候选模块：

- `C2f_SAPA`：`ultralytics/nn/modules/C2f_SAPA.py:608`
- `C2f_SPP`：`ultralytics/nn/modules/block.py:2222`
- `C2f_DSM`：`ultralytics/nn/modules/block.py:2561`
- 其他非标准导入：`iRMB`、`C2f_FocalModulation`、`ADownRes`、`StableADownRes`、`DFD_C2f`、`SPDownConv`
- 既有蒸馏包装：`ultralytics/nn/distill_model.py`

`tasks.py` 的可见定制点：

- 12-20 行导入上述非标准模块；
- 1824-1868 行把 `C2f_SAPA/C2f_SPP/C2f_DSM` 加入 `base_modules`；
- 1870-1892 行把三者加入 `repeat_modules`；
- `parse_model` 随后按标准 C2f 类路径插入 repeat 数并完成通道缩放。

`modules/__init__.py` 导入并导出 `C2f_SPP/C2f_DSM`。本次未撤销、格式化或覆盖任何既有改动。

额外发现：`default.yaml`、`trainer.py`、`torch_utils.py` 已存在一套 `DistillationModel` 相关逻辑。这些代码在本次工作开始前已经存在，本次没有启用或修改。

按静态分析契约分类：

- 通用基础模块：`ultralytics/engine/model.py`、`engine/trainer.py`、`models/yolo/detect/train.py`、`nn/tasks.py` 的基础模型/解析框架、`utils/loss.py`、`data/`。
- 项目定制候选模块：上述 C2f/下采样模块、`nn/distill_model.py`、Hong Kong YAML、`main.py`、`train_hongkong.sh`。
- 不确定模块：缺少 Git 基线时，`trainer.py`、`torch_utils.py`、`default.yaml` 中无法仅凭静态文本判断来源的其他改动。

## E. 训练调用链

详见 `kd_source_call_chain.md`。摘要：

```text
Model.train
→ YOLO.task_map["detect"]["trainer"]
→ DetectionTrainer
→ BaseTrainer._do_train
→ DetectionTrainer.preprocess_batch
→ self.model(batch)
→ BaseModel.forward(dict)
→ BaseModel.loss
→ self.forward(batch["img"])
→ v8DetectionLoss.__call__
→ v8DetectionLoss.loss
→ loss, loss_items
→ loss.sum()
→ GradScaler.backward
→ optimizer_step
→ EMA.update
```

## F. 检测 loss 返回与缩放

- `loss`：`Tensor[3]`，顺序 box/cls/dfl；criterion 已乘 batch size。
- `loss_items`：detach 的 `Tensor[3]`，没有乘 batch size。
- trainer 对 `loss` 执行 `.sum()`；不再乘 batch size。
- DDP 额外乘 `world_size`。
- 本次 batch=32 实测：
  - `loss = [38.26755, 30.25657, 37.56116]`
  - `loss_items = [1.195861, 0.945518, 1.173786]`
  - `loss.sum() = 106.08529`

## G. AMP、optimizer 与 EMA

- autocast 覆盖预处理、Student 前向、criterion、loss sum 和 DDP 缩放。
- backward 位于 autocast 外，由 GradScaler 执行。
- optimizer 从 `unwrap_model(self.model)` 的全部模块参数分组构建。
- EMA 在 optimizer step 后更新。
- 推荐实现必须让 `self.model` 始终只是 Student；Teacher 应作为 trainer 的普通运行期属性，且在 Student optimizer 和 EMA 创建完成后加载。

现有 `DistillationModel` 不适合直接复用：

- wrapper 在 optimizer 构建前成为 `self.model`；
- Teacher 虽冻结，但仍会被 optimizer 参数遍历收集；
- projector 进入 optimizer 和 EMA；
- 使用 score-weighted L2，不是本任务的无投影通道关系 Gram。

## H. Teacher/Student 单 batch 结果

环境：Python 3.12.13、torch 2.4.1+cu121、Ultralytics 8.4.84、CUDA:0 RTX 4090。

| 层 | Teacher | Student | 一致 |
| --- | --- | --- | --- |
| 15 / P3 | `[32,64,80,80]`, fp16, requires_grad=False | `[32,64,80,80]`, fp16, requires_grad=True | 是 |
| 18 / P4 | `[32,96,40,40]`, fp16, requires_grad=False | `[32,96,40,40]`, fp16, requires_grad=True | 是 |
| 21 / P5 | `[32,256,20,20]`, fp16, requires_grad=False | `[32,256,20,20]`, fp16, requires_grad=True | 是 |

Teacher 使用 `torch.no_grad()`；Student 通过 `student(batch)` 的训练 loss 字典路径执行。没有 backward、optimizer step 或 checkpoint 写入。参数 SHA256 前后一致。

## I. hooks 可行性

- 真实容器路径：加载后的 `DetectionModel.model[index]`；在 YOLO wrapper 表达中是 `yolo.model.model[index]`。
- Teacher 纯预测 `teacher(batch["img"])`：三层 hook 各触发 1 次。
- Student loss 路径 `student(batch)`：三层 hook 各触发 1 次。
- 三层输出类型全部为 `Tensor`，不是 tuple/list。
- 每个 batch 前必须清空缓存；batch 后也应删除对 feature 的引用。
- Student feature 持有 autograd graph；若字典跨 batch 保留，会造成计算图和显存长期引用。
- 正确清理：保存 `handle = module.register_forward_hook(...)`；在 `finally` 中逐个 `handle.remove()`，并 `cache.clear()`。长期训练时 hook 可注册一次，但缓存必须每 batch 清空。

## J. batch 字典与 GT 前景掩码条件

真实 preprocess 后 keys：

```text
batch_idx, bboxes, cls, im_file, img, ori_shape, ratio_pad, resized_shape
```

实测：

- `img`: `[32,3,640,640]`，float32，CUDA:0，范围 `[0,1]`
- `batch_idx`: `[248]`
- `cls`: `[248,1]`
- `bboxes`: `[248,4]`
- bbox：`xywh`，归一化，实测范围 `[0.003157, 0.991779]`
- `ori_shape` 样例：`[768,1024]`
- `resized_shape` 样例：`[640,640]`
- `ratio_pad` 样例：`[0.625,0.625]`

源码证据：`dataset.py:230-241` 最后追加 `Format(bbox_format="xywh", normalize=True)`；LetterBox 在 `augment.py:1846-1849` 先缩放并加 padding；`Format` 在 2348-2350 行按最终图像宽高归一化。因此 batch 中 bbox 对应增强/LetterBox 后的 640×640 输入，而不是原图坐标。

映射到任意特征图 `(Hf,Wf)`：

1. 对每个 `xywh` 归一化框计算 `x1=(cx-w/2)Wf`、`x2=(cx+w/2)Wf`；
2. `y1=(cy-h/2)Hf`、`y2=(cy+h/2)Hf`；
3. clamp 到合法范围，左上取 floor，右下取 ceil；
4. 依据 `batch_idx` 在 `[B,1,Hf,Wf]` mask 对应区域置 1；
5. 分别使用 `(80,80)`、`(40,40)`、`(20,20)` 生成 P3/P4/P5 mask。

已有信息足够；由于 bbox 已映射到最终输入，无需再次应用 `ratio_pad`。

## K. 推荐的蒸馏插入方案

推荐“方案 D 作为独立入口 + 方案 C 作为内部机制”：

- 新建独立 `distill_train.py`；
- 其中使用自定义 `KDDetectionTrainer(DetectionTrainer)`；
- 不改 `ultralytics/**` 核心文件；
- `self.model` 只保留标准 Student；
- 先让父类创建 Student optimizer/EMA，再把冻结 Teacher 放在 trainer 属性中；
- 在自定义训练 batch 段执行 Teacher no-grad 前向、Student 标准 loss、FP32 通道 Gram KD，并在 backward 前相加；
- validation/test 继续调用标准 Student validator；
- checkpoint/EMA/optimizer 都只见到 Student。

当前 8.4.84 在“前向完成”和“backward”之间没有可覆盖的小型 loss hook，因此自定义 trainer 需要对 `_do_train` 的 batch 核心段做版本锁定的最小复制。应配套回归测试，防止以后升级 Ultralytics 时静默漂移。

原检测 loss 是 `loss_vec`（已乘 B）。后续 KD 若定义为 batch/scale 平均的 `kd_mean`，加入总反向标量时应显式使用 `lambda_kd * kd_mean * B`，以与检测 loss 的 batch 缩放一致；日志项可保留未乘 B 的 detach 值。`lambda_kd` 本阶段不决定。

## L. 不推荐方案与原因

- A / 改 `utils/loss.py`：把 Teacher 生命周期和关系 loss 混入标准检测准则，污染 validation，并永久改核心。
- B / 改 `DetectionModel.loss`：若 Teacher 注册为子模块，会进入 optimizer、EMA、checkpoint；若用非注册属性规避，又带来序列化和生命周期脆弱性。
- 仅 C 且直接改核心 trainer：能实现，但违反“不永久修改核心训练循环”。
- 当前既有 `DistillationModel`：optimizer/EMA 包含范围和 loss 形式均不符合本任务要求。
- E / callback：现有 callback 时点在前向前或 optimizer 后，无法安全修改 backward 前总 loss。

## M. 关系 Gram 的显存与数值建议

batch=32 时，单个 `[B,C,C]` Gram 的纯矩阵存储：

| 层 | 形状 | FP16 | FP32 |
| --- | --- | ---: | ---: |
| P3 | `[32,64,64]` | 0.250 MiB | 0.500 MiB |
| P4 | `[32,96,96]` | 0.5625 MiB | 1.125 MiB |
| P5 | `[32,256,256]` | 4.000 MiB | 8.000 MiB |
| 合计/单模型 |  | 4.8125 MiB | 9.625 MiB |
| Teacher+Student |  | 9.625 MiB | 19.250 MiB |

以上不含 feature、归一化临时量、矩阵乘 workspace 和 Student autograd 保存。作为对照，P3 空间 Gram `[32,6400,6400]` 单模型 FP32 约 4.883 GiB，禁止构造是正确的。

建议：

- 可先对每个通道在空间维中心化，再做 L2 normalization；这样 Gram 表达通道相关性而不是能量大小。
- 即使模型处于 AMP，也在 `autocast(enabled=False)` 中把 feature 转 float32 计算 Gram。
- 使用 batched `bmm` 保留逐样本 `[B,C,C]`，不要先跨 batch 聚合。
- trace 归一化可用 `G / (trace(G)+1e-6)`；在 FP32 中计算。
- P5 的输出矩阵只有 8 MiB/模型，不因显存强制降采样；若运行时间过高，可对 Teacher/Student 使用相同空间采样索引或 pooling。P3/P5 的 bmm 计算量都不小，不应只凭 C 判断。
- Teacher feature 在 `no_grad` 中获得，并在计算 Gram 前显式 `.detach()`；Teacher Gram 也保持 detach。
- 若峰值显存仍高，可按 batch 维分块逐样本/小块计算并累计 loss。

## N. 后续文件清单与未解决问题

建议后续新增：

- `scripts/distill_train.py`
- `ultralytics_ext/kd_trainer.py`
- `ultralytics_ext/relation_kd.py`
- `ultralytics_ext/feature_hooks.py`
- `tests/test_kd_pipeline.py`

预计需要修改的 Ultralytics 核心文件：无。

本阶段实际新增：

- `scripts/inspect_kd_pipeline.py`
- `runs/analysis/kd_code_inspection/` 下五份指定输出
- `.research-code-analysis/kd_code_inspection/ProjectAnalysisPackage.json`

UNKNOWN / 待解决：

1. 无 `.git`，无法给出相对上游 8.4.84 的可信 diff 或 commit。
2. 历史 `main.py` 快照不存在，无法证明每个 run 当时的完整脚本文本；完整有效参数已由各自 `args.yaml` 保存。
3. 用户先验参数量/mAP 与唯一结构匹配的 train-9 checkpoint 不一致。
4. 历史控制台没有保存 `Transferred x/y items`；源码路径证明 baseline 无预训练迁移，但不能复原当时日志原文。
5. KD 权重、各层权重、归一化细节与评估协议尚未决定；按约束本阶段不自动选择。

## O. 安全性与执行结论

- 未修改 `loss.py`、`tasks.py`、`trainer.py` 或任何核心训练代码。
- 未修改 Teacher/Student checkpoint。
- 未启动正式训练。
- 单 batch 仅执行 Teacher no-grad 前向和 Student 标准 loss 前向。
- 未执行 backward 或 optimizer step。
- Teacher/Student 参数哈希在诊断前后完全一致。
- 运行状态：SUCCESS。
