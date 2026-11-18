# Ultralytics 8.4.84 检测训练调用链

> 证据范围：`/data/cle/sunfuxiang-2/ultralytics-main` 当前本地源码。  
> 完成状态：部分完成（训练与 loss 调用链已完整确认；仓库缺少 `.git`，无法确认相对上游版本的 diff）。

## 1. 主调用链

```text
YOLO(...).train(...)
  └─ Model.train(trainer=None, **kwargs)
      ├─ YOLO.task_map["detect"]["trainer"] -> DetectionTrainer
      ├─ DetectionTrainer(overrides=args, _callbacks=...)
      └─ BaseTrainer.train()
          └─ BaseTrainer._do_train()
              ├─ BaseTrainer._setup_train()
              │   ├─ BaseTrainer.setup_model()
              │   ├─ DetectionTrainer.get_model()
              │   │   └─ DetectionModel(cfg, nc, ch)
              │   ├─ DetectionTrainer.get_dataloader()
              │   ├─ BaseTrainer.build_optimizer(self.model, ...)
              │   ├─ GradScaler(...)
              │   └─ ModelEMA(self.model)
              └─ 对每个 batch
                  ├─ DetectionTrainer.preprocess_batch(batch)
                  ├─ self.model(batch)
                  │   └─ BaseModel.forward(x=dict)
                  │       └─ BaseModel.loss(batch)
                  │           ├─ self.forward(batch["img"])
                  │           │   └─ BaseModel.predict()
                  │           │       └─ BaseModel._predict_once()
                  │           └─ v8DetectionLoss.__call__(preds, batch)
                  │               └─ v8DetectionLoss.loss()
                  │                   └─ loss * batch_size, loss.detach()
                  ├─ self.loss = loss.sum()
                  ├─ DDP: self.loss *= world_size
                  ├─ scaler.scale(self.loss).backward()
                  └─ optimizer_step()
                      ├─ scaler.unscale_(optimizer)
                      ├─ clip_grad_norm_
                      ├─ scaler.step(optimizer)
                      ├─ scaler.update()
                      ├─ optimizer.zero_grad()
                      └─ ema.update(self.model)
```

## 2. 精确符号与行号

| 阶段 | 文件与行号 | 类 / 函数签名 | 当前源码行为 |
| --- | --- | --- | --- |
| Python 训练入口 | `ultralytics/engine/model.py:722-726` | `Model.train(self, trainer=None, **kwargs)` | `YOLO` 继承该方法，没有检测专用 `train` 覆盖。 |
| Trainer 映射 | `ultralytics/models/yolo/model.py:87-102` | `YOLO.task_map` | detect 任务映射到 `DetectionModel`、`DetectionTrainer`、`DetectionValidator`。 |
| Trainer 创建 | `ultralytics/engine/model.py:770-811` | `Model.train(...)` | 合并 cfg/kwargs，在 805 行实例化 trainer；仅当 `self.ckpt` 存在时，806-811 行才向新模型传权重。 |
| 检测 Trainer | `ultralytics/models/yolo/detect/train.py:24,55-63` | `DetectionTrainer(BaseTrainer).__init__(...)` | 直接调用 `BaseTrainer.__init__`。 |
| 模型构建 | `ultralytics/models/yolo/detect/train.py:182-196` | `DetectionTrainer.get_model(cfg=None, weights=None, verbose=True)` | 先构造 `DetectionModel`，只有 `weights` 非空才执行 `model.load(weights)`。 |
| 训练循环 | `ultralytics/engine/trainer.py:390-540` | `BaseTrainer._do_train(self)` | 执行 setup、epoch/batch、前向、反向、优化、日志及回调。 |
| batch 预处理 | `ultralytics/models/yolo/detect/train.py:107-137` | `DetectionTrainer.preprocess_batch(self, batch)` | 所有 Tensor 搬到 device；`img.float()/255`；可选 multi-scale。 |
| 前向与 loss | `ultralytics/engine/trainer.py:455-473` | `_do_train` 内部 | 457-470 行处于 autocast；464 行 `loss, loss_items = self.model(batch)`；465 行 `loss.sum()`；473 行 GradScaler backward。 |
| 通用 forward | `ultralytics/nn/tasks.py:149-164` | `BaseModel.forward(self, x, *args, **kwargs)` | dict 输入调用 `self.loss`；Tensor 输入调用 `self.predict`。`DetectionModel` 继承此实现。 |
| 单次网络前向 | `ultralytics/nn/tasks.py:183-211` | `BaseModel._predict_once(...)` | 逐层执行 `x = m(x)`；Layer 15/18/21 hook 正是在 203 行调用时触发。 |
| 通用 loss | `ultralytics/nn/tasks.py:345-357` | `BaseModel.loss(self, batch, preds=None)` | 首次延迟创建 criterion；未提供 preds 时执行 `self.forward(batch["img"])`；随后调用 criterion。 |
| criterion 选择 | `ultralytics/nn/tasks.py:538-540` | `DetectionModel.init_criterion(self)` | 普通检测返回 `v8DetectionLoss(self)`。 |
| 检测 loss 入口 | `ultralytics/utils/loss.py:470-476` | `v8DetectionLoss.__call__(self, preds, batch)` | 解析预测输出后调用 `self.loss(...)`。 |
| 检测 loss 返回 | `ultralytics/utils/loss.py:478-482` | `v8DetectionLoss.loss(self, preds, batch)` | 返回 `loss * batch_size, loss_detach`。 |
| optimizer 构建调用 | `ultralytics/engine/trainer.py:282-292` | `_build_train_pipeline` 内部 | 以 `self.model` 作为 `build_optimizer` 的唯一模型参数来源。 |
| optimizer 参数分组 | `ultralytics/engine/trainer.py:1043-1127` | `BaseTrainer.build_optimizer(self, model, ...)` | 遍历 `unwrap_model(model).named_modules()` 和每个模块的 `named_parameters(recurse=False)`，按 weight/BN/bias/Muon 分组。 |
| GradScaler 创建 | `ultralytics/engine/trainer.py:338-350` | `_setup_train` 内部 | AMP 检查后创建 CUDA GradScaler。 |
| optimizer/EMA 更新 | `ultralytics/engine/trainer.py:786-794` | `BaseTrainer.optimizer_step(self)` | unscale、裁剪、step、scaler update、清梯度，最后更新 EMA。 |
| EMA 创建 | `ultralytics/engine/trainer.py:373-377` | `_setup_train` 内部 | 完成训练 pipeline 后执行 `ModelEMA(self.model)`。 |
| checkpoint 保存 | `ultralytics/engine/trainer.py:661-706` | `BaseTrainer.save_model(self)` | 保存 EMA、optimizer、scaler、train_args、metrics 等。 |

## 3. loss 返回与缩放规则

- `v8DetectionLoss.get_assigned_targets_and_loss()` 构造三项 loss：box、cls、dfl，并分别乘 `hyp.box`、`hyp.cls`、`hyp.dfl`（`loss.py:404-462`）。
- `v8DetectionLoss.loss()` 返回：
  - `loss`: `Tensor[3]`，已在 `loss.py:482` 乘当前 batch size；
  - `loss_items`: `Tensor[3]`，是未乘 batch size的 detach 值。
- 训练循环在 `trainer.py:465` 对 `loss` 做 `.sum()`，没有再次乘 batch size。
- DDP 时在 `trainer.py:466-467` 再把总标量乘 `world_size`，用于抵消 DDP 梯度平均。
- 单 batch 实测：
  - `loss.shape == [3]`，值为 `[38.26755, 30.25657, 37.56116]`；
  - `loss_items.shape == [3]`，值为 `[1.195861, 0.945518, 1.173786]`；
  - 三项满足 `loss ≈ loss_items × 32`；
  - trainer 将反向传播标量设为 `loss.sum()`，本次诊断未执行 backward。

## 4. AMP、GradScaler、DDP 和 EMA

- autocast 的实际范围是 `trainer.py:457-470`：包含 `preprocess_batch`、模型前向、criterion、loss sum、DDP 缩放和 loss 日志累计。
- backward 位于 autocast 块外，但使用 `self.scaler.scale(self.loss).backward()`。
- optimizer step 使用 GradScaler 的 `unscale_ → step → update`。
- DDP 在 `trainer.py:362-367` 包装 `self.model`；当前源码在非 compile 模式设置 `find_unused_parameters=True`。
- EMA 在 optimizer step 成功后更新，验证和 checkpoint 使用 EMA 路径。

## 5. callback 可否插入 KD

不能安全地仅靠现有 callback 修改反向传播前的总 loss：

- `on_train_batch_start` 位于 `trainer.py:436`，发生在预处理和前向之前；
- 下一次可用的 batch 回调是 `on_batch_end`（526）和 `on_train_batch_end`（530），都发生在 backward/optimizer step 之后；
- 457-473 的“前向 → loss.sum → backward”之间没有 callback 事件。

因此，方案 E 不满足“在 backward 前稳定地把 KD loss 加到总 loss”的要求。

## 6. 当前本地蒸馏相关代码（既有，不是本次新增）

- `ultralytics/cfg/default.yaml:101-102` 已存在 `distill_model` 与 `dis`。
- `ultralytics/engine/trainer.py:30,301-377` 已导入并包装 `DistillationModel`。
- `ultralytics/nn/distill_model.py:33-307` 已实现 hook、Teacher 前向、投影器和 score-weighted L2 loss。
- 当前实现与本任务目标不一致：
  - 它使用两层 1×1 Conv projector，而本任务要求同通道关系 Gram；
  - wrapper 在 optimizer 构建前成为 `self.model`，因此冻结 Teacher 参数对象仍会被 optimizer 参数遍历收集；
  - projector 进入 optimizer 和 EMA；
  - 最终 `strip_optimizer` 会剥离 wrapper，仅保留 Student，但训练期对象隔离仍不满足本任务的严格目标。

当前任务没有调用或修改该既有蒸馏路径。
