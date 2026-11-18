# Multi-scale Channel Gram Relation Distillation

简称：**Gram RelKD**。

## Teacher 与 Student

- Teacher：L1-96，权重 `runs/detect/runs/detect/hongkong_L1-96/weights/best.pt`。
- Student：SR1-P4Only，结构 `ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml`。
- 蒸馏特征：Layer 15/P3、Layer 18/P4、Layer 21/P5。
- 对应形状：`[B,64,80,80]`、`[B,96,40,40]`、`[B,256,20,20]`（imgsz=640）。

## 单尺度关系构造

给定特征 `F ∈ R^(B×C×H×W)`：

1. 将特征转换为 float32。
2. adaptive average pooling 到 20×20。
3. 展平空间维，得到 `X ∈ R^(B×C×N)`，其中 `N=400`。
4. 空间中心化：`X ← X - mean(X, dim=-1)`。
5. 沿空间维进行 L2 归一化：`X ← normalize(X, p=2, dim=-1)`。
6. 构造通道关系矩阵：`G = X X^T ∈ R^(B×C×C)`。
7. Teacher 的 `G_t` detach，Student 的 `G_s` 保留梯度。

单尺度损失：

`L_l = mean_B( ||G_s - G_t||_F² / C )`

三尺度等权：

`L_Gram = (L_P3 + L_P4 + L_P5) / 3`

总训练损失：

`L_total = L_detection + λ_Gram × ramp × batch_size × L_Gram`

正式参数：

- `λ_Gram = 0.0431658384`
- 10 epochs 线性 warmup
- pool size 20
- P3/P4/P5 layer weights = 1/1/1

代码证据位于 `ultralytics_ext/relation_kd.py`、`ultralytics_ext/feature_hooks.py`、`ultralytics_ext/kd_trainer.py` 和 `scripts/distill_train.py`。

## Teacher 隔离不变量

Teacher 始终 `eval()`、`requires_grad=False`，在 `no_grad` 下前向；不得进入 Student optimizer、EMA 或 checkpoint。Teacher 和 KD 损失代码只参与训练，推理时仅加载 SR1-P4Only Student。

因此相对同一 Student：额外推理 Params=0、额外 GFLOPs=0、额外 Teacher=0。Gram RelKD 不修改 Student 推理图；SR1-P4Only 本身才是结构变化来源。

