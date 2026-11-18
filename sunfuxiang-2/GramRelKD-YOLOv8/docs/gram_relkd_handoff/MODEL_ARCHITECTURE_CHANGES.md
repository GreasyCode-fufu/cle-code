# 模型结构演进

## 第一层：原始 YOLOv8n

本项目记录的有效检测金字塔约为 P3=64、P4=128、P5=256；测试记录约 3,140,748 Params、8.3 GFLOPs、test mAP50-95≈0.429。这些是本仓库与本数据集的历史记录，不是对所有 YOLOv8n 实现的通用声明。

## 第二层：L1-96

L1-96 保持 P3=64 和 P5=256，将关键 P4 输出从 128 调整为 96：

- Backbone Layer 6：P4 输出 96。
- Head Layer 12：top-down P4 输出 96。
- Head Layer 18：bottom-up P4 输出 96。

最终外部金字塔为 P3/P4/P5=64/96/256。L1-96 的正式记录为 fused Params=2,641,337、GFLOPs≈7.2、test mAP50-95=0.429、mAP75=0.460。

## 第三层：SR1-P4Only

SR1-P4Only 保持 P3/P4/P5 外部输出为 64/96/256，只压缩两个 P4 C2f 的内部 hidden：

- Layer 12：hidden 48→40。
- Layer 18：hidden 48→40。
- Layer 21：P5 hidden 保持 128，不压缩。

因此 Teacher 和 Student 可直接比较同形状外部特征：

- Layer 15/P3：`[B,64,80,80]`
- Layer 18/P4：`[B,96,40,40]`
- Layer 21/P5：`[B,256,20,20]`

最终 Student：fused Params=2,603,129，unfused Params=2,607,913，GFLOPs=7.044096。

## 必须区分的两件事

SR1-P4Only 改变推理结构；Gram RelKD 只改变训练期监督，不改变推理结构。部署 checkpoint 不含 Teacher、Gram 损失或判别器。

