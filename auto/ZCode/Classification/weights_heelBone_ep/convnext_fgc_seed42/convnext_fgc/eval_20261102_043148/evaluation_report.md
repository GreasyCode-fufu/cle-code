# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_fgc_seed42/convnext_fgc/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8438 |
| Accuracy | 0.8449 |
| Macro-AUROC (OvR) | 0.9740 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8452 |
| Balanced Acc | 0.8703 |
| Macro-Sensitivity | 0.8703 |
| Macro-Specificity | 0.9247 |
| Macro-Precision | 0.8328 |
| AUC-PR (Macro) | 0.9604 |
| Cohen's Kappa | 0.7613 |
| MCC | 0.7670 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 940.1 (1.06 ms/张) |
| End-to-End FPS | 282.7 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8376 | 0.9167 | 0.8333 | 0.8354 | 0.9190 | 0.9623 | 0.9418 | 197 |
| normal | 0.7899 | 0.9464 | 0.9277 | 0.8532 | 0.8380 | 0.9664 | 0.9607 | 276 |
| sever | 0.9833 | 0.9112 | 0.7375 | 0.8429 | 0.9954 | 0.9934 | 0.9787 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=165 FP=33 TN=363 FN=32
- **normal**: TP=218 FP=17 TN=300 FN=58
- **sever**: TP=118 FP=42 TN=431 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 165 | 16 | 16 |
| normal | 32 | 218 | 26 |
| sever | 1 | 1 | 118 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.8333    0.8376    0.8354       197
      normal     0.9277    0.7899    0.8532       276
       sever     0.7375    0.9833    0.8429       120

    accuracy                         0.8449       593
   macro avg     0.8328    0.8703    0.8438       593
weighted avg     0.8578    0.8449    0.8452       593

```