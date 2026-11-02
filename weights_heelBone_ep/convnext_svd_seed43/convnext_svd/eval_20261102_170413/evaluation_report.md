# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_svd_seed43/convnext_svd/model_best.pth`
- 测试集: `./datasets/heelBone/test`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8542 |
| Accuracy | 0.8449 |
| Macro-AUROC (OvR) | 0.9731 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8428 |
| Balanced Acc | 0.8816 |
| Macro-Sensitivity | 0.8816 |
| Macro-Specificity | 0.9240 |
| Macro-Precision | 0.8484 |
| AUC-PR (Macro) | 0.9607 |
| Cohen's Kappa | 0.7623 |
| MCC | 0.7744 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 1806.8 (0.55 ms/张) |
| End-to-End FPS | 310.3 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.9239 | 0.8510 | 0.7552 | 0.8311 | 0.9574 | 0.9590 | 0.9375 | 197 |
| normal | 0.7210 | 0.9716 | 0.9567 | 0.8223 | 0.8000 | 0.9618 | 0.9508 | 276 |
| sever | 1.0000 | 0.9493 | 0.8333 | 0.9091 | 1.0000 | 0.9984 | 0.9939 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=182 FP=59 TN=337 FN=15
- **normal**: TP=199 FP=9 TN=308 FN=77
- **sever**: TP=120 FP=24 TN=449 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 182 | 9 | 6 |
| normal | 59 | 199 | 18 |
| sever | 0 | 0 | 120 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7552    0.9239    0.8311       197
      normal     0.9567    0.7210    0.8223       276
       sever     0.8333    1.0000    0.9091       120

    accuracy                         0.8449       593
   macro avg     0.8484    0.8816    0.8542       593
weighted avg     0.8648    0.8449    0.8428       593

```