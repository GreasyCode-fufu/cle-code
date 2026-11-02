# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_heelBone_ep/convnext_svd_seed43/convnext_svd/model_best.pth`
- 测试集: `./datasets/heelBone/val`  (共 593 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8593 |
| Accuracy | 0.8499 |
| Macro-AUROC (OvR) | 0.9737 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8492 |
| Balanced Acc | 0.8788 |
| Macro-Sensitivity | 0.8788 |
| Macro-Specificity | 0.9243 |
| Macro-Precision | 0.8500 |
| AUC-PR (Macro) | 0.9627 |
| Cohen's Kappa | 0.7679 |
| MCC | 0.7734 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 1807.0 (0.55 ms/张) |
| End-to-End FPS | 278.7 |
| 测试样本数 | 593 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| heelspur | 0.8731 | 0.8687 | 0.7679 | 0.8171 | 0.9322 | 0.9552 | 0.9297 | 197 |
| normal | 0.7717 | 0.9464 | 0.9261 | 0.8419 | 0.8264 | 0.9671 | 0.9624 | 276 |
| sever | 0.9917 | 0.9577 | 0.8561 | 0.9189 | 0.9978 | 0.9989 | 0.9959 | 120 |

### 每类 TP/FP/TN/FN

- **heelspur**: TP=172 FP=52 TN=344 FN=25
- **normal**: TP=213 FP=17 TN=300 FN=63
- **sever**: TP=119 FP=20 TN=453 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | heelspur | normal | sever |
|---|---|---|---|
| heelspur | 172 | 17 | 8 |
| normal | 51 | 213 | 12 |
| sever | 1 | 0 | 119 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

    heelspur     0.7679    0.8731    0.8171       197
      normal     0.9261    0.7717    0.8419       276
       sever     0.8561    0.9917    0.9189       120

    accuracy                         0.8499       593
   macro avg     0.8500    0.8788    0.8593       593
weighted avg     0.8594    0.8499    0.8492       593

```