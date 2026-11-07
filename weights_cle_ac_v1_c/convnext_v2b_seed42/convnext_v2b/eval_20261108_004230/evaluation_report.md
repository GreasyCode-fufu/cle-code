# convnext_v2b 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v2b_seed42/convnext_v2b/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9580 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9934 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9583 |
| Balanced Acc | 0.9568 |
| Macro-Sensitivity | 0.9568 |
| Macro-Specificity | 0.9788 |
| Macro-Precision | 0.9601 |
| AUC-PR (Macro) | 0.9875 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9377 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.93G |
| Params | 28.22M |
| Pure-Inference FPS | 1483.2 (0.67 ms/张) |
| End-to-End FPS | 257.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9565 | 0.9266 | 0.9484 | 0.9832 | 0.9919 | 0.9856 | 208 |
| Malignant_lesions | 0.9148 | 0.9850 | 0.9641 | 0.9388 | 0.9633 | 0.9887 | 0.9777 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9948 | 0.9895 | 0.9869 | 0.9922 | 0.9997 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=16 TN=352 FN=6
- **Malignant_lesions**: TP=161 FP=6 TN=394 FN=15
- **body_of_stomach_normal**: TP=189 FP=2 TN=382 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 14 | 161 | 1 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9266    0.9712    0.9484       208
     Malignant_lesions     0.9641    0.9148    0.9388       176
body_of_stomach_normal     0.9895    0.9844    0.9869       192

              accuracy                         0.9583       576
             macro avg     0.9601    0.9568    0.9580       576
          weighted avg     0.9590    0.9583    0.9583       576

```