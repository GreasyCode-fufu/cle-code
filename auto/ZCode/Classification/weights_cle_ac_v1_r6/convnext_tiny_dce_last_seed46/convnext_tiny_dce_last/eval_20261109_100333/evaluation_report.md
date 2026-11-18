# convnext_tiny_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_dce_last_seed46/convnext_tiny_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9612 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9615 |
| Balanced Acc | 0.9599 |
| Macro-Sensitivity | 0.9599 |
| Macro-Specificity | 0.9805 |
| Macro-Precision | 0.9637 |
| AUC-PR (Macro) | 0.9912 |
| Cohen's Kappa | 0.9425 |
| MCC | 0.9430 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1702.1 (0.59 ms/张) |
| End-to-End FPS | 5.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9647 | 0.9398 | 0.9575 | 0.9861 | 0.9937 | 0.9887 | 208 |
| Malignant_lesions | 0.9091 | 0.9925 | 0.9816 | 0.9440 | 0.9613 | 0.9928 | 0.9853 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9844 | 0.9695 | 0.9820 | 0.9974 | 0.9997 | 0.9995 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=13 TN=355 FN=5
- **Malignant_lesions**: TP=160 FP=3 TN=397 FN=16
- **body_of_stomach_normal**: TP=191 FP=6 TN=378 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 3 | 2 |
| Malignant_lesions | 12 | 160 | 4 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9398    0.9760    0.9575       208
     Malignant_lesions     0.9816    0.9091    0.9440       176
body_of_stomach_normal     0.9695    0.9948    0.9820       192

              accuracy                         0.9618       576
             macro avg     0.9637    0.9599    0.9612       576
          weighted avg     0.9625    0.9618    0.9615       576

```