# convnext_tiny_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_dce_last_seed45/convnext_tiny_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9651 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9653 |
| Balanced Acc | 0.9639 |
| Macro-Sensitivity | 0.9639 |
| Macro-Specificity | 0.9823 |
| Macro-Precision | 0.9669 |
| AUC-PR (Macro) | 0.9940 |
| Cohen's Kappa | 0.9477 |
| MCC | 0.9480 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1712.5 (0.58 ms/张) |
| End-to-End FPS | 225.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9647 | 0.9401 | 0.9600 | 0.9889 | 0.9957 | 0.9928 | 208 |
| Malignant_lesions | 0.9318 | 0.9900 | 0.9762 | 0.9535 | 0.9706 | 0.9953 | 0.9901 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=13 TN=355 FN=4
- **Malignant_lesions**: TP=164 FP=4 TN=396 FN=12
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 4 | 0 |
| Malignant_lesions | 9 | 164 | 3 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9401    0.9808    0.9600       208
     Malignant_lesions     0.9762    0.9318    0.9535       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9653       576
             macro avg     0.9669    0.9639    0.9651       576
          weighted avg     0.9659    0.9653    0.9653       576

```