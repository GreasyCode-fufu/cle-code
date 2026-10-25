# convnext_wtb_msf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wtb_msf_seed42/convnext_wtb_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9470 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9476 |
| Balanced Acc | 0.9447 |
| Macro-Sensitivity | 0.9447 |
| Macro-Specificity | 0.9732 |
| Macro-Precision | 0.9526 |
| AUC-PR (Macro) | 0.9903 |
| Cohen's Kappa | 0.9215 |
| MCC | 0.9231 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.52G |
| FLOPs | 9.05G |
| Params | 29.54M |
| Pure-Inference FPS | 1168.7 (0.86 ms/张) |
| End-to-End FPS | 248.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9348 | 0.8947 | 0.9358 | 0.9885 | 0.9942 | 0.9903 | 208 |
| Malignant_lesions | 0.8636 | 0.9875 | 0.9682 | 0.9129 | 0.9427 | 0.9925 | 0.9839 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9974 | 0.9948 | 0.9922 | 0.9948 | 0.9981 | 0.9968 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=24 TN=344 FN=4
- **Malignant_lesions**: TP=152 FP=5 TN=395 FN=24
- **body_of_stomach_normal**: TP=190 FP=1 TN=383 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 4 | 0 |
| Malignant_lesions | 23 | 152 | 1 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8947    0.9808    0.9358       208
     Malignant_lesions     0.9682    0.8636    0.9129       176
body_of_stomach_normal     0.9948    0.9896    0.9922       192

              accuracy                         0.9479       576
             macro avg     0.9526    0.9447    0.9470       576
          weighted avg     0.9505    0.9479    0.9476       576

```