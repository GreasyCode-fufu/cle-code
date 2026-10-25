# convnext_v2abc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2abc_seed44/convnext_v2abc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9734 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9971 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9790 |
| Balanced Acc | 0.9746 |
| Macro-Sensitivity | 0.9746 |
| Macro-Specificity | 0.9895 |
| Macro-Precision | 0.9723 |
| AUC-PR (Macro) | 0.9918 |
| Cohen's Kappa | 0.9665 |
| MCC | 0.9665 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.66G |
| FLOPs | 9.31G |
| Params | 28.67M |
| Pure-Inference FPS | 507.8 (1.97 ms/张) |
| End-to-End FPS | 192.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9828 | 0.9794 | 0.9769 | 0.9785 | 0.9967 | 0.9959 | 195 |
| Malignant_lesions | 0.9494 | 0.9856 | 0.9375 | 0.9434 | 0.9885 | 0.9947 | 0.9794 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=4 TN=228 FN=5
- **Malignant_lesions**: TP=75 FP=5 TN=343 FN=4
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 5 | 0 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9794    0.9744    0.9769       195
     Malignant_lesions     0.9375    0.9494    0.9434        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9789       427
             macro avg     0.9723    0.9746    0.9734       427
          weighted avg     0.9790    0.9789    0.9790       427

```