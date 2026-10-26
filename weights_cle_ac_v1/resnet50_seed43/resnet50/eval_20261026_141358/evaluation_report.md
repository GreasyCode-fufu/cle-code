# resnet50 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnet50_seed43/resnet50/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9743 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9979 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9789 |
| Balanced Acc | 0.9746 |
| Macro-Sensitivity | 0.9746 |
| Macro-Specificity | 0.9892 |
| Macro-Precision | 0.9741 |
| AUC-PR (Macro) | 0.9923 |
| Cohen's Kappa | 0.9665 |
| MCC | 0.9665 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.13G |
| FLOPs | 8.26G |
| Params | 23.51M |
| Pure-Inference FPS | 1158.5 (0.86 ms/张) |
| End-to-End FPS | 227.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9828 | 0.9794 | 0.9769 | 0.9785 | 0.9977 | 0.9976 | 195 |
| Malignant_lesions | 0.9494 | 0.9885 | 0.9494 | 0.9494 | 0.9885 | 0.9960 | 0.9794 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=4 TN=228 FN=5
- **Malignant_lesions**: TP=75 FP=4 TN=344 FN=4
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 4 | 1 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9794    0.9744    0.9769       195
     Malignant_lesions     0.9494    0.9494    0.9494        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9789       427
             macro avg     0.9741    0.9746    0.9743       427
          weighted avg     0.9789    0.9789    0.9789       427

```