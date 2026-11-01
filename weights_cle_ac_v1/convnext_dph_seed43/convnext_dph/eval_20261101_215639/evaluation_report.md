# convnext_dph 评估报告

- 权重: `weights_cle_ac_v1/convnext_dph_seed43/convnext_dph/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9777 |
| Accuracy | 0.9813 |
| Macro-AUROC (OvR) | 0.9988 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9814 |
| Balanced Acc | 0.9838 |
| Macro-Sensitivity | 0.9838 |
| Macro-Specificity | 0.9916 |
| Macro-Precision | 0.9723 |
| AUC-PR (Macro) | 0.9961 |
| Cohen's Kappa | 0.9704 |
| MCC | 0.9706 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 1666.3 (0.60 ms/张) |
| End-to-End FPS | 223.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9957 | 0.9947 | 0.9792 | 0.9706 | 0.9989 | 0.9987 | 195 |
| Malignant_lesions | 0.9873 | 0.9828 | 0.9286 | 0.9571 | 0.9971 | 0.9976 | 0.9897 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=1 TN=231 FN=7
- **Malignant_lesions**: TP=78 FP=6 TN=342 FN=1
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 6 | 1 |
| Malignant_lesions | 1 | 78 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9947    0.9641    0.9792       195
     Malignant_lesions     0.9286    0.9873    0.9571        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9813       427
             macro avg     0.9723    0.9838    0.9777       427
          weighted avg     0.9820    0.9813    0.9814       427

```