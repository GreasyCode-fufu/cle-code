# vit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/vit_base_patch16_224_seed43/vit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9712 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9981 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9766 |
| Balanced Acc | 0.9729 |
| Macro-Sensitivity | 0.9729 |
| Macro-Specificity | 0.9885 |
| Macro-Precision | 0.9697 |
| AUC-PR (Macro) | 0.9924 |
| Cohen's Kappa | 0.9628 |
| MCC | 0.9629 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 218.2 (4.58 ms/张) |
| End-to-End FPS | 119.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9871 | 0.9844 | 0.9767 | 0.9745 | 0.9984 | 0.9982 | 195 |
| Malignant_lesions | 0.9494 | 0.9856 | 0.9375 | 0.9434 | 0.9885 | 0.9957 | 0.9791 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=3 TN=229 FN=6
- **Malignant_lesions**: TP=75 FP=5 TN=343 FN=4
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 5 | 1 |
| Malignant_lesions | 3 | 75 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9844    0.9692    0.9767       195
     Malignant_lesions     0.9375    0.9494    0.9434        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9766       427
             macro avg     0.9697    0.9729    0.9712       427
          weighted avg     0.9767    0.9766    0.9766       427

```