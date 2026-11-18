# convnext_v5se 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5se_seed42/convnext_v5se/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9765 |
| Accuracy | 0.9813 |
| Macro-AUROC (OvR) | 0.9981 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9813 |
| Balanced Acc | 0.9788 |
| Macro-Sensitivity | 0.9788 |
| Macro-Specificity | 0.9909 |
| Macro-Precision | 0.9742 |
| AUC-PR (Macro) | 0.9941 |
| Cohen's Kappa | 0.9703 |
| MCC | 0.9703 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.47M |
| Pure-Inference FPS | 438.1 (2.28 ms/张) |
| End-to-End FPS | 181.5 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9744 | 0.9871 | 0.9845 | 0.9794 | 0.9786 | 0.9979 | 0.9976 | 195 |
| Malignant_lesions | 0.9620 | 0.9856 | 0.9383 | 0.9500 | 0.9913 | 0.9965 | 0.9846 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=3 TN=229 FN=5
- **Malignant_lesions**: TP=76 FP=5 TN=343 FN=3
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 5 | 0 |
| Malignant_lesions | 3 | 76 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9845    0.9744    0.9794       195
     Malignant_lesions     0.9383    0.9620    0.9500        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9813       427
             macro avg     0.9742    0.9788    0.9765       427
          weighted avg     0.9815    0.9813    0.9813       427

```