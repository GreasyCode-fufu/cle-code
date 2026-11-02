# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_ef_seed45/convnext_ef/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9882 |
| Accuracy | 0.9906 |
| Macro-AUROC (OvR) | 0.9986 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9907 |
| Balanced Acc | 0.9907 |
| Macro-Sensitivity | 0.9907 |
| Macro-Specificity | 0.9957 |
| Macro-Precision | 0.9859 |
| AUC-PR (Macro) | 0.9937 |
| Cohen's Kappa | 0.9851 |
| MCC | 0.9852 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 384.6 (2.60 ms/张) |
| End-to-End FPS | 169.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9846 | 0.9957 | 0.9948 | 0.9897 | 0.9872 | 0.9985 | 0.9985 | 195 |
| Malignant_lesions | 0.9873 | 0.9914 | 0.9630 | 0.9750 | 0.9971 | 0.9972 | 0.9827 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=1 TN=231 FN=3
- **Malignant_lesions**: TP=78 FP=3 TN=345 FN=1
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 3 | 0 |
| Malignant_lesions | 1 | 78 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9948    0.9846    0.9897       195
     Malignant_lesions     0.9630    0.9873    0.9750        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9906       427
             macro avg     0.9859    0.9907    0.9882       427
          weighted avg     0.9908    0.9906    0.9907       427

```