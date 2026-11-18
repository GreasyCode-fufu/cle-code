# convnext_v6fx 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6fx_seed43/convnext_v6fx/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9849 |
| Accuracy | 0.9883 |
| Macro-AUROC (OvR) | 0.9997 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9882 |
| Balanced Acc | 0.9789 |
| Macro-Sensitivity | 0.9789 |
| Macro-Specificity | 0.9928 |
| Macro-Precision | 0.9917 |
| AUC-PR (Macro) | 0.9992 |
| Cohen's Kappa | 0.9813 |
| MCC | 0.9815 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.70M |
| Pure-Inference FPS | 614.3 (1.63 ms/张) |
| End-to-End FPS | 195.7 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 1.0000 | 0.9784 | 0.9750 | 0.9873 | 1.0000 | 0.9997 | 0.9996 | 195 |
| Malignant_lesions | 0.9367 | 1.0000 | 1.0000 | 0.9673 | 0.9858 | 0.9995 | 0.9979 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=5 TN=227 FN=0
- **Malignant_lesions**: TP=74 FP=0 TN=348 FN=5
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 0 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9750    1.0000    0.9873       195
     Malignant_lesions     1.0000    0.9367    0.9673        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9883       427
             macro avg     0.9917    0.9789    0.9849       427
          weighted avg     0.9886    0.9883    0.9882       427

```