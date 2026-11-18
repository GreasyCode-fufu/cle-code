# shufflenet_v2_x1_0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/shufflenet_v2_x1_0_seed42/shufflenet_v2_x1_0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9269 |
| Accuracy | 0.9271 |
| Macro-AUROC (OvR) | 0.9851 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9271 |
| Balanced Acc | 0.9242 |
| Macro-Sensitivity | 0.9242 |
| Macro-Specificity | 0.9626 |
| Macro-Precision | 0.9332 |
| AUC-PR (Macro) | 0.9689 |
| Cohen's Kappa | 0.8900 |
| MCC | 0.8918 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 151.69M |
| FLOPs | 303.38M |
| Params | 1.26M |
| Pure-Inference FPS | 7260.2 (0.14 ms/张) |
| End-to-End FPS | 105.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9158 | 0.8652 | 0.9087 | 0.9740 | 0.9798 | 0.9550 | 208 |
| Malignant_lesions | 0.8523 | 0.9825 | 0.9554 | 0.9009 | 0.9379 | 0.9772 | 0.9549 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9896 | 0.9788 | 0.9711 | 0.9819 | 0.9983 | 0.9968 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=31 TN=337 FN=9
- **Malignant_lesions**: TP=150 FP=7 TN=393 FN=26
- **body_of_stomach_normal**: TP=185 FP=4 TN=380 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 24 | 150 | 2 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8652    0.9567    0.9087       208
     Malignant_lesions     0.9554    0.8523    0.9009       176
body_of_stomach_normal     0.9788    0.9635    0.9711       192

              accuracy                         0.9271       576
             macro avg     0.9332    0.9242    0.9269       576
          weighted avg     0.9307    0.9271    0.9271       576

```