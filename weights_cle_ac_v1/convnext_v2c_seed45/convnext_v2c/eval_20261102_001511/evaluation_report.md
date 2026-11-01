# convnext_v2c 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2c_seed45/convnext_v2c/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9546 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9547 |
| Balanced Acc | 0.9547 |
| Macro-Sensitivity | 0.9547 |
| Macro-Specificity | 0.9774 |
| Macro-Precision | 0.9545 |
| AUC-PR (Macro) | 0.9924 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9322 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 1662.8 (0.60 ms/张) |
| End-to-End FPS | 271.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9701 | 0.9466 | 0.9420 | 0.9649 | 0.9940 | 0.9903 | 208 |
| Malignant_lesions | 0.9318 | 0.9750 | 0.9425 | 0.9371 | 0.9701 | 0.9946 | 0.9878 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9870 | 0.9745 | 0.9845 | 0.9974 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=11 TN=357 FN=13
- **Malignant_lesions**: TP=164 FP=10 TN=390 FN=12
- **body_of_stomach_normal**: TP=191 FP=5 TN=379 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 10 | 3 |
| Malignant_lesions | 10 | 164 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9466    0.9375    0.9420       208
     Malignant_lesions     0.9425    0.9318    0.9371       176
body_of_stomach_normal     0.9745    0.9948    0.9845       192

              accuracy                         0.9549       576
             macro avg     0.9545    0.9547    0.9546       576
          weighted avg     0.9547    0.9549    0.9547       576

```