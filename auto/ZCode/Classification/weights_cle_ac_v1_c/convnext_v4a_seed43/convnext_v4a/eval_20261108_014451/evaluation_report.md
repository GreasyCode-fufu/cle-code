# convnext_v4a 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4a_seed43/convnext_v4a/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9443 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9948 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9431 |
| Balanced Acc | 0.9423 |
| Macro-Sensitivity | 0.9423 |
| Macro-Specificity | 0.9705 |
| Macro-Precision | 0.9474 |
| AUC-PR (Macro) | 0.9908 |
| Cohen's Kappa | 0.9137 |
| MCC | 0.9143 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 2.55G |
| FLOPs | 5.09G |
| Params | 17.62M |
| Pure-Inference FPS | 284.0 (3.52 ms/张) |
| End-to-End FPS | 148.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9375 | 0.8964 | 0.9256 | 0.9746 | 0.9907 | 0.9844 | 208 |
| Malignant_lesions | 0.9432 | 0.9975 | 0.9940 | 0.9679 | 0.9756 | 0.9978 | 0.9956 | 176 |
| body_of_stomach_normal | 0.9271 | 0.9766 | 0.9519 | 0.9393 | 0.9640 | 0.9957 | 0.9925 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=23 TN=345 FN=9
- **Malignant_lesions**: TP=166 FP=1 TN=399 FN=10
- **body_of_stomach_normal**: TP=178 FP=9 TN=375 FN=14

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 1 | 8 |
| Malignant_lesions | 9 | 166 | 1 |
| body_of_stomach_normal | 14 | 0 | 178 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8964    0.9567    0.9256       208
     Malignant_lesions     0.9940    0.9432    0.9679       176
body_of_stomach_normal     0.9519    0.9271    0.9393       192

              accuracy                         0.9427       576
             macro avg     0.9474    0.9423    0.9443       576
          weighted avg     0.9447    0.9427    0.9431       576

```