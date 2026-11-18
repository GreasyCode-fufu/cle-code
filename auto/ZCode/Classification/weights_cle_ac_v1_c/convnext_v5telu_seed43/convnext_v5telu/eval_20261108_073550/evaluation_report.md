# convnext_v5telu 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5telu_seed43/convnext_v5telu/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9564 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9958 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9566 |
| Balanced Acc | 0.9573 |
| Macro-Sensitivity | 0.9573 |
| Macro-Specificity | 0.9785 |
| Macro-Precision | 0.9559 |
| AUC-PR (Macro) | 0.9919 |
| Cohen's Kappa | 0.9348 |
| MCC | 0.9350 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.25M |
| Pure-Inference FPS | 402.5 (2.48 ms/张) |
| End-to-End FPS | 170.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9783 | 0.9602 | 0.9438 | 0.9600 | 0.9940 | 0.9899 | 208 |
| Malignant_lesions | 0.9545 | 0.9625 | 0.9180 | 0.9359 | 0.9796 | 0.9936 | 0.9863 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9948 | 0.9896 | 0.9896 | 0.9948 | 0.9998 | 0.9995 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=8 TN=360 FN=15
- **Malignant_lesions**: TP=168 FP=15 TN=385 FN=8
- **body_of_stomach_normal**: TP=190 FP=2 TN=382 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 13 | 2 |
| Malignant_lesions | 8 | 168 | 0 |
| body_of_stomach_normal | 0 | 2 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9602    0.9279    0.9438       208
     Malignant_lesions     0.9180    0.9545    0.9359       176
body_of_stomach_normal     0.9896    0.9896    0.9896       192

              accuracy                         0.9566       576
             macro avg     0.9559    0.9573    0.9564       576
          weighted avg     0.9571    0.9566    0.9566       576

```