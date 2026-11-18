# convnext_tiny_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_dce4m_seed42/convnext_tiny_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9566 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9955 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9567 |
| Balanced Acc | 0.9568 |
| Macro-Sensitivity | 0.9568 |
| Macro-Specificity | 0.9783 |
| Macro-Precision | 0.9564 |
| AUC-PR (Macro) | 0.9915 |
| Cohen's Kappa | 0.9348 |
| MCC | 0.9348 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1602.3 (0.62 ms/张) |
| End-to-End FPS | 7.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9701 | 0.9469 | 0.9446 | 0.9675 | 0.9937 | 0.9890 | 208 |
| Malignant_lesions | 0.9489 | 0.9700 | 0.9330 | 0.9408 | 0.9773 | 0.9938 | 0.9873 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9948 | 0.9895 | 0.9843 | 0.9896 | 0.9989 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=11 TN=357 FN=12
- **Malignant_lesions**: TP=167 FP=12 TN=388 FN=9
- **body_of_stomach_normal**: TP=188 FP=2 TN=382 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 8 | 167 | 1 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9469    0.9423    0.9446       208
     Malignant_lesions     0.9330    0.9489    0.9408       176
body_of_stomach_normal     0.9895    0.9792    0.9843       192

              accuracy                         0.9566       576
             macro avg     0.9564    0.9568    0.9566       576
          weighted avg     0.9568    0.9566    0.9567       576

```