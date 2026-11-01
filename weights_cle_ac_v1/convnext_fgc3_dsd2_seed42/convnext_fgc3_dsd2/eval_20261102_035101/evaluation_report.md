# convnext_fgc3_dsd2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc3_dsd2_seed42/convnext_fgc3_dsd2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9423 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9923 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9427 |
| Balanced Acc | 0.9415 |
| Macro-Sensitivity | 0.9415 |
| Macro-Specificity | 0.9711 |
| Macro-Precision | 0.9435 |
| AUC-PR (Macro) | 0.9861 |
| Cohen's Kappa | 0.9138 |
| MCC | 0.9140 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.41G |
| FLOPs | 8.81G |
| Params | 27.61M |
| Pure-Inference FPS | 690.1 (1.45 ms/张) |
| End-to-End FPS | 211.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9511 | 0.9163 | 0.9314 | 0.9695 | 0.9891 | 0.9834 | 208 |
| Malignant_lesions | 0.9034 | 0.9725 | 0.9353 | 0.9191 | 0.9581 | 0.9922 | 0.9810 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9896 | 0.9791 | 0.9765 | 0.9870 | 0.9957 | 0.9940 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=18 TN=350 FN=11
- **Malignant_lesions**: TP=159 FP=11 TN=389 FN=17
- **body_of_stomach_normal**: TP=187 FP=4 TN=380 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 14 | 159 | 3 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9163    0.9471    0.9314       208
     Malignant_lesions     0.9353    0.9034    0.9191       176
body_of_stomach_normal     0.9791    0.9740    0.9765       192

              accuracy                         0.9427       576
             macro avg     0.9435    0.9415    0.9423       576
          weighted avg     0.9430    0.9427    0.9427       576

```