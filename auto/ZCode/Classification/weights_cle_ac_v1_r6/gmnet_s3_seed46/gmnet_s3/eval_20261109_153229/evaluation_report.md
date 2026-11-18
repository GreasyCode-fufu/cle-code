# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/gmnet_s3_seed46/gmnet_s3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9334 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9923 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9337 |
| Balanced Acc | 0.9326 |
| Macro-Sensitivity | 0.9326 |
| Macro-Specificity | 0.9666 |
| Macro-Precision | 0.9349 |
| AUC-PR (Macro) | 0.9849 |
| Cohen's Kappa | 0.9007 |
| MCC | 0.9010 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1323.2 (0.76 ms/张) |
| End-to-End FPS | 212.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9457 | 0.9065 | 0.9194 | 0.9613 | 0.9890 | 0.9824 | 208 |
| Malignant_lesions | 0.8807 | 0.9725 | 0.9337 | 0.9064 | 0.9488 | 0.9888 | 0.9740 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9818 | 0.9643 | 0.9742 | 0.9921 | 0.9990 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=20 TN=348 FN=14
- **Malignant_lesions**: TP=155 FP=11 TN=389 FN=21
- **body_of_stomach_normal**: TP=189 FP=7 TN=377 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 11 | 3 |
| Malignant_lesions | 17 | 155 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9065    0.9327    0.9194       208
     Malignant_lesions     0.9337    0.8807    0.9064       176
body_of_stomach_normal     0.9643    0.9844    0.9742       192

              accuracy                         0.9340       576
             macro avg     0.9349    0.9326    0.9334       576
          weighted avg     0.9341    0.9340    0.9337       576

```