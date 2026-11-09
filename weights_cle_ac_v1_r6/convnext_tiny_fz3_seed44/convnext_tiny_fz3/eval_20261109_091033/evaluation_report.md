# convnext_tiny_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_seed44/convnext_tiny_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9392 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9924 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9392 |
| Balanced Acc | 0.9389 |
| Macro-Sensitivity | 0.9389 |
| Macro-Specificity | 0.9694 |
| Macro-Precision | 0.9397 |
| AUC-PR (Macro) | 0.9859 |
| Cohen's Kappa | 0.9086 |
| MCC | 0.9087 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 893.6 (1.12 ms/张) |
| End-to-End FPS | 113.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9538 | 0.9190 | 0.9234 | 0.9590 | 0.9889 | 0.9824 | 208 |
| Malignant_lesions | 0.9148 | 0.9725 | 0.9360 | 0.9253 | 0.9629 | 0.9902 | 0.9792 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9818 | 0.9639 | 0.9689 | 0.9869 | 0.9981 | 0.9961 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=17 TN=351 FN=15
- **Malignant_lesions**: TP=161 FP=11 TN=389 FN=15
- **body_of_stomach_normal**: TP=187 FP=7 TN=377 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 10 | 5 |
| Malignant_lesions | 13 | 161 | 2 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9190    0.9279    0.9234       208
     Malignant_lesions     0.9360    0.9148    0.9253       176
body_of_stomach_normal     0.9639    0.9740    0.9689       192

              accuracy                         0.9392       576
             macro avg     0.9397    0.9389    0.9392       576
          weighted avg     0.9392    0.9392    0.9392       576

```