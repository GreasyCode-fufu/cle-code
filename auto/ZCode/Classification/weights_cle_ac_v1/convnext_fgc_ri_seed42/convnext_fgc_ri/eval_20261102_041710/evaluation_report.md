# convnext_fgc_ri 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_ri_seed42/convnext_fgc_ri/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9545 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9948 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9548 |
| Balanced Acc | 0.9543 |
| Macro-Sensitivity | 0.9543 |
| Macro-Specificity | 0.9773 |
| Macro-Precision | 0.9547 |
| AUC-PR (Macro) | 0.9886 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9321 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 1081.5 (0.92 ms/张) |
| End-to-End FPS | 235.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9674 | 0.9426 | 0.9448 | 0.9700 | 0.9927 | 0.9897 | 208 |
| Malignant_lesions | 0.9261 | 0.9750 | 0.9422 | 0.9341 | 0.9677 | 0.9920 | 0.9767 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9896 | 0.9794 | 0.9845 | 0.9948 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=12 TN=356 FN=11
- **Malignant_lesions**: TP=163 FP=10 TN=390 FN=13
- **body_of_stomach_normal**: TP=190 FP=4 TN=380 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 10 | 163 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9426    0.9471    0.9448       208
     Malignant_lesions     0.9422    0.9261    0.9341       176
body_of_stomach_normal     0.9794    0.9896    0.9845       192

              accuracy                         0.9549       576
             macro avg     0.9547    0.9543    0.9545       576
          weighted avg     0.9547    0.9549    0.9548       576

```