# convnext_v4b 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v4b_seed42/convnext_v4b/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9545 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9924 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9547 |
| Balanced Acc | 0.9543 |
| Macro-Sensitivity | 0.9543 |
| Macro-Specificity | 0.9773 |
| Macro-Precision | 0.9549 |
| AUC-PR (Macro) | 0.9819 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9322 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.93G |
| FLOPs | 7.87G |
| Params | 14.08M |
| Pure-Inference FPS | 372.2 (2.69 ms/张) |
| End-to-End FPS | 171.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9674 | 0.9426 | 0.9448 | 0.9700 | 0.9891 | 0.9833 | 208 |
| Malignant_lesions | 0.9261 | 0.9775 | 0.9477 | 0.9368 | 0.9678 | 0.9887 | 0.9635 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=12 TN=356 FN=11
- **Malignant_lesions**: TP=163 FP=9 TN=391 FN=13
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 10 | 163 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9426    0.9471    0.9448       208
     Malignant_lesions     0.9477    0.9261    0.9368       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9549       576
             macro avg     0.9549    0.9543    0.9545       576
          weighted avg     0.9547    0.9549    0.9547       576

```