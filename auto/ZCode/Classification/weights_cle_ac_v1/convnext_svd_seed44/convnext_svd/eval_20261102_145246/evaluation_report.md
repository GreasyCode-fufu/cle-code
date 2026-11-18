# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_svd_seed44/convnext_svd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9540 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9547 |
| Balanced Acc | 0.9534 |
| Macro-Sensitivity | 0.9534 |
| Macro-Specificity | 0.9773 |
| Macro-Precision | 0.9549 |
| AUC-PR (Macro) | 0.9908 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9322 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 415.9 (2.40 ms/张) |
| End-to-End FPS | 176.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9674 | 0.9437 | 0.9549 | 0.9807 | 0.9950 | 0.9920 | 208 |
| Malignant_lesions | 0.9148 | 0.9775 | 0.9471 | 0.9306 | 0.9631 | 0.9917 | 0.9827 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9870 | 0.9741 | 0.9766 | 0.9896 | 0.9989 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=12 TN=356 FN=7
- **Malignant_lesions**: TP=161 FP=9 TN=391 FN=15
- **body_of_stomach_normal**: TP=188 FP=5 TN=379 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 11 | 161 | 4 |
| body_of_stomach_normal | 1 | 3 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9437    0.9663    0.9549       208
     Malignant_lesions     0.9471    0.9148    0.9306       176
body_of_stomach_normal     0.9741    0.9792    0.9766       192

              accuracy                         0.9549       576
             macro avg     0.9549    0.9534    0.9540       576
          weighted avg     0.9548    0.9549    0.9547       576

```