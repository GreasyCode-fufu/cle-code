# convnext_v2ab 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2ab_seed43/convnext_v2ab/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9457 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9922 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9461 |
| Balanced Acc | 0.9450 |
| Macro-Sensitivity | 0.9450 |
| Macro-Specificity | 0.9728 |
| Macro-Precision | 0.9469 |
| AUC-PR (Macro) | 0.9854 |
| Cohen's Kappa | 0.9190 |
| MCC | 0.9192 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.48G |
| FLOPs | 8.97G |
| Params | 28.22M |
| Pure-Inference FPS | 919.1 (1.09 ms/张) |
| End-to-End FPS | 236.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9538 | 0.9206 | 0.9336 | 0.9696 | 0.9902 | 0.9786 | 208 |
| Malignant_lesions | 0.9034 | 0.9750 | 0.9408 | 0.9217 | 0.9582 | 0.9871 | 0.9791 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9896 | 0.9793 | 0.9818 | 0.9922 | 0.9992 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=17 TN=351 FN=11
- **Malignant_lesions**: TP=159 FP=10 TN=390 FN=17
- **body_of_stomach_normal**: TP=189 FP=4 TN=380 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 15 | 159 | 2 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9206    0.9471    0.9336       208
     Malignant_lesions     0.9408    0.9034    0.9217       176
body_of_stomach_normal     0.9793    0.9844    0.9818       192

              accuracy                         0.9462       576
             macro avg     0.9469    0.9450    0.9457       576
          weighted avg     0.9463    0.9462    0.9461       576

```