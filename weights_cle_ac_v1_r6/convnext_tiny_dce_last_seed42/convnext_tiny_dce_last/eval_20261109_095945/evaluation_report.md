# convnext_tiny_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_dce_last_seed42/convnext_tiny_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9495 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9946 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9497 |
| Balanced Acc | 0.9498 |
| Macro-Sensitivity | 0.9498 |
| Macro-Specificity | 0.9749 |
| Macro-Precision | 0.9493 |
| AUC-PR (Macro) | 0.9900 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9244 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1707.5 (0.59 ms/张) |
| End-to-End FPS | 236.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9674 | 0.9417 | 0.9372 | 0.9622 | 0.9926 | 0.9870 | 208 |
| Malignant_lesions | 0.9375 | 0.9650 | 0.9218 | 0.9296 | 0.9723 | 0.9927 | 0.9853 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9986 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=12 TN=356 FN=14
- **Malignant_lesions**: TP=165 FP=14 TN=386 FN=11
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 13 | 1 |
| Malignant_lesions | 9 | 165 | 2 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9417    0.9327    0.9372       208
     Malignant_lesions     0.9218    0.9375    0.9296       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9497       576
             macro avg     0.9493    0.9498    0.9495       576
          weighted avg     0.9498    0.9497    0.9497       576

```