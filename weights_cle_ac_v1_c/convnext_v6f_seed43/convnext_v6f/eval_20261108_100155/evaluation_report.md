# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6f_seed43/convnext_v6f/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9653 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9653 |
| Balanced Acc | 0.9657 |
| Macro-Sensitivity | 0.9657 |
| Macro-Specificity | 0.9827 |
| Macro-Precision | 0.9649 |
| AUC-PR (Macro) | 0.9908 |
| Cohen's Kappa | 0.9478 |
| MCC | 0.9479 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 481.5 (2.08 ms/张) |
| End-to-End FPS | 187.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9810 | 0.9659 | 0.9588 | 0.9730 | 0.9923 | 0.9901 | 208 |
| Malignant_lesions | 0.9659 | 0.9775 | 0.9497 | 0.9577 | 0.9849 | 0.9938 | 0.9850 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9896 | 0.9792 | 0.9792 | 0.9896 | 0.9986 | 0.9973 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=7 TN=361 FN=10
- **Malignant_lesions**: TP=170 FP=9 TN=391 FN=6
- **body_of_stomach_normal**: TP=188 FP=4 TN=380 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 9 | 1 |
| Malignant_lesions | 3 | 170 | 3 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9659    0.9519    0.9588       208
     Malignant_lesions     0.9497    0.9659    0.9577       176
body_of_stomach_normal     0.9792    0.9792    0.9792       192

              accuracy                         0.9653       576
             macro avg     0.9649    0.9657    0.9653       576
          weighted avg     0.9654    0.9653    0.9653       576

```