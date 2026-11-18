# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz8m_seed44/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9578 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9581 |
| Balanced Acc | 0.9566 |
| Macro-Sensitivity | 0.9566 |
| Macro-Specificity | 0.9788 |
| Macro-Precision | 0.9601 |
| AUC-PR (Macro) | 0.9909 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9377 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3741.8 (0.27 ms/张) |
| End-to-End FPS | 207.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9620 | 0.9352 | 0.9528 | 0.9833 | 0.9917 | 0.9881 | 208 |
| Malignant_lesions | 0.9091 | 0.9900 | 0.9756 | 0.9412 | 0.9612 | 0.9943 | 0.9871 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9844 | 0.9694 | 0.9794 | 0.9947 | 0.9988 | 0.9976 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=14 TN=354 FN=6
- **Malignant_lesions**: TP=160 FP=4 TN=396 FN=16
- **body_of_stomach_normal**: TP=190 FP=6 TN=378 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 12 | 160 | 4 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9352    0.9712    0.9528       208
     Malignant_lesions     0.9756    0.9091    0.9412       176
body_of_stomach_normal     0.9694    0.9896    0.9794       192

              accuracy                         0.9583       576
             macro avg     0.9601    0.9566    0.9578       576
          weighted avg     0.9589    0.9583    0.9581       576

```