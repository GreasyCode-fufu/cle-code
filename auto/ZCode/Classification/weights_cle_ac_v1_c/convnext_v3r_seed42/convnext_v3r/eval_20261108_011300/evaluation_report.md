# convnext_v3r 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v3r_seed42/convnext_v3r/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9562 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9960 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9565 |
| Balanced Acc | 0.9570 |
| Macro-Sensitivity | 0.9570 |
| Macro-Specificity | 0.9785 |
| Macro-Precision | 0.9557 |
| AUC-PR (Macro) | 0.9921 |
| Cohen's Kappa | 0.9348 |
| MCC | 0.9350 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.65G |
| FLOPs | 9.29G |
| Params | 28.68M |
| Pure-Inference FPS | 497.3 (2.01 ms/张) |
| End-to-End FPS | 189.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9837 | 0.9700 | 0.9510 | 0.9628 | 0.9947 | 0.9915 | 208 |
| Malignant_lesions | 0.9489 | 0.9675 | 0.9278 | 0.9382 | 0.9773 | 0.9940 | 0.9860 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9844 | 0.9694 | 0.9794 | 0.9947 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=6 TN=362 FN=14
- **Malignant_lesions**: TP=167 FP=13 TN=387 FN=9
- **body_of_stomach_normal**: TP=190 FP=6 TN=378 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 12 | 2 |
| Malignant_lesions | 5 | 167 | 4 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9700    0.9327    0.9510       208
     Malignant_lesions     0.9278    0.9489    0.9382       176
body_of_stomach_normal     0.9694    0.9896    0.9794       192

              accuracy                         0.9566       576
             macro avg     0.9557    0.9570    0.9562       576
          weighted avg     0.9569    0.9566    0.9565       576

```