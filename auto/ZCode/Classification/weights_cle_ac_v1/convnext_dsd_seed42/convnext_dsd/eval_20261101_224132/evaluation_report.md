# convnext_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dsd_seed42/convnext_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9549 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9945 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9549 |
| Balanced Acc | 0.9553 |
| Macro-Sensitivity | 0.9553 |
| Macro-Specificity | 0.9774 |
| Macro-Precision | 0.9546 |
| AUC-PR (Macro) | 0.9891 |
| Cohen's Kappa | 0.9322 |
| MCC | 0.9322 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 626.7 (1.60 ms/张) |
| End-to-End FPS | 207.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9701 | 0.9463 | 0.9395 | 0.9623 | 0.9915 | 0.9863 | 208 |
| Malignant_lesions | 0.9489 | 0.9700 | 0.9330 | 0.9408 | 0.9773 | 0.9923 | 0.9820 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9922 | 0.9844 | 0.9844 | 0.9922 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=11 TN=357 FN=14
- **Malignant_lesions**: TP=167 FP=12 TN=388 FN=9
- **body_of_stomach_normal**: TP=189 FP=3 TN=381 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 12 | 2 |
| Malignant_lesions | 8 | 167 | 1 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9463    0.9327    0.9395       208
     Malignant_lesions     0.9330    0.9489    0.9408       176
body_of_stomach_normal     0.9844    0.9844    0.9844       192

              accuracy                         0.9549       576
             macro avg     0.9546    0.9553    0.9549       576
          weighted avg     0.9549    0.9549    0.9549       576

```