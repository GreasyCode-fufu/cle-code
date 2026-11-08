# mbv3_r6_s2_fz8 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_fz8_seed46/mbv3_r6_s2_fz8/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9509 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9944 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9511 |
| Balanced Acc | 0.9493 |
| Macro-Sensitivity | 0.9493 |
| Macro-Specificity | 0.9752 |
| Macro-Precision | 0.9541 |
| AUC-PR (Macro) | 0.9896 |
| Cohen's Kappa | 0.9268 |
| MCC | 0.9275 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 4520.9 (0.22 ms/张) |
| End-to-End FPS | 256.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9511 | 0.9178 | 0.9415 | 0.9804 | 0.9911 | 0.9854 | 208 |
| Malignant_lesions | 0.8920 | 0.9900 | 0.9752 | 0.9318 | 0.9542 | 0.9931 | 0.9854 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9844 | 0.9694 | 0.9794 | 0.9947 | 0.9991 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=18 TN=350 FN=7
- **Malignant_lesions**: TP=157 FP=4 TN=396 FN=19
- **body_of_stomach_normal**: TP=190 FP=6 TN=378 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 4 | 3 |
| Malignant_lesions | 16 | 157 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9178    0.9663    0.9415       208
     Malignant_lesions     0.9752    0.8920    0.9318       176
body_of_stomach_normal     0.9694    0.9896    0.9794       192

              accuracy                         0.9514       576
             macro avg     0.9541    0.9493    0.9509       576
          weighted avg     0.9525    0.9514    0.9511       576

```