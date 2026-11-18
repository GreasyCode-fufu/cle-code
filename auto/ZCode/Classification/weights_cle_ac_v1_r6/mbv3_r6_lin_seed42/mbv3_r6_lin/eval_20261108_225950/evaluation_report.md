# mbv3_r6_lin 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_lin_seed42/mbv3_r6_lin/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9114 |
| Accuracy | 0.9132 |
| Macro-AUROC (OvR) | 0.9898 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9121 |
| Balanced Acc | 0.9097 |
| Macro-Sensitivity | 0.9097 |
| Macro-Specificity | 0.9558 |
| Macro-Precision | 0.9180 |
| AUC-PR (Macro) | 0.9814 |
| Cohen's Kappa | 0.8691 |
| MCC | 0.8711 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.97M |
| Pure-Inference FPS | 4340.9 (0.23 ms/张) |
| End-to-End FPS | 225.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9239 | 0.8744 | 0.9049 | 0.9632 | 0.9841 | 0.9754 | 208 |
| Malignant_lesions | 0.8125 | 0.9825 | 0.9533 | 0.8773 | 0.9225 | 0.9874 | 0.9731 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9609 | 0.9261 | 0.9519 | 0.9893 | 0.9979 | 0.9956 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=28 TN=340 FN=13
- **Malignant_lesions**: TP=143 FP=7 TN=393 FN=33
- **body_of_stomach_normal**: TP=188 FP=15 TN=369 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 7 | 6 |
| Malignant_lesions | 24 | 143 | 9 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8744    0.9375    0.9049       208
     Malignant_lesions     0.9533    0.8125    0.8773       176
body_of_stomach_normal     0.9261    0.9792    0.9519       192

              accuracy                         0.9132       576
             macro avg     0.9180    0.9097    0.9114       576
          weighted avg     0.9158    0.9132    0.9121       576

```