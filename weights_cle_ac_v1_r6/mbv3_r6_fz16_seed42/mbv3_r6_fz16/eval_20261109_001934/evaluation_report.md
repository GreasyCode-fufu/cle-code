# mbv3_r6_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz16_seed42/mbv3_r6_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9707 |
| Accuracy | 0.9705 |
| Macro-AUROC (OvR) | 0.9956 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9705 |
| Balanced Acc | 0.9705 |
| Macro-Sensitivity | 0.9705 |
| Macro-Specificity | 0.9851 |
| Macro-Precision | 0.9710 |
| AUC-PR (Macro) | 0.9918 |
| Cohen's Kappa | 0.9556 |
| MCC | 0.9556 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4860.5 (0.21 ms/张) |
| End-to-End FPS | 88.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9755 | 0.9571 | 0.9617 | 0.9809 | 0.9926 | 0.9901 | 208 |
| Malignant_lesions | 0.9659 | 0.9875 | 0.9714 | 0.9687 | 0.9850 | 0.9955 | 0.9881 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9986 | 0.9971 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=9 TN=359 FN=7
- **Malignant_lesions**: TP=170 FP=5 TN=395 FN=6
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 5 | 170 | 1 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9571    0.9663    0.9617       208
     Malignant_lesions     0.9714    0.9659    0.9687       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9705       576
             macro avg     0.9710    0.9705    0.9707       576
          weighted avg     0.9706    0.9705    0.9705       576

```