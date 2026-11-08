# mbv3_r6_mlp256 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_mlp256_seed42/mbv3_r6_mlp256/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9155 |
| Accuracy | 0.9184 |
| Macro-AUROC (OvR) | 0.9926 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9167 |
| Balanced Acc | 0.9133 |
| Macro-Sensitivity | 0.9133 |
| Macro-Specificity | 0.9581 |
| Macro-Precision | 0.9266 |
| AUC-PR (Macro) | 0.9853 |
| Cohen's Kappa | 0.8768 |
| MCC | 0.8806 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.93M |
| FLOPs | 463.85M |
| Params | 3.22M |
| Pure-Inference FPS | 4168.9 (0.24 ms/张) |
| End-to-End FPS | 234.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9130 | 0.8627 | 0.9116 | 0.9796 | 0.9881 | 0.9822 | 208 |
| Malignant_lesions | 0.7841 | 0.9900 | 0.9718 | 0.8679 | 0.9124 | 0.9904 | 0.9750 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9714 | 0.9453 | 0.9669 | 0.9947 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=32 TN=336 FN=7
- **Malignant_lesions**: TP=138 FP=4 TN=396 FN=38
- **body_of_stomach_normal**: TP=190 FP=11 TN=373 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 4 | 3 |
| Malignant_lesions | 30 | 138 | 8 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8627    0.9663    0.9116       208
     Malignant_lesions     0.9718    0.7841    0.8679       176
body_of_stomach_normal     0.9453    0.9896    0.9669       192

              accuracy                         0.9184       576
             macro avg     0.9266    0.9133    0.9155       576
          weighted avg     0.9236    0.9184    0.9167       576

```