# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz16bn_seed45/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9320 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9902 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9325 |
| Balanced Acc | 0.9299 |
| Macro-Sensitivity | 0.9299 |
| Macro-Specificity | 0.9655 |
| Macro-Precision | 0.9363 |
| AUC-PR (Macro) | 0.9816 |
| Cohen's Kappa | 0.8980 |
| MCC | 0.8992 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4721.3 (0.21 ms/张) |
| End-to-End FPS | 218.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9293 | 0.8860 | 0.9266 | 0.9828 | 0.9862 | 0.9780 | 208 |
| Malignant_lesions | 0.8864 | 0.9725 | 0.9341 | 0.9096 | 0.9511 | 0.9859 | 0.9697 | 176 |
| body_of_stomach_normal | 0.9323 | 0.9948 | 0.9890 | 0.9598 | 0.9671 | 0.9985 | 0.9971 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=26 TN=342 FN=6
- **Malignant_lesions**: TP=156 FP=11 TN=389 FN=20
- **body_of_stomach_normal**: TP=179 FP=2 TN=382 FN=13

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 5 | 1 |
| Malignant_lesions | 19 | 156 | 1 |
| body_of_stomach_normal | 7 | 6 | 179 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8860    0.9712    0.9266       208
     Malignant_lesions     0.9341    0.8864    0.9096       176
body_of_stomach_normal     0.9890    0.9323    0.9598       192

              accuracy                         0.9323       576
             macro avg     0.9363    0.9299    0.9320       576
          weighted avg     0.9350    0.9323    0.9325       576

```