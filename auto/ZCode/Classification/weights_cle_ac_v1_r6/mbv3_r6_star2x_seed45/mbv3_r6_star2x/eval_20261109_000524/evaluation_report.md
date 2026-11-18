# mbv3_r6_star2x 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2x_seed45/mbv3_r6_star2x/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9223 |
| Accuracy | 0.9236 |
| Macro-AUROC (OvR) | 0.9863 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9230 |
| Balanced Acc | 0.9190 |
| Macro-Sensitivity | 0.9190 |
| Macro-Specificity | 0.9606 |
| Macro-Precision | 0.9330 |
| AUC-PR (Macro) | 0.9725 |
| Cohen's Kappa | 0.8847 |
| MCC | 0.8882 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.10M |
| FLOPs | 464.20M |
| Params | 3.39M |
| Pure-Inference FPS | 4348.2 (0.23 ms/张) |
| End-to-End FPS | 70.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9022 | 0.8487 | 0.9058 | 0.9822 | 0.9797 | 0.9585 | 208 |
| Malignant_lesions | 0.8068 | 0.9875 | 0.9660 | 0.8793 | 0.9207 | 0.9805 | 0.9613 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9988 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=36 TN=332 FN=6
- **Malignant_lesions**: TP=142 FP=5 TN=395 FN=34
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 33 | 142 | 1 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8487    0.9712    0.9058       208
     Malignant_lesions     0.9660    0.8068    0.8793       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9236       576
             macro avg     0.9330    0.9190    0.9223       576
          weighted avg     0.9297    0.9236    0.9230       576

```