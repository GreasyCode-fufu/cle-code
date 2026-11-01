# convnext_dph_fesr2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_fesr2_seed44/convnext_dph_fesr2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9486 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9492 |
| Balanced Acc | 0.9471 |
| Macro-Sensitivity | 0.9471 |
| Macro-Specificity | 0.9743 |
| Macro-Precision | 0.9523 |
| AUC-PR (Macro) | 0.9906 |
| Cohen's Kappa | 0.9241 |
| MCC | 0.9251 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 455.5 (2.20 ms/张) |
| End-to-End FPS | 184.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9538 | 0.9224 | 0.9461 | 0.9832 | 0.9921 | 0.9884 | 208 |
| Malignant_lesions | 0.8807 | 0.9900 | 0.9748 | 0.9254 | 0.9496 | 0.9939 | 0.9856 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9792 | 0.9596 | 0.9744 | 0.9947 | 0.9988 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=17 TN=351 FN=6
- **Malignant_lesions**: TP=155 FP=4 TN=396 FN=21
- **body_of_stomach_normal**: TP=190 FP=8 TN=376 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 15 | 155 | 6 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9224    0.9712    0.9461       208
     Malignant_lesions     0.9748    0.8807    0.9254       176
body_of_stomach_normal     0.9596    0.9896    0.9744       192

              accuracy                         0.9497       576
             macro avg     0.9523    0.9471    0.9486       576
          weighted avg     0.9508    0.9497    0.9492       576

```