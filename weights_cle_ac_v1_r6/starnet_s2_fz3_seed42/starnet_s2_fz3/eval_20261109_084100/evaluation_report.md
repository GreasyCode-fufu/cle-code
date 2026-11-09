# starnet_s2_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/starnet_s2_fz3_seed42/starnet_s2_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9172 |
| Accuracy | 0.9184 |
| Macro-AUROC (OvR) | 0.9854 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9177 |
| Balanced Acc | 0.9167 |
| Macro-Sensitivity | 0.9167 |
| Macro-Specificity | 0.9588 |
| Macro-Precision | 0.9189 |
| AUC-PR (Macro) | 0.9715 |
| Cohen's Kappa | 0.8772 |
| MCC | 0.8777 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 549.29M |
| FLOPs | 1.10G |
| Params | 3.43M |
| Pure-Inference FPS | 3544.2 (0.28 ms/张) |
| End-to-End FPS | 26.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9135 | 0.9402 | 0.8962 | 0.9048 | 0.9505 | 0.9782 | 0.9691 | 208 |
| Malignant_lesions | 0.8523 | 0.9675 | 0.9202 | 0.8850 | 0.9370 | 0.9795 | 0.9482 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9688 | 0.9403 | 0.9618 | 0.9920 | 0.9985 | 0.9971 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=190 FP=22 TN=346 FN=18
- **Malignant_lesions**: TP=150 FP=13 TN=387 FN=26
- **body_of_stomach_normal**: TP=189 FP=12 TN=372 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 190 | 11 | 7 |
| Malignant_lesions | 21 | 150 | 5 |
| body_of_stomach_normal | 1 | 2 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8962    0.9135    0.9048       208
     Malignant_lesions     0.9202    0.8523    0.8850       176
body_of_stomach_normal     0.9403    0.9844    0.9618       192

              accuracy                         0.9184       576
             macro avg     0.9189    0.9167    0.9172       576
          weighted avg     0.9183    0.9184    0.9177       576

```