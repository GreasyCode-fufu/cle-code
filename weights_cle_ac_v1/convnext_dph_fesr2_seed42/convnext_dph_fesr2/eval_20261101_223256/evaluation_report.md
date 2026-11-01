# convnext_dph_fesr2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_fesr2_seed42/convnext_dph_fesr2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9561 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9947 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9565 |
| Balanced Acc | 0.9566 |
| Macro-Sensitivity | 0.9566 |
| Macro-Specificity | 0.9784 |
| Macro-Precision | 0.9559 |
| AUC-PR (Macro) | 0.9894 |
| Cohen's Kappa | 0.9348 |
| MCC | 0.9349 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 725.3 (1.38 ms/张) |
| End-to-End FPS | 213.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9783 | 0.9606 | 0.9489 | 0.9651 | 0.9929 | 0.9889 | 208 |
| Malignant_lesions | 0.9375 | 0.9725 | 0.9375 | 0.9375 | 0.9725 | 0.9922 | 0.9811 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9844 | 0.9695 | 0.9820 | 0.9974 | 0.9989 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=8 TN=360 FN=13
- **Malignant_lesions**: TP=165 FP=11 TN=389 FN=11
- **body_of_stomach_normal**: TP=191 FP=6 TN=378 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 11 | 2 |
| Malignant_lesions | 7 | 165 | 4 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9606    0.9375    0.9489       208
     Malignant_lesions     0.9375    0.9375    0.9375       176
body_of_stomach_normal     0.9695    0.9948    0.9820       192

              accuracy                         0.9566       576
             macro avg     0.9559    0.9566    0.9561       576
          weighted avg     0.9565    0.9566    0.9565       576

```