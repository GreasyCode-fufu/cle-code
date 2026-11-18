# thinkingvit_deit 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/thinkingvit_deit/seed43/thinkingvit_deit/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9530 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9928 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9531 |
| Balanced Acc | 0.9528 |
| Macro-Sensitivity | 0.9528 |
| Macro-Specificity | 0.9765 |
| Macro-Precision | 0.9531 |
| AUC-PR (Macro) | 0.9862 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9295 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 130.13M |
| FLOPs | 260.26M |
| Params | 369.41K |
| Pure-Inference FPS | 1792.8 (0.56 ms/张) |
| End-to-End FPS | 233.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9647 | 0.9378 | 0.9400 | 0.9673 | 0.9900 | 0.9833 | 208 |
| Malignant_lesions | 0.9318 | 0.9725 | 0.9371 | 0.9345 | 0.9701 | 0.9890 | 0.9763 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9922 | 0.9844 | 0.9844 | 0.9922 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=13 TN=355 FN=12
- **Malignant_lesions**: TP=164 FP=11 TN=389 FN=12
- **body_of_stomach_normal**: TP=189 FP=3 TN=381 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 10 | 164 | 2 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9378    0.9423    0.9400       208
     Malignant_lesions     0.9371    0.9318    0.9345       176
body_of_stomach_normal     0.9844    0.9844    0.9844       192

              accuracy                         0.9531       576
             macro avg     0.9531    0.9528    0.9530       576
          weighted avg     0.9531    0.9531    0.9531       576

```