# convnext_v2a 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2a_seed43/convnext_v2a/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9531 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9928 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9532 |
| Balanced Acc | 0.9535 |
| Macro-Sensitivity | 0.9535 |
| Macro-Specificity | 0.9766 |
| Macro-Precision | 0.9530 |
| AUC-PR (Macro) | 0.9870 |
| Cohen's Kappa | 0.9296 |
| MCC | 0.9296 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.47G |
| FLOPs | 8.95G |
| Params | 27.81M |
| Pure-Inference FPS | 870.6 (1.15 ms/张) |
| End-to-End FPS | 196.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9701 | 0.9469 | 0.9446 | 0.9675 | 0.9912 | 0.9849 | 208 |
| Malignant_lesions | 0.9545 | 0.9675 | 0.9282 | 0.9412 | 0.9797 | 0.9882 | 0.9783 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9922 | 0.9840 | 0.9737 | 0.9820 | 0.9989 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=11 TN=357 FN=12
- **Malignant_lesions**: TP=168 FP=13 TN=387 FN=8
- **body_of_stomach_normal**: TP=185 FP=3 TN=381 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 10 | 2 |
| Malignant_lesions | 7 | 168 | 1 |
| body_of_stomach_normal | 4 | 3 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9469    0.9423    0.9446       208
     Malignant_lesions     0.9282    0.9545    0.9412       176
body_of_stomach_normal     0.9840    0.9635    0.9737       192

              accuracy                         0.9531       576
             macro avg     0.9530    0.9535    0.9531       576
          weighted avg     0.9535    0.9531    0.9532       576

```