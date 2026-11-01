# convnext_fesr2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fesr2_seed44/convnext_fesr2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9478 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9482 |
| Balanced Acc | 0.9479 |
| Macro-Sensitivity | 0.9479 |
| Macro-Specificity | 0.9740 |
| Macro-Precision | 0.9487 |
| AUC-PR (Macro) | 0.9904 |
| Cohen's Kappa | 0.9217 |
| MCC | 0.9222 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1326.7 (0.75 ms/张) |
| End-to-End FPS | 257.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9647 | 0.9387 | 0.9476 | 0.9753 | 0.9928 | 0.9879 | 208 |
| Malignant_lesions | 0.9545 | 0.9600 | 0.9130 | 0.9333 | 0.9796 | 0.9930 | 0.9853 | 176 |
| body_of_stomach_normal | 0.9323 | 0.9974 | 0.9944 | 0.9624 | 0.9672 | 0.9989 | 0.9981 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=13 TN=355 FN=9
- **Malignant_lesions**: TP=168 FP=16 TN=384 FN=8
- **body_of_stomach_normal**: TP=179 FP=1 TN=383 FN=13

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 8 | 168 | 0 |
| body_of_stomach_normal | 5 | 8 | 179 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9387    0.9567    0.9476       208
     Malignant_lesions     0.9130    0.9545    0.9333       176
body_of_stomach_normal     0.9944    0.9323    0.9624       192

              accuracy                         0.9479       576
             macro avg     0.9487    0.9479    0.9478       576
          weighted avg     0.9494    0.9479    0.9482       576

```