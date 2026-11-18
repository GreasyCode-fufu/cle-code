# convnext_v2bc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2bc_seed44/convnext_v2bc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9515 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9955 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9515 |
| Balanced Acc | 0.9504 |
| Macro-Sensitivity | 0.9504 |
| Macro-Specificity | 0.9753 |
| Macro-Precision | 0.9531 |
| AUC-PR (Macro) | 0.9919 |
| Cohen's Kappa | 0.9268 |
| MCC | 0.9271 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 28.67M |
| Pure-Inference FPS | 1455.8 (0.69 ms/张) |
| End-to-End FPS | 259.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9511 | 0.9171 | 0.9365 | 0.9749 | 0.9928 | 0.9891 | 208 |
| Malignant_lesions | 0.9205 | 0.9800 | 0.9529 | 0.9364 | 0.9655 | 0.9943 | 0.9877 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9948 | 0.9894 | 0.9816 | 0.9871 | 0.9994 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=18 TN=350 FN=9
- **Malignant_lesions**: TP=162 FP=8 TN=392 FN=14
- **body_of_stomach_normal**: TP=187 FP=2 TN=382 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 13 | 162 | 1 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9171    0.9567    0.9365       208
     Malignant_lesions     0.9529    0.9205    0.9364       176
body_of_stomach_normal     0.9894    0.9740    0.9816       192

              accuracy                         0.9514       576
             macro avg     0.9531    0.9504    0.9515       576
          weighted avg     0.9521    0.9514    0.9515       576

```