# convnext_fgc_dsd_fse 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_dsd_fse_seed43/convnext_fgc_dsd_fse/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9371 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9933 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9375 |
| Balanced Acc | 0.9349 |
| Macro-Sensitivity | 0.9349 |
| Macro-Specificity | 0.9680 |
| Macro-Precision | 0.9416 |
| AUC-PR (Macro) | 0.9871 |
| Cohen's Kappa | 0.9058 |
| MCC | 0.9070 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 296.3 (3.38 ms/张) |
| End-to-End FPS | 152.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9293 | 0.8850 | 0.9217 | 0.9771 | 0.9908 | 0.9857 | 208 |
| Malignant_lesions | 0.8693 | 0.9800 | 0.9503 | 0.9080 | 0.9446 | 0.9900 | 0.9770 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9948 | 0.9894 | 0.9816 | 0.9871 | 0.9992 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=26 TN=342 FN=8
- **Malignant_lesions**: TP=153 FP=8 TN=392 FN=23
- **body_of_stomach_normal**: TP=187 FP=2 TN=382 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 7 | 1 |
| Malignant_lesions | 22 | 153 | 1 |
| body_of_stomach_normal | 4 | 1 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8850    0.9615    0.9217       208
     Malignant_lesions     0.9503    0.8693    0.9080       176
body_of_stomach_normal     0.9894    0.9740    0.9816       192

              accuracy                         0.9375       576
             macro avg     0.9416    0.9349    0.9371       576
          weighted avg     0.9397    0.9375    0.9375       576

```