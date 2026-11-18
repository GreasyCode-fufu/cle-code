# convnext_fgc_g8 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_g8_seed42/convnext_fgc_g8/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9505 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9963 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9511 |
| Balanced Acc | 0.9491 |
| Macro-Sensitivity | 0.9491 |
| Macro-Specificity | 0.9753 |
| Macro-Precision | 0.9534 |
| AUC-PR (Macro) | 0.9929 |
| Cohen's Kappa | 0.9268 |
| MCC | 0.9275 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 628.4 (1.59 ms/张) |
| End-to-End FPS | 206.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9565 | 0.9269 | 0.9508 | 0.9860 | 0.9965 | 0.9944 | 208 |
| Malignant_lesions | 0.8920 | 0.9875 | 0.9691 | 0.9290 | 0.9541 | 0.9938 | 0.9868 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9818 | 0.9641 | 0.9716 | 0.9895 | 0.9987 | 0.9974 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=16 TN=352 FN=5
- **Malignant_lesions**: TP=157 FP=5 TN=395 FN=19
- **body_of_stomach_normal**: TP=188 FP=7 TN=377 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 5 | 0 |
| Malignant_lesions | 12 | 157 | 7 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9269    0.9760    0.9508       208
     Malignant_lesions     0.9691    0.8920    0.9290       176
body_of_stomach_normal     0.9641    0.9792    0.9716       192

              accuracy                         0.9514       576
             macro avg     0.9534    0.9491    0.9505       576
          weighted avg     0.9522    0.9514    0.9511       576

```