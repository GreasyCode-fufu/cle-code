# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_small_seed43/convnext_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9528 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9930 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9532 |
| Balanced Acc | 0.9533 |
| Macro-Sensitivity | 0.9533 |
| Macro-Specificity | 0.9767 |
| Macro-Precision | 0.9524 |
| AUC-PR (Macro) | 0.9861 |
| Cohen's Kappa | 0.9296 |
| MCC | 0.9297 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 382.1 (2.62 ms/张) |
| End-to-End FPS | 169.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9755 | 0.9559 | 0.9466 | 0.9651 | 0.9908 | 0.9769 | 208 |
| Malignant_lesions | 0.9432 | 0.9625 | 0.9171 | 0.9300 | 0.9747 | 0.9892 | 0.9832 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9990 | 0.9982 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=9 TN=359 FN=13
- **Malignant_lesions**: TP=166 FP=15 TN=385 FN=10
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 13 | 0 |
| Malignant_lesions | 7 | 166 | 3 |
| body_of_stomach_normal | 2 | 2 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9559    0.9375    0.9466       208
     Malignant_lesions     0.9171    0.9432    0.9300       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9531       576
             macro avg     0.9524    0.9533    0.9528       576
          weighted avg     0.9535    0.9531    0.9532       576

```