# convnext_v3cs 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v3cs_seed43/convnext_v3cs/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9634 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9969 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9635 |
| Balanced Acc | 0.9630 |
| Macro-Sensitivity | 0.9630 |
| Macro-Specificity | 0.9816 |
| Macro-Precision | 0.9639 |
| AUC-PR (Macro) | 0.9939 |
| Cohen's Kappa | 0.9452 |
| MCC | 0.9452 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.65G |
| FLOPs | 9.29G |
| Params | 28.40M |
| Pure-Inference FPS | 540.3 (1.85 ms/张) |
| End-to-End FPS | 188.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9728 | 0.9524 | 0.9569 | 0.9781 | 0.9956 | 0.9931 | 208 |
| Malignant_lesions | 0.9432 | 0.9850 | 0.9651 | 0.9540 | 0.9752 | 0.9959 | 0.9903 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9870 | 0.9742 | 0.9793 | 0.9921 | 0.9991 | 0.9984 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=10 TN=358 FN=8
- **Malignant_lesions**: TP=166 FP=6 TN=394 FN=10
- **body_of_stomach_normal**: TP=189 FP=5 TN=379 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 7 | 166 | 3 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9524    0.9615    0.9569       208
     Malignant_lesions     0.9651    0.9432    0.9540       176
body_of_stomach_normal     0.9742    0.9844    0.9793       192

              accuracy                         0.9635       576
             macro avg     0.9639    0.9630    0.9634       576
          weighted avg     0.9636    0.9635    0.9635       576

```