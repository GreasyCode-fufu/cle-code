# mbv3_sft_b4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b4_seed44/mbv3_sft_b4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9030 |
| Accuracy | 0.9045 |
| Macro-AUROC (OvR) | 0.9907 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9037 |
| Balanced Acc | 0.9001 |
| Macro-Sensitivity | 0.9001 |
| Macro-Specificity | 0.9510 |
| Macro-Precision | 0.9137 |
| AUC-PR (Macro) | 0.9822 |
| Cohen's Kappa | 0.8558 |
| MCC | 0.8593 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3140.0 (0.32 ms/张) |
| End-to-End FPS | 157.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.8913 | 0.8305 | 0.8829 | 0.9647 | 0.9857 | 0.9769 | 208 |
| Malignant_lesions | 0.7841 | 0.9825 | 0.9517 | 0.8598 | 0.9118 | 0.9879 | 0.9728 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9792 | 0.9590 | 0.9664 | 0.9869 | 0.9983 | 0.9968 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=40 TN=328 FN=12
- **Malignant_lesions**: TP=138 FP=7 TN=393 FN=38
- **body_of_stomach_normal**: TP=187 FP=8 TN=376 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 7 | 5 |
| Malignant_lesions | 35 | 138 | 3 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8305    0.9423    0.8829       208
     Malignant_lesions     0.9517    0.7841    0.8598       176
body_of_stomach_normal     0.9590    0.9740    0.9664       192

              accuracy                         0.9045       576
             macro avg     0.9137    0.9001    0.9030       576
          weighted avg     0.9104    0.9045    0.9037       576

```