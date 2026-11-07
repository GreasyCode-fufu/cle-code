# convnext_v3x 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v3x_seed43/convnext_v3x/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9568 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9955 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9567 |
| Balanced Acc | 0.9566 |
| Macro-Sensitivity | 0.9566 |
| Macro-Specificity | 0.9782 |
| Macro-Precision | 0.9570 |
| AUC-PR (Macro) | 0.9923 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9348 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.28G |
| Params | 29.11M |
| Pure-Inference FPS | 748.2 (1.34 ms/张) |
| End-to-End FPS | 223.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9647 | 0.9381 | 0.9426 | 0.9699 | 0.9946 | 0.9903 | 208 |
| Malignant_lesions | 0.9489 | 0.9725 | 0.9382 | 0.9435 | 0.9774 | 0.9957 | 0.9902 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9974 | 0.9947 | 0.9842 | 0.9871 | 0.9962 | 0.9963 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=13 TN=355 FN=11
- **Malignant_lesions**: TP=167 FP=11 TN=389 FN=9
- **body_of_stomach_normal**: TP=187 FP=1 TN=383 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 11 | 0 |
| Malignant_lesions | 8 | 167 | 1 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9381    0.9471    0.9426       208
     Malignant_lesions     0.9382    0.9489    0.9435       176
body_of_stomach_normal     0.9947    0.9740    0.9842       192

              accuracy                         0.9566       576
             macro avg     0.9570    0.9566    0.9568       576
          weighted avg     0.9570    0.9566    0.9567       576

```