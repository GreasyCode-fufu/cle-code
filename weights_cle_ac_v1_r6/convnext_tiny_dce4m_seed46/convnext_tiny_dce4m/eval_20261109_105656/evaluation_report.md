# convnext_tiny_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_dce4m_seed46/convnext_tiny_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9647 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9959 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9651 |
| Balanced Acc | 0.9637 |
| Macro-Sensitivity | 0.9637 |
| Macro-Specificity | 0.9824 |
| Macro-Precision | 0.9663 |
| AUC-PR (Macro) | 0.9922 |
| Cohen's Kappa | 0.9477 |
| MCC | 0.9480 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1578.4 (0.63 ms/张) |
| End-to-End FPS | 7.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9674 | 0.9442 | 0.9598 | 0.9861 | 0.9942 | 0.9894 | 208 |
| Malignant_lesions | 0.9205 | 0.9875 | 0.9701 | 0.9446 | 0.9658 | 0.9938 | 0.9879 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9922 | 0.9845 | 0.9896 | 0.9974 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=12 TN=356 FN=5
- **Malignant_lesions**: TP=162 FP=5 TN=395 FN=14
- **body_of_stomach_normal**: TP=191 FP=3 TN=381 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 5 | 0 |
| Malignant_lesions | 11 | 162 | 3 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9442    0.9760    0.9598       208
     Malignant_lesions     0.9701    0.9205    0.9446       176
body_of_stomach_normal     0.9845    0.9948    0.9896       192

              accuracy                         0.9653       576
             macro avg     0.9663    0.9637    0.9647       576
          weighted avg     0.9655    0.9653    0.9651       576

```