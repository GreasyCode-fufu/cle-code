# convnext_tiny_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_dce4m_seed45/convnext_tiny_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9633 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9979 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9635 |
| Balanced Acc | 0.9620 |
| Macro-Sensitivity | 0.9620 |
| Macro-Specificity | 0.9814 |
| Macro-Precision | 0.9654 |
| AUC-PR (Macro) | 0.9960 |
| Cohen's Kappa | 0.9451 |
| MCC | 0.9455 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1549.5 (0.65 ms/张) |
| End-to-End FPS | 8.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9808 | 0.9620 | 0.9358 | 0.9577 | 0.9888 | 0.9970 | 0.9950 | 208 |
| Malignant_lesions | 0.9261 | 0.9900 | 0.9760 | 0.9504 | 0.9682 | 0.9970 | 0.9938 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9997 | 0.9993 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=204 FP=14 TN=354 FN=4
- **Malignant_lesions**: TP=163 FP=4 TN=396 FN=13
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 204 | 4 | 0 |
| Malignant_lesions | 10 | 163 | 3 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9358    0.9808    0.9577       208
     Malignant_lesions     0.9760    0.9261    0.9504       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9635       576
             macro avg     0.9654    0.9620    0.9633       576
          weighted avg     0.9643    0.9635    0.9635       576

```