# convnext_tiny_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_dce4m_seed43/convnext_tiny_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9426 |
| Accuracy | 0.9427 |
| Macro-AUROC (OvR) | 0.9934 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9427 |
| Balanced Acc | 0.9425 |
| Macro-Sensitivity | 0.9425 |
| Macro-Specificity | 0.9712 |
| Macro-Precision | 0.9426 |
| AUC-PR (Macro) | 0.9875 |
| Cohen's Kappa | 0.9139 |
| MCC | 0.9139 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1526.6 (0.66 ms/张) |
| End-to-End FPS | 8.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9620 | 0.9327 | 0.9327 | 0.9620 | 0.9914 | 0.9847 | 208 |
| Malignant_lesions | 0.9261 | 0.9700 | 0.9314 | 0.9288 | 0.9676 | 0.9913 | 0.9822 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9818 | 0.9637 | 0.9662 | 0.9843 | 0.9976 | 0.9955 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=14 TN=354 FN=14
- **Malignant_lesions**: TP=163 FP=12 TN=388 FN=13
- **body_of_stomach_normal**: TP=186 FP=7 TN=377 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 11 | 3 |
| Malignant_lesions | 9 | 163 | 4 |
| body_of_stomach_normal | 5 | 1 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9327    0.9327    0.9327       208
     Malignant_lesions     0.9314    0.9261    0.9288       176
body_of_stomach_normal     0.9637    0.9688    0.9662       192

              accuracy                         0.9427       576
             macro avg     0.9426    0.9425    0.9426       576
          weighted avg     0.9427    0.9427    0.9427       576

```