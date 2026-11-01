# convnext_fesr2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fesr2_seed43/convnext_fesr2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9475 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9947 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9479 |
| Balanced Acc | 0.9473 |
| Macro-Sensitivity | 0.9473 |
| Macro-Specificity | 0.9739 |
| Macro-Precision | 0.9476 |
| AUC-PR (Macro) | 0.9904 |
| Cohen's Kappa | 0.9217 |
| MCC | 0.9217 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 756.5 (1.32 ms/张) |
| End-to-End FPS | 218.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9674 | 0.9426 | 0.9448 | 0.9700 | 0.9944 | 0.9903 | 208 |
| Malignant_lesions | 0.9261 | 0.9700 | 0.9314 | 0.9288 | 0.9676 | 0.9909 | 0.9831 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9844 | 0.9688 | 0.9688 | 0.9844 | 0.9989 | 0.9978 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=12 TN=356 FN=11
- **Malignant_lesions**: TP=163 FP=12 TN=388 FN=13
- **body_of_stomach_normal**: TP=186 FP=6 TN=378 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 8 | 163 | 5 |
| body_of_stomach_normal | 4 | 2 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9426    0.9471    0.9448       208
     Malignant_lesions     0.9314    0.9261    0.9288       176
body_of_stomach_normal     0.9688    0.9688    0.9688       192

              accuracy                         0.9479       576
             macro avg     0.9476    0.9473    0.9475       576
          weighted avg     0.9479    0.9479    0.9479       576

```