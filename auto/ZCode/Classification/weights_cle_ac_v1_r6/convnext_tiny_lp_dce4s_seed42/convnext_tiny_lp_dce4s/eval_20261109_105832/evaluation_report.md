# convnext_tiny_lp_dce4s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_lp_dce4s_seed42/convnext_tiny_lp_dce4s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9496 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9955 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9495 |
| Balanced Acc | 0.9490 |
| Macro-Sensitivity | 0.9490 |
| Macro-Specificity | 0.9745 |
| Macro-Precision | 0.9509 |
| AUC-PR (Macro) | 0.9917 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9246 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1693.9 (0.59 ms/张) |
| End-to-End FPS | 242.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9674 | 0.9423 | 0.9423 | 0.9674 | 0.9929 | 0.9888 | 208 |
| Malignant_lesions | 0.9205 | 0.9875 | 0.9701 | 0.9446 | 0.9658 | 0.9958 | 0.9909 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9688 | 0.9403 | 0.9618 | 0.9920 | 0.9978 | 0.9956 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=12 TN=356 FN=12
- **Malignant_lesions**: TP=162 FP=5 TN=395 FN=14
- **body_of_stomach_normal**: TP=189 FP=12 TN=372 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 5 | 7 |
| Malignant_lesions | 9 | 162 | 5 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9423    0.9423    0.9423       208
     Malignant_lesions     0.9701    0.9205    0.9446       176
body_of_stomach_normal     0.9403    0.9844    0.9618       192

              accuracy                         0.9497       576
             macro avg     0.9509    0.9490    0.9496       576
          weighted avg     0.9501    0.9497    0.9495       576

```