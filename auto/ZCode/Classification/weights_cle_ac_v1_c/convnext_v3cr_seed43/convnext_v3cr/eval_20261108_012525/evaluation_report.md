# convnext_v3cr 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v3cr_seed43/convnext_v3cr/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9495 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9947 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9498 |
| Balanced Acc | 0.9495 |
| Macro-Sensitivity | 0.9495 |
| Macro-Specificity | 0.9748 |
| Macro-Precision | 0.9496 |
| AUC-PR (Macro) | 0.9901 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9244 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.64G |
| FLOPs | 9.27G |
| Params | 28.26M |
| Pure-Inference FPS | 1594.0 (0.63 ms/张) |
| End-to-End FPS | 263.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9647 | 0.9378 | 0.9400 | 0.9673 | 0.9921 | 0.9880 | 208 |
| Malignant_lesions | 0.9375 | 0.9650 | 0.9218 | 0.9296 | 0.9723 | 0.9931 | 0.9844 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9948 | 0.9894 | 0.9789 | 0.9845 | 0.9990 | 0.9979 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=13 TN=355 FN=12
- **Malignant_lesions**: TP=165 FP=14 TN=386 FN=11
- **body_of_stomach_normal**: TP=186 FP=2 TN=382 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 10 | 165 | 1 |
| body_of_stomach_normal | 3 | 3 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9378    0.9423    0.9400       208
     Malignant_lesions     0.9218    0.9375    0.9296       176
body_of_stomach_normal     0.9894    0.9688    0.9789       192

              accuracy                         0.9497       576
             macro avg     0.9496    0.9495    0.9495       576
          weighted avg     0.9501    0.9497    0.9498       576

```