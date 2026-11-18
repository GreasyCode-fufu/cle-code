# convnext_small 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_small_seed42/convnext_small/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9492 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9948 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9496 |
| Balanced Acc | 0.9499 |
| Macro-Sensitivity | 0.9499 |
| Macro-Specificity | 0.9750 |
| Macro-Precision | 0.9488 |
| AUC-PR (Macro) | 0.9901 |
| Cohen's Kappa | 0.9244 |
| MCC | 0.9245 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 8.68G |
| FLOPs | 17.37G |
| Params | 49.41M |
| Pure-Inference FPS | 417.5 (2.40 ms/张) |
| End-to-End FPS | 177.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9783 | 0.9602 | 0.9438 | 0.9600 | 0.9942 | 0.9900 | 208 |
| Malignant_lesions | 0.9375 | 0.9650 | 0.9218 | 0.9296 | 0.9723 | 0.9916 | 0.9826 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9818 | 0.9643 | 0.9742 | 0.9921 | 0.9988 | 0.9976 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=8 TN=360 FN=15
- **Malignant_lesions**: TP=165 FP=14 TN=386 FN=11
- **body_of_stomach_normal**: TP=189 FP=7 TN=377 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 12 | 3 |
| Malignant_lesions | 7 | 165 | 4 |
| body_of_stomach_normal | 1 | 2 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9602    0.9279    0.9438       208
     Malignant_lesions     0.9218    0.9375    0.9296       176
body_of_stomach_normal     0.9643    0.9844    0.9742       192

              accuracy                         0.9497       576
             macro avg     0.9488    0.9499    0.9492       576
          weighted avg     0.9498    0.9497    0.9496       576

```