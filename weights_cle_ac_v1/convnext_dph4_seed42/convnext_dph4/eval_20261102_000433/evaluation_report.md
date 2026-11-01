# convnext_dph4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph4_seed42/convnext_dph4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9492 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9495 |
| Balanced Acc | 0.9488 |
| Macro-Sensitivity | 0.9488 |
| Macro-Specificity | 0.9746 |
| Macro-Precision | 0.9498 |
| AUC-PR (Macro) | 0.9925 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9243 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 482.6 (2.07 ms/张) |
| End-to-End FPS | 186.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9620 | 0.9336 | 0.9403 | 0.9699 | 0.9949 | 0.9914 | 208 |
| Malignant_lesions | 0.9148 | 0.9750 | 0.9415 | 0.9280 | 0.9630 | 0.9940 | 0.9873 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9870 | 0.9742 | 0.9793 | 0.9921 | 0.9994 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=14 TN=354 FN=11
- **Malignant_lesions**: TP=161 FP=10 TN=390 FN=15
- **body_of_stomach_normal**: TP=189 FP=5 TN=379 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 12 | 161 | 3 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9336    0.9471    0.9403       208
     Malignant_lesions     0.9415    0.9148    0.9280       176
body_of_stomach_normal     0.9742    0.9844    0.9793       192

              accuracy                         0.9497       576
             macro avg     0.9498    0.9488    0.9492       576
          weighted avg     0.9496    0.9497    0.9495       576

```