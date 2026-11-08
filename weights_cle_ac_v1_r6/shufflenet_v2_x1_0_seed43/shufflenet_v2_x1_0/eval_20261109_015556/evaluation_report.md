# shufflenet_v2_x1_0 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/shufflenet_v2_x1_0_seed43/shufflenet_v2_x1_0/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9259 |
| Accuracy | 0.9271 |
| Macro-AUROC (OvR) | 0.9829 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9266 |
| Balanced Acc | 0.9243 |
| Macro-Sensitivity | 0.9243 |
| Macro-Specificity | 0.9629 |
| Macro-Precision | 0.9299 |
| AUC-PR (Macro) | 0.9672 |
| Cohen's Kappa | 0.8901 |
| MCC | 0.8912 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 151.69M |
| FLOPs | 303.38M |
| Params | 1.26M |
| Pure-Inference FPS | 5822.9 (0.17 ms/张) |
| End-to-End FPS | 28.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9293 | 0.8834 | 0.9142 | 0.9688 | 0.9789 | 0.9468 | 208 |
| Malignant_lesions | 0.8466 | 0.9750 | 0.9371 | 0.8896 | 0.9353 | 0.9711 | 0.9572 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9844 | 0.9691 | 0.9741 | 0.9895 | 0.9987 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=26 TN=342 FN=11
- **Malignant_lesions**: TP=149 FP=10 TN=390 FN=27
- **body_of_stomach_normal**: TP=188 FP=6 TN=378 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 23 | 149 | 4 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8834    0.9471    0.9142       208
     Malignant_lesions     0.9371    0.8466    0.8896       176
body_of_stomach_normal     0.9691    0.9792    0.9741       192

              accuracy                         0.9271       576
             macro avg     0.9299    0.9243    0.9259       576
          weighted avg     0.9284    0.9271    0.9266       576

```