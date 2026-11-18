# mbv3_r6_star1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star1_seed43/mbv3_r6_star1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9622 |
| Accuracy | 0.9696 |
| Macro-AUROC (OvR) | 0.9980 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9698 |
| Balanced Acc | 0.9677 |
| Macro-Sensitivity | 0.9677 |
| Macro-Specificity | 0.9856 |
| Macro-Precision | 0.9573 |
| AUC-PR (Macro) | 0.9938 |
| Cohen's Kappa | 0.9518 |
| MCC | 0.9520 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.18M |
| FLOPs | 464.35M |
| Params | 3.47M |
| Pure-Inference FPS | 3347.9 (0.30 ms/张) |
| End-to-End FPS | 246.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9538 | 0.9828 | 0.9789 | 0.9662 | 0.9620 | 0.9979 | 0.9975 | 195 |
| Malignant_lesions | 0.9494 | 0.9741 | 0.8929 | 0.9202 | 0.9883 | 0.9962 | 0.9839 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=186 FP=4 TN=228 FN=9
- **Malignant_lesions**: TP=75 FP=9 TN=339 FN=4
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 186 | 9 | 0 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9789    0.9538    0.9662       195
     Malignant_lesions     0.8929    0.9494    0.9202        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9696       427
             macro avg     0.9573    0.9677    0.9622       427
          weighted avg     0.9706    0.9696    0.9698       427

```