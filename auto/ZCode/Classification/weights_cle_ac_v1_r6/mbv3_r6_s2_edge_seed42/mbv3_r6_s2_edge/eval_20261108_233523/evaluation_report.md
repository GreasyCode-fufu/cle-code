# mbv3_r6_s2_edge 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_edge_seed42/mbv3_r6_s2_edge/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9653 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9972 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9718 |
| Balanced Acc | 0.9644 |
| Macro-Sensitivity | 0.9644 |
| Macro-Specificity | 0.9854 |
| Macro-Precision | 0.9662 |
| AUC-PR (Macro) | 0.9903 |
| Cohen's Kappa | 0.9553 |
| MCC | 0.9553 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 3220.3 (0.31 ms/张) |
| End-to-End FPS | 230.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9741 | 0.9692 | 0.9692 | 0.9741 | 0.9969 | 0.9966 | 195 |
| Malignant_lesions | 0.9241 | 0.9856 | 0.9359 | 0.9299 | 0.9828 | 0.9947 | 0.9744 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=6 TN=226 FN=6
- **Malignant_lesions**: TP=73 FP=5 TN=343 FN=6
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 5 | 1 |
| Malignant_lesions | 6 | 73 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9692    0.9692    0.9692       195
     Malignant_lesions     0.9359    0.9241    0.9299        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9719       427
             macro avg     0.9662    0.9644    0.9653       427
          weighted avg     0.9718    0.9719    0.9718       427

```