# cpubone_b2 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/cpubone_b2/seed43/cpubone_b2/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9068 |
| Accuracy | 0.9097 |
| Macro-AUROC (OvR) | 0.9868 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9081 |
| Balanced Acc | 0.9055 |
| Macro-Sensitivity | 0.9055 |
| Macro-Specificity | 0.9541 |
| Macro-Precision | 0.9138 |
| AUC-PR (Macro) | 0.9754 |
| Cohen's Kappa | 0.8638 |
| MCC | 0.8662 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.36G |
| FLOPs | 2.72G |
| Params | 21.31M |
| Pure-Inference FPS | 2825.4 (0.35 ms/张) |
| End-to-End FPS | 239.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9212 | 0.8711 | 0.9053 | 0.9658 | 0.9831 | 0.9733 | 208 |
| Malignant_lesions | 0.7898 | 0.9775 | 0.9392 | 0.8580 | 0.9136 | 0.9798 | 0.9580 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9635 | 0.9310 | 0.9570 | 0.9920 | 0.9974 | 0.9947 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=29 TN=339 FN=12
- **Malignant_lesions**: TP=139 FP=9 TN=391 FN=37
- **body_of_stomach_normal**: TP=189 FP=14 TN=370 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 8 | 4 |
| Malignant_lesions | 27 | 139 | 10 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8711    0.9423    0.9053       208
     Malignant_lesions     0.9392    0.7898    0.8580       176
body_of_stomach_normal     0.9310    0.9844    0.9570       192

              accuracy                         0.9097       576
             macro avg     0.9138    0.9055    0.9068       576
          weighted avg     0.9119    0.9097    0.9081       576

```