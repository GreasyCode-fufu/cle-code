# mambaout_tiny 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/mambaout_tiny/seed43/mambaout_tiny/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9706 |
| Accuracy | 0.9705 |
| Macro-AUROC (OvR) | 0.9974 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9705 |
| Balanced Acc | 0.9706 |
| Macro-Sensitivity | 0.9706 |
| Macro-Specificity | 0.9851 |
| Macro-Precision | 0.9707 |
| AUC-PR (Macro) | 0.9951 |
| Cohen's Kappa | 0.9556 |
| MCC | 0.9556 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.48G |
| FLOPs | 8.96G |
| Params | 24.25M |
| Pure-Inference FPS | 367.4 (2.72 ms/张) |
| End-to-End FPS | 15.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9755 | 0.9569 | 0.9592 | 0.9782 | 0.9962 | 0.9928 | 208 |
| Malignant_lesions | 0.9659 | 0.9825 | 0.9605 | 0.9632 | 0.9850 | 0.9961 | 0.9926 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9974 | 0.9947 | 0.9895 | 0.9922 | 0.9999 | 0.9998 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=9 TN=359 FN=8
- **Malignant_lesions**: TP=170 FP=7 TN=393 FN=6
- **body_of_stomach_normal**: TP=189 FP=1 TN=383 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 7 | 1 |
| Malignant_lesions | 6 | 170 | 0 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9569    0.9615    0.9592       208
     Malignant_lesions     0.9605    0.9659    0.9632       176
body_of_stomach_normal     0.9947    0.9844    0.9895       192

              accuracy                         0.9705       576
             macro avg     0.9707    0.9706    0.9706       576
          weighted avg     0.9706    0.9705    0.9705       576

```