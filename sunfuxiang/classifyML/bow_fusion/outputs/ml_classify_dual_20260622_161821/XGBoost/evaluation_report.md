# XGBoost 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8950 |
| Accuracy | 0.8942 |
| Macro-AUROC (OvR) | 0.9636 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8936 |
| Balanced Acc | 0.8979 |
| Macro-Sensitivity | 0.8979 |
| Macro-Specificity | 0.9471 |
| Macro-Precision | 0.8934 |
| AUC-PR (Macro) | 0.9226 |
| Cohen's Kappa | 0.8409 |
| MCC | 0.8416 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 373.55M |
| FLOPs | 747.10M |
| Params | 3.70M |
| Pure-Inference FPS | 563100.8 (0.00 ms/张) |
| End-to-End FPS | 774.6 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8516 | 0.9470 | 0.9083 | 0.8790 | 0.9118 | 0.9477 | 0.9183 | 256 |
| Malignant_lesions | 0.8714 | 0.9393 | 0.8673 | 0.8694 | 0.9413 | 0.9539 | 0.8907 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9549 | 0.9045 | 0.9365 | 0.9867 | 0.9891 | 0.9588 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=218 FP=22 TN=393 FN=38
- **Malignant_lesions**: TP=183 FP=28 TN=433 FN=27
- **body_of_stomach_normal**: TP=199 FP=21 TN=445 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 218 | 26 | 12 |
| Malignant_lesions | 18 | 183 | 9 |
| body_of_stomach_normal | 4 | 2 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9083    0.8516    0.8790       256
     Malignant_lesions     0.8673    0.8714    0.8694       210
body_of_stomach_normal     0.9045    0.9707    0.9365       205

              accuracy                         0.8942       671
             macro avg     0.8934    0.8979    0.8950       671
          weighted avg     0.8943    0.8942    0.8936       671

```