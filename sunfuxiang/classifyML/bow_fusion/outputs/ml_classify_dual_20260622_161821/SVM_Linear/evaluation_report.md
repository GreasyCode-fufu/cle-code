# SVM-Linear 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8981 |
| Accuracy | 0.8972 |
| Macro-AUROC (OvR) | 0.9745 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8965 |
| Balanced Acc | 0.9011 |
| Macro-Sensitivity | 0.9011 |
| Macro-Specificity | 0.9485 |
| Macro-Precision | 0.8965 |
| AUC-PR (Macro) | 0.9517 |
| Cohen's Kappa | 0.8454 |
| MCC | 0.8460 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 373.55M |
| FLOPs | 747.10M |
| Params | 3.70M |
| Pure-Inference FPS | 569251.2 (0.00 ms/张) |
| End-to-End FPS | 774.7 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8516 | 0.9470 | 0.9083 | 0.8790 | 0.9118 | 0.9617 | 0.9284 | 256 |
| Malignant_lesions | 0.8762 | 0.9414 | 0.8720 | 0.8741 | 0.9435 | 0.9698 | 0.9377 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9571 | 0.9091 | 0.9412 | 0.9889 | 0.9918 | 0.9889 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=218 FP=22 TN=393 FN=38
- **Malignant_lesions**: TP=184 FP=27 TN=434 FN=26
- **body_of_stomach_normal**: TP=200 FP=20 TN=446 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 218 | 25 | 13 |
| Malignant_lesions | 19 | 184 | 7 |
| body_of_stomach_normal | 3 | 2 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9083    0.8516    0.8790       256
     Malignant_lesions     0.8720    0.8762    0.8741       210
body_of_stomach_normal     0.9091    0.9756    0.9412       205

              accuracy                         0.8972       671
             macro avg     0.8965    0.9011    0.8981       671
          weighted avg     0.8972    0.8972    0.8965       671

```