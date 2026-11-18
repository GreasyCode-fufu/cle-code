# XGBoost 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8906 |
| Accuracy | 0.8897 |
| Macro-AUROC (OvR) | 0.9724 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8890 |
| Balanced Acc | 0.8931 |
| Macro-Sensitivity | 0.8931 |
| Macro-Specificity | 0.9446 |
| Macro-Precision | 0.8892 |
| AUC-PR (Macro) | 0.9534 |
| Cohen's Kappa | 0.8341 |
| MCC | 0.8346 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 372.86M |
| FLOPs | 745.72M |
| Params | 3.69M |
| Pure-Inference FPS | 532018.5 (0.00 ms/张) |
| End-to-End FPS | 794.2 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8516 | 0.9373 | 0.8934 | 0.8720 | 0.9110 | 0.9604 | 0.9494 | 256 |
| Malignant_lesions | 0.8571 | 0.9436 | 0.8738 | 0.8654 | 0.9355 | 0.9642 | 0.9327 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9528 | 0.9005 | 0.9343 | 0.9867 | 0.9926 | 0.9783 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=218 FP=26 TN=389 FN=38
- **Malignant_lesions**: TP=180 FP=26 TN=435 FN=30
- **body_of_stomach_normal**: TP=199 FP=22 TN=444 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 218 | 23 | 15 |
| Malignant_lesions | 23 | 180 | 7 |
| body_of_stomach_normal | 3 | 3 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8934    0.8516    0.8720       256
     Malignant_lesions     0.8738    0.8571    0.8654       210
body_of_stomach_normal     0.9005    0.9707    0.9343       205

              accuracy                         0.8897       671
             macro avg     0.8892    0.8931    0.8906       671
          weighted avg     0.8894    0.8897    0.8890       671

```