# VotingEnsemble 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8936 |
| Accuracy | 0.8927 |
| Macro-AUROC (OvR) | 0.9787 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8920 |
| Balanced Acc | 0.8963 |
| Macro-Sensitivity | 0.8963 |
| Macro-Specificity | 0.9461 |
| Macro-Precision | 0.8923 |
| AUC-PR (Macro) | 0.9641 |
| Cohen's Kappa | 0.8386 |
| MCC | 0.8392 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 372.86M |
| FLOPs | 745.72M |
| Params | 3.69M |
| Pure-Inference FPS | 42845.9 (0.02 ms/张) |
| End-to-End FPS | 780.9 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8516 | 0.9398 | 0.8971 | 0.8737 | 0.9112 | 0.9713 | 0.9570 | 256 |
| Malignant_lesions | 0.8667 | 0.9479 | 0.8835 | 0.8750 | 0.9398 | 0.9699 | 0.9461 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9506 | 0.8964 | 0.9321 | 0.9866 | 0.9950 | 0.9891 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=218 FP=25 TN=390 FN=38
- **Malignant_lesions**: TP=182 FP=24 TN=437 FN=28
- **body_of_stomach_normal**: TP=199 FP=23 TN=443 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 218 | 22 | 16 |
| Malignant_lesions | 21 | 182 | 7 |
| body_of_stomach_normal | 4 | 2 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8971    0.8516    0.8737       256
     Malignant_lesions     0.8835    0.8667    0.8750       210
body_of_stomach_normal     0.8964    0.9707    0.9321       205

              accuracy                         0.8927       671
             macro avg     0.8923    0.8963    0.8936       671
          weighted avg     0.8926    0.8927    0.8920       671

```