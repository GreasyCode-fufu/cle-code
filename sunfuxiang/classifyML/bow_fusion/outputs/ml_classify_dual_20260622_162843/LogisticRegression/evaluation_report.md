# LogisticRegression 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8935 |
| Accuracy | 0.8927 |
| Macro-AUROC (OvR) | 0.9754 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8919 |
| Balanced Acc | 0.8964 |
| Macro-Sensitivity | 0.8964 |
| Macro-Specificity | 0.9461 |
| Macro-Precision | 0.8923 |
| AUC-PR (Macro) | 0.9574 |
| Cohen's Kappa | 0.8386 |
| MCC | 0.8393 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 372.86M |
| FLOPs | 745.72M |
| Params | 3.69M |
| Pure-Inference FPS | 1495418.7 (0.00 ms/张) |
| End-to-End FPS | 795.0 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8516 | 0.9398 | 0.8971 | 0.8737 | 0.9112 | 0.9653 | 0.9540 | 256 |
| Malignant_lesions | 0.8619 | 0.9479 | 0.8829 | 0.8723 | 0.9378 | 0.9686 | 0.9432 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9506 | 0.8969 | 0.9346 | 0.9888 | 0.9922 | 0.9751 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=218 FP=25 TN=390 FN=38
- **Malignant_lesions**: TP=181 FP=24 TN=437 FN=29
- **body_of_stomach_normal**: TP=200 FP=23 TN=443 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 218 | 22 | 16 |
| Malignant_lesions | 22 | 181 | 7 |
| body_of_stomach_normal | 3 | 2 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8971    0.8516    0.8737       256
     Malignant_lesions     0.8829    0.8619    0.8723       210
body_of_stomach_normal     0.8969    0.9756    0.9346       205

              accuracy                         0.8927       671
             macro avg     0.8923    0.8964    0.8935       671
          weighted avg     0.8926    0.8927    0.8919       671

```