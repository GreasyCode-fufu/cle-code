# SVM-Linear 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8893 |
| Accuracy | 0.8882 |
| Macro-AUROC (OvR) | 0.9792 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8874 |
| Balanced Acc | 0.8921 |
| Macro-Sensitivity | 0.8921 |
| Macro-Specificity | 0.9438 |
| Macro-Precision | 0.8881 |
| AUC-PR (Macro) | 0.9648 |
| Cohen's Kappa | 0.8318 |
| MCC | 0.8326 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 372.86M |
| FLOPs | 745.72M |
| Params | 3.69M |
| Pure-Inference FPS | 202269.5 (0.00 ms/张) |
| End-to-End FPS | 792.3 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8438 | 0.9349 | 0.8889 | 0.8657 | 0.9065 | 0.9720 | 0.9574 | 256 |
| Malignant_lesions | 0.8619 | 0.9479 | 0.8829 | 0.8723 | 0.9378 | 0.9706 | 0.9469 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9485 | 0.8924 | 0.9299 | 0.9866 | 0.9951 | 0.9901 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=216 FP=27 TN=388 FN=40
- **Malignant_lesions**: TP=181 FP=24 TN=437 FN=29
- **body_of_stomach_normal**: TP=199 FP=24 TN=442 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 216 | 22 | 18 |
| Malignant_lesions | 23 | 181 | 6 |
| body_of_stomach_normal | 4 | 2 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8889    0.8438    0.8657       256
     Malignant_lesions     0.8829    0.8619    0.8723       210
body_of_stomach_normal     0.8924    0.9707    0.9299       205

              accuracy                         0.8882       671
             macro avg     0.8881    0.8921    0.8893       671
          weighted avg     0.8881    0.8882    0.8874       671

```