# RandomForest 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8951 |
| Accuracy | 0.8942 |
| Macro-AUROC (OvR) | 0.9604 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8935 |
| Balanced Acc | 0.8976 |
| Macro-Sensitivity | 0.8976 |
| Macro-Specificity | 0.9468 |
| Macro-Precision | 0.8938 |
| AUC-PR (Macro) | 0.9189 |
| Cohen's Kappa | 0.8408 |
| MCC | 0.8413 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 372.86M |
| FLOPs | 745.72M |
| Params | 3.69M |
| Pure-Inference FPS | 61862.6 (0.02 ms/张) |
| End-to-End FPS | 785.3 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8555 | 0.9398 | 0.8975 | 0.8760 | 0.9133 | 0.9438 | 0.9067 | 256 |
| Malignant_lesions | 0.8667 | 0.9458 | 0.8792 | 0.8729 | 0.9397 | 0.9494 | 0.8875 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9549 | 0.9045 | 0.9365 | 0.9867 | 0.9882 | 0.9624 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=219 FP=25 TN=390 FN=37
- **Malignant_lesions**: TP=182 FP=25 TN=436 FN=28
- **body_of_stomach_normal**: TP=199 FP=21 TN=445 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 219 | 23 | 14 |
| Malignant_lesions | 21 | 182 | 7 |
| body_of_stomach_normal | 4 | 2 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8975    0.8555    0.8760       256
     Malignant_lesions     0.8792    0.8667    0.8729       210
body_of_stomach_normal     0.9045    0.9707    0.9365       205

              accuracy                         0.8942       671
             macro avg     0.8938    0.8976    0.8951       671
          weighted avg     0.8939    0.8942    0.8935       671

```