# RandomForest 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8932 |
| Accuracy | 0.8927 |
| Macro-AUROC (OvR) | 0.9568 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8921 |
| Balanced Acc | 0.8957 |
| Macro-Sensitivity | 0.8957 |
| Macro-Specificity | 0.9463 |
| Macro-Precision | 0.8918 |
| AUC-PR (Macro) | 0.9097 |
| Cohen's Kappa | 0.8386 |
| MCC | 0.8391 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 373.55M |
| FLOPs | 747.10M |
| Params | 3.70M |
| Pure-Inference FPS | 63522.8 (0.02 ms/张) |
| End-to-End FPS | 766.4 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8594 | 0.9446 | 0.9053 | 0.8818 | 0.9159 | 0.9390 | 0.9073 | 256 |
| Malignant_lesions | 0.8571 | 0.9371 | 0.8612 | 0.8592 | 0.9351 | 0.9477 | 0.8813 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9571 | 0.9087 | 0.9387 | 0.9867 | 0.9836 | 0.9405 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=220 FP=23 TN=392 FN=36
- **Malignant_lesions**: TP=180 FP=29 TN=432 FN=30
- **body_of_stomach_normal**: TP=199 FP=20 TN=446 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 220 | 26 | 10 |
| Malignant_lesions | 20 | 180 | 10 |
| body_of_stomach_normal | 3 | 3 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9053    0.8594    0.8818       256
     Malignant_lesions     0.8612    0.8571    0.8592       210
body_of_stomach_normal     0.9087    0.9707    0.9387       205

              accuracy                         0.8927       671
             macro avg     0.8918    0.8957    0.8932       671
          weighted avg     0.8926    0.8927    0.8921       671

```