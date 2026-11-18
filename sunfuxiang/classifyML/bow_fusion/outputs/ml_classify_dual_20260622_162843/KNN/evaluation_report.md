# KNN 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca-2/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8891 |
| Accuracy | 0.8882 |
| Macro-AUROC (OvR) | 0.9354 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8874 |
| Balanced Acc | 0.8913 |
| Macro-Sensitivity | 0.8913 |
| Macro-Specificity | 0.9436 |
| Macro-Precision | 0.8883 |
| AUC-PR (Macro) | 0.8665 |
| Cohen's Kappa | 0.8317 |
| MCC | 0.8323 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 372.86M |
| FLOPs | 745.72M |
| Params | 3.69M |
| Pure-Inference FPS | 3311.1 (0.30 ms/张) |
| End-to-End FPS | 641.3 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8555 | 0.9301 | 0.8831 | 0.8690 | 0.9125 | 0.9164 | 0.8525 | 256 |
| Malignant_lesions | 0.8476 | 0.9501 | 0.8856 | 0.8662 | 0.9319 | 0.9173 | 0.8338 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9506 | 0.8964 | 0.9321 | 0.9866 | 0.9725 | 0.9131 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=219 FP=29 TN=386 FN=37
- **Malignant_lesions**: TP=178 FP=23 TN=438 FN=32
- **body_of_stomach_normal**: TP=199 FP=23 TN=443 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 219 | 21 | 16 |
| Malignant_lesions | 25 | 178 | 7 |
| body_of_stomach_normal | 4 | 2 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8831    0.8555    0.8690       256
     Malignant_lesions     0.8856    0.8476    0.8662       210
body_of_stomach_normal     0.8964    0.9707    0.9321       205

              accuracy                         0.8882       671
             macro avg     0.8883    0.8913    0.8891       671
          weighted avg     0.8879    0.8882    0.8874       671

```