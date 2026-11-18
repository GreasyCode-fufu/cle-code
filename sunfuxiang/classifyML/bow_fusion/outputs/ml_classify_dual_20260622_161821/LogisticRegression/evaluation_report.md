# LogisticRegression 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8982 |
| Accuracy | 0.8972 |
| Macro-AUROC (OvR) | 0.9733 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8966 |
| Balanced Acc | 0.9008 |
| Macro-Sensitivity | 0.9008 |
| Macro-Specificity | 0.9484 |
| Macro-Precision | 0.8966 |
| AUC-PR (Macro) | 0.9566 |
| Cohen's Kappa | 0.8453 |
| MCC | 0.8458 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 373.55M |
| FLOPs | 747.10M |
| Params | 3.70M |
| Pure-Inference FPS | 1412131.5 (0.00 ms/张) |
| End-to-End FPS | 775.3 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8555 | 0.9446 | 0.9050 | 0.8795 | 0.9138 | 0.9629 | 0.9552 | 256 |
| Malignant_lesions | 0.8762 | 0.9393 | 0.8679 | 0.8720 | 0.9434 | 0.9657 | 0.9294 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9614 | 0.9171 | 0.9431 | 0.9868 | 0.9912 | 0.9851 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=219 FP=23 TN=392 FN=37
- **Malignant_lesions**: TP=184 FP=28 TN=433 FN=26
- **body_of_stomach_normal**: TP=199 FP=18 TN=448 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 219 | 26 | 11 |
| Malignant_lesions | 19 | 184 | 7 |
| body_of_stomach_normal | 4 | 2 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9050    0.8555    0.8795       256
     Malignant_lesions     0.8679    0.8762    0.8720       210
body_of_stomach_normal     0.9171    0.9707    0.9431       205

              accuracy                         0.8972       671
             macro avg     0.8966    0.9008    0.8982       671
          weighted avg     0.8971    0.8972    0.8966       671

```