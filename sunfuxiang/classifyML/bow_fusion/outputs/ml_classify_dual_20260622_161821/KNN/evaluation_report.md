# KNN 评估报告

- 权重: `bow_fusion/weights/dual_fast_t0_clecpca/model_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9000 |
| Accuracy | 0.8987 |
| Macro-AUROC (OvR) | 0.9335 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8981 |
| Balanced Acc | 0.9027 |
| Macro-Sensitivity | 0.9027 |
| Macro-Specificity | 0.9491 |
| Macro-Precision | 0.8983 |
| AUC-PR (Macro) | 0.8607 |
| Cohen's Kappa | 0.8476 |
| MCC | 0.8481 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 373.55M |
| FLOPs | 747.10M |
| Params | 3.70M |
| Pure-Inference FPS | 3020.8 (0.33 ms/张) |
| End-to-End FPS | 617.2 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8516 | 0.9446 | 0.9046 | 0.8773 | 0.9116 | 0.9098 | 0.8460 | 256 |
| Malignant_lesions | 0.8857 | 0.9393 | 0.8692 | 0.8774 | 0.9475 | 0.9206 | 0.8259 | 210 |
| body_of_stomach_normal | 0.9707 | 0.9635 | 0.9213 | 0.9454 | 0.9868 | 0.9703 | 0.9100 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=218 FP=23 TN=392 FN=38
- **Malignant_lesions**: TP=186 FP=28 TN=433 FN=24
- **body_of_stomach_normal**: TP=199 FP=17 TN=449 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 218 | 26 | 12 |
| Malignant_lesions | 19 | 186 | 5 |
| body_of_stomach_normal | 4 | 2 | 199 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9046    0.8516    0.8773       256
     Malignant_lesions     0.8692    0.8857    0.8774       210
body_of_stomach_normal     0.9213    0.9707    0.9454       205

              accuracy                         0.8987       671
             macro avg     0.8983    0.9027    0.9000       671
          weighted avg     0.8986    0.8987    0.8981       671

```