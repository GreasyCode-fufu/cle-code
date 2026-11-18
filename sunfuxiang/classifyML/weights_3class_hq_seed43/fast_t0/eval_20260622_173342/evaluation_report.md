# fast_t0 评估报告

- 权重: `weights_3class_hq_seed43/fast_t0/fast_t0_3class_hq_seed43_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class_highquality/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9032 |
| Accuracy | 0.9016 |
| Macro-AUROC (OvR) | 0.9789 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9012 |
| Balanced Acc | 0.9065 |
| Macro-Sensitivity | 0.9065 |
| Macro-Specificity | 0.9509 |
| Macro-Precision | 0.9017 |
| AUC-PR (Macro) | 0.9629 |
| Cohen's Kappa | 0.8522 |
| MCC | 0.8531 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 339.37M |
| FLOPs | 678.74M |
| Params | 2.63M |
| Pure-Inference FPS | 9106.9 (0.11 ms/张) |
| End-to-End FPS | 219.3 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8438 | 0.9542 | 0.9191 | 0.8798 | 0.9083 | 0.9692 | 0.9569 | 256 |
| Malignant_lesions | 0.9000 | 0.9284 | 0.8514 | 0.8750 | 0.9532 | 0.9709 | 0.9384 | 210 |
| body_of_stomach_normal | 0.9756 | 0.9700 | 0.9346 | 0.9547 | 0.9891 | 0.9968 | 0.9932 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=216 FP=19 TN=396 FN=40
- **Malignant_lesions**: TP=189 FP=33 TN=428 FN=21
- **body_of_stomach_normal**: TP=200 FP=14 TN=452 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 216 | 31 | 9 |
| Malignant_lesions | 16 | 189 | 5 |
| body_of_stomach_normal | 3 | 2 | 200 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9191    0.8438    0.8798       256
     Malignant_lesions     0.8514    0.9000    0.8750       210
body_of_stomach_normal     0.9346    0.9756    0.9547       205

              accuracy                         0.9016       671
             macro avg     0.9017    0.9065    0.9032       671
          weighted avg     0.9026    0.9016    0.9012       671

```