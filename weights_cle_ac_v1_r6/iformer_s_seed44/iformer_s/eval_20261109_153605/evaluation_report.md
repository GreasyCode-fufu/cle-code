# iformer_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/iformer_s_seed44/iformer_s/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9408 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9916 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9408 |
| Balanced Acc | 0.9400 |
| Macro-Sensitivity | 0.9400 |
| Macro-Specificity | 0.9701 |
| Macro-Precision | 0.9420 |
| AUC-PR (Macro) | 0.9843 |
| Cohen's Kappa | 0.9112 |
| MCC | 0.9114 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.08G |
| FLOPs | 2.16G |
| Params | 6.24M |
| Pure-Inference FPS | 3052.7 (0.33 ms/张) |
| End-to-End FPS | 229.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9511 | 0.9155 | 0.9264 | 0.9642 | 0.9874 | 0.9794 | 208 |
| Malignant_lesions | 0.9034 | 0.9775 | 0.9464 | 0.9244 | 0.9583 | 0.9895 | 0.9771 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9818 | 0.9641 | 0.9716 | 0.9895 | 0.9979 | 0.9963 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=18 TN=350 FN=13
- **Malignant_lesions**: TP=159 FP=9 TN=391 FN=17
- **body_of_stomach_normal**: TP=188 FP=7 TN=377 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 9 | 4 |
| Malignant_lesions | 14 | 159 | 3 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9155    0.9375    0.9264       208
     Malignant_lesions     0.9464    0.9034    0.9244       176
body_of_stomach_normal     0.9641    0.9792    0.9716       192

              accuracy                         0.9410       576
             macro avg     0.9420    0.9400    0.9408       576
          weighted avg     0.9411    0.9410    0.9408       576

```