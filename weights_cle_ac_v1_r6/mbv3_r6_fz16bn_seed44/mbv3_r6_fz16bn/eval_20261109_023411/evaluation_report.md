# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz16bn_seed44/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9173 |
| Accuracy | 0.9184 |
| Macro-AUROC (OvR) | 0.9913 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9179 |
| Balanced Acc | 0.9141 |
| Macro-Sensitivity | 0.9141 |
| Macro-Specificity | 0.9580 |
| Macro-Precision | 0.9275 |
| AUC-PR (Macro) | 0.9834 |
| Cohen's Kappa | 0.8768 |
| MCC | 0.8801 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4252.3 (0.24 ms/张) |
| End-to-End FPS | 88.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.8995 | 0.8439 | 0.8989 | 0.9764 | 0.9864 | 0.9787 | 208 |
| Malignant_lesions | 0.8068 | 0.9850 | 0.9595 | 0.8765 | 0.9206 | 0.9884 | 0.9729 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9896 | 0.9791 | 0.9765 | 0.9870 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=37 TN=331 FN=8
- **Malignant_lesions**: TP=142 FP=6 TN=394 FN=34
- **body_of_stomach_normal**: TP=187 FP=4 TN=380 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 6 | 2 |
| Malignant_lesions | 32 | 142 | 2 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8439    0.9615    0.8989       208
     Malignant_lesions     0.9595    0.8068    0.8765       176
body_of_stomach_normal     0.9791    0.9740    0.9765       192

              accuracy                         0.9184       576
             macro avg     0.9275    0.9141    0.9173       576
          weighted avg     0.9243    0.9184    0.9179       576

```