# convnext_wds_msf 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_wds_msf_seed42/convnext_wds_msf/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9577 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9959 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9582 |
| Balanced Acc | 0.9575 |
| Macro-Sensitivity | 0.9575 |
| Macro-Specificity | 0.9791 |
| Macro-Precision | 0.9581 |
| AUC-PR (Macro) | 0.9917 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9374 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.51G |
| FLOPs | 9.02G |
| Params | 28.25M |
| Pure-Inference FPS | 491.4 (2.04 ms/张) |
| End-to-End FPS | 190.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9728 | 0.9522 | 0.9544 | 0.9755 | 0.9946 | 0.9916 | 208 |
| Malignant_lesions | 0.9261 | 0.9775 | 0.9477 | 0.9368 | 0.9678 | 0.9936 | 0.9848 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=10 TN=358 FN=9
- **Malignant_lesions**: TP=163 FP=9 TN=391 FN=13
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 9 | 163 | 4 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9522    0.9567    0.9544       208
     Malignant_lesions     0.9477    0.9261    0.9368       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9583       576
             macro avg     0.9581    0.9575    0.9577       576
          weighted avg     0.9582    0.9583    0.9582       576

```