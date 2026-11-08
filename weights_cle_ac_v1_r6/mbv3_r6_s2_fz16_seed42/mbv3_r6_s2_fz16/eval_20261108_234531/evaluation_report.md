# mbv3_r6_s2_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_fz16_seed42/mbv3_r6_s2_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9584 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9946 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9583 |
| Balanced Acc | 0.9578 |
| Macro-Sensitivity | 0.9578 |
| Macro-Specificity | 0.9788 |
| Macro-Precision | 0.9593 |
| AUC-PR (Macro) | 0.9896 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9374 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 4562.4 (0.22 ms/张) |
| End-to-End FPS | 263.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9620 | 0.9340 | 0.9429 | 0.9725 | 0.9909 | 0.9844 | 208 |
| Malignant_lesions | 0.9318 | 0.9850 | 0.9647 | 0.9480 | 0.9704 | 0.9935 | 0.9859 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9896 | 0.9794 | 0.9845 | 0.9948 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=14 TN=354 FN=10
- **Malignant_lesions**: TP=164 FP=6 TN=394 FN=12
- **body_of_stomach_normal**: TP=190 FP=4 TN=380 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 6 | 4 |
| Malignant_lesions | 12 | 164 | 0 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9340    0.9519    0.9429       208
     Malignant_lesions     0.9647    0.9318    0.9480       176
body_of_stomach_normal     0.9794    0.9896    0.9845       192

              accuracy                         0.9583       576
             macro avg     0.9593    0.9578    0.9584       576
          weighted avg     0.9585    0.9583    0.9583       576

```