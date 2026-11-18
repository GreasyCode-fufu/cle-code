# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce4m_seed44/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9586 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9934 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9584 |
| Balanced Acc | 0.9584 |
| Macro-Sensitivity | 0.9584 |
| Macro-Specificity | 0.9790 |
| Macro-Precision | 0.9588 |
| AUC-PR (Macro) | 0.9879 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9374 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5652.3 (0.18 ms/张) |
| End-to-End FPS | 258.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9647 | 0.9381 | 0.9426 | 0.9699 | 0.9887 | 0.9811 | 208 |
| Malignant_lesions | 0.9489 | 0.9775 | 0.9489 | 0.9489 | 0.9775 | 0.9926 | 0.9846 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9948 | 0.9895 | 0.9843 | 0.9896 | 0.9990 | 0.9980 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=13 TN=355 FN=11
- **Malignant_lesions**: TP=167 FP=9 TN=391 FN=9
- **body_of_stomach_normal**: TP=188 FP=2 TN=382 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 9 | 167 | 0 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9381    0.9471    0.9426       208
     Malignant_lesions     0.9489    0.9489    0.9489       176
body_of_stomach_normal     0.9895    0.9792    0.9843       192

              accuracy                         0.9583       576
             macro avg     0.9588    0.9584    0.9586       576
          weighted avg     0.9585    0.9583    0.9584       576

```