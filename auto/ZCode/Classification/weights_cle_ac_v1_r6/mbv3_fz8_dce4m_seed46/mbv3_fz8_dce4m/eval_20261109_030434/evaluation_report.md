# mbv3_fz8_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_fz8_dce4m_seed46/mbv3_fz8_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9619 |
| Accuracy | 0.9618 |
| Macro-AUROC (OvR) | 0.9939 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9618 |
| Balanced Acc | 0.9607 |
| Macro-Sensitivity | 0.9607 |
| Macro-Specificity | 0.9805 |
| Macro-Precision | 0.9637 |
| AUC-PR (Macro) | 0.9885 |
| Cohen's Kappa | 0.9425 |
| MCC | 0.9428 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5211.6 (0.19 ms/张) |
| End-to-End FPS | 13.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9592 | 0.9306 | 0.9481 | 0.9806 | 0.9912 | 0.9828 | 208 |
| Malignant_lesions | 0.9261 | 0.9900 | 0.9760 | 0.9504 | 0.9682 | 0.9911 | 0.9836 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9922 | 0.9845 | 0.9870 | 0.9948 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=15 TN=353 FN=7
- **Malignant_lesions**: TP=163 FP=4 TN=396 FN=13
- **body_of_stomach_normal**: TP=190 FP=3 TN=381 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 4 | 3 |
| Malignant_lesions | 13 | 163 | 0 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9306    0.9663    0.9481       208
     Malignant_lesions     0.9760    0.9261    0.9504       176
body_of_stomach_normal     0.9845    0.9896    0.9870       192

              accuracy                         0.9618       576
             macro avg     0.9637    0.9607    0.9619       576
          weighted avg     0.9624    0.9618    0.9618       576

```