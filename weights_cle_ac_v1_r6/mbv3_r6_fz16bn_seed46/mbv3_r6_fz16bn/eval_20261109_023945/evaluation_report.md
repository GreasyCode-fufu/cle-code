# mbv3_r6_fz16bn 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz16bn_seed46/mbv3_r6_fz16bn/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9283 |
| Accuracy | 0.9288 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9286 |
| Balanced Acc | 0.9259 |
| Macro-Sensitivity | 0.9259 |
| Macro-Specificity | 0.9636 |
| Macro-Precision | 0.9337 |
| AUC-PR (Macro) | 0.9882 |
| Cohen's Kappa | 0.8927 |
| MCC | 0.8942 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4281.7 (0.23 ms/张) |
| End-to-End FPS | 60.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9266 | 0.8811 | 0.9195 | 0.9771 | 0.9909 | 0.9850 | 208 |
| Malignant_lesions | 0.8580 | 0.9850 | 0.9618 | 0.9069 | 0.9403 | 0.9918 | 0.9830 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9792 | 0.9583 | 0.9583 | 0.9792 | 0.9983 | 0.9967 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=27 TN=341 FN=8
- **Malignant_lesions**: TP=151 FP=6 TN=394 FN=25
- **body_of_stomach_normal**: TP=184 FP=8 TN=376 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 20 | 151 | 5 |
| body_of_stomach_normal | 7 | 1 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8811    0.9615    0.9195       208
     Malignant_lesions     0.9618    0.8580    0.9069       176
body_of_stomach_normal     0.9583    0.9583    0.9583       192

              accuracy                         0.9288       576
             macro avg     0.9337    0.9259    0.9283       576
          weighted avg     0.9315    0.9288    0.9286       576

```