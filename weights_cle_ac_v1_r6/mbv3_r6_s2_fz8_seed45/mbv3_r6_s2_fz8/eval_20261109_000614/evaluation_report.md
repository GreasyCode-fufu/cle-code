# mbv3_r6_s2_fz8 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_fz8_seed45/mbv3_r6_s2_fz8/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9403 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9913 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9407 |
| Balanced Acc | 0.9378 |
| Macro-Sensitivity | 0.9378 |
| Macro-Specificity | 0.9696 |
| Macro-Precision | 0.9465 |
| AUC-PR (Macro) | 0.9835 |
| Cohen's Kappa | 0.9110 |
| MCC | 0.9127 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 4512.0 (0.22 ms/张) |
| End-to-End FPS | 80.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9293 | 0.8860 | 0.9266 | 0.9828 | 0.9873 | 0.9773 | 208 |
| Malignant_lesions | 0.8580 | 0.9900 | 0.9742 | 0.9124 | 0.9406 | 0.9871 | 0.9742 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9896 | 0.9793 | 0.9818 | 0.9922 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=26 TN=342 FN=6
- **Malignant_lesions**: TP=151 FP=4 TN=396 FN=25
- **body_of_stomach_normal**: TP=189 FP=4 TN=380 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 23 | 151 | 2 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8860    0.9712    0.9266       208
     Malignant_lesions     0.9742    0.8580    0.9124       176
body_of_stomach_normal     0.9793    0.9844    0.9818       192

              accuracy                         0.9410       576
             macro avg     0.9465    0.9378    0.9403       576
          weighted avg     0.9440    0.9410    0.9407       576

```