# mbv3_r6_s2_fz16 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_fz16_seed46/mbv3_r6_s2_fz16/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9406 |
| Accuracy | 0.9410 |
| Macro-AUROC (OvR) | 0.9939 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9408 |
| Balanced Acc | 0.9384 |
| Macro-Sensitivity | 0.9384 |
| Macro-Specificity | 0.9697 |
| Macro-Precision | 0.9455 |
| AUC-PR (Macro) | 0.9885 |
| Cohen's Kappa | 0.9110 |
| MCC | 0.9123 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 4796.3 (0.21 ms/张) |
| End-to-End FPS | 235.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9321 | 0.8889 | 0.9238 | 0.9772 | 0.9893 | 0.9828 | 208 |
| Malignant_lesions | 0.8693 | 0.9875 | 0.9684 | 0.9162 | 0.9450 | 0.9932 | 0.9843 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9896 | 0.9793 | 0.9818 | 0.9922 | 0.9993 | 0.9986 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=25 TN=343 FN=8
- **Malignant_lesions**: TP=153 FP=5 TN=395 FN=23
- **body_of_stomach_normal**: TP=189 FP=4 TN=380 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 22 | 153 | 1 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8889    0.9615    0.9238       208
     Malignant_lesions     0.9684    0.8693    0.9162       176
body_of_stomach_normal     0.9793    0.9844    0.9818       192

              accuracy                         0.9410       576
             macro avg     0.9455    0.9384    0.9406       576
          weighted avg     0.9433    0.9410    0.9408       576

```