# mbv3_r6_star2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_star2_seed46/mbv3_r6_star2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9373 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9880 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9374 |
| Balanced Acc | 0.9364 |
| Macro-Sensitivity | 0.9364 |
| Macro-Specificity | 0.9683 |
| Macro-Precision | 0.9389 |
| AUC-PR (Macro) | 0.9769 |
| Cohen's Kappa | 0.9059 |
| MCC | 0.9062 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 5840.3 (0.17 ms/张) |
| End-to-End FPS | 295.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9429 | 0.9023 | 0.9173 | 0.9612 | 0.9812 | 0.9697 | 208 |
| Malignant_lesions | 0.8920 | 0.9750 | 0.9401 | 0.9155 | 0.9535 | 0.9836 | 0.9626 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9870 | 0.9742 | 0.9793 | 0.9921 | 0.9992 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=21 TN=347 FN=14
- **Malignant_lesions**: TP=157 FP=10 TN=390 FN=19
- **body_of_stomach_normal**: TP=189 FP=5 TN=379 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 10 | 4 |
| Malignant_lesions | 18 | 157 | 1 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9023    0.9327    0.9173       208
     Malignant_lesions     0.9401    0.8920    0.9155       176
body_of_stomach_normal     0.9742    0.9844    0.9793       192

              accuracy                         0.9375       576
             macro avg     0.9389    0.9364    0.9373       576
          weighted avg     0.9378    0.9375    0.9374       576

```