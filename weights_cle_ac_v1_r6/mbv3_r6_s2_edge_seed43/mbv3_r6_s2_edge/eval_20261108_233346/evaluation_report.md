# mbv3_r6_s2_edge 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_edge_seed43/mbv3_r6_s2_edge/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9246 |
| Accuracy | 0.9253 |
| Macro-AUROC (OvR) | 0.9932 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9250 |
| Balanced Acc | 0.9226 |
| Macro-Sensitivity | 0.9226 |
| Macro-Specificity | 0.9619 |
| Macro-Precision | 0.9299 |
| AUC-PR (Macro) | 0.9869 |
| Cohen's Kappa | 0.8875 |
| MCC | 0.8889 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 2508.1 (0.40 ms/张) |
| End-to-End FPS | 177.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9239 | 0.8756 | 0.9099 | 0.9687 | 0.9894 | 0.9830 | 208 |
| Malignant_lesions | 0.8466 | 0.9825 | 0.9551 | 0.8976 | 0.9357 | 0.9915 | 0.9802 | 176 |
| body_of_stomach_normal | 0.9740 | 0.9792 | 0.9590 | 0.9664 | 0.9869 | 0.9987 | 0.9976 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=28 TN=340 FN=11
- **Malignant_lesions**: TP=149 FP=7 TN=393 FN=27
- **body_of_stomach_normal**: TP=187 FP=8 TN=376 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 7 | 4 |
| Malignant_lesions | 23 | 149 | 4 |
| body_of_stomach_normal | 5 | 0 | 187 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8756    0.9471    0.9099       208
     Malignant_lesions     0.9551    0.8466    0.8976       176
body_of_stomach_normal     0.9590    0.9740    0.9664       192

              accuracy                         0.9253       576
             macro avg     0.9299    0.9226    0.9246       576
          weighted avg     0.9277    0.9253    0.9250       576

```