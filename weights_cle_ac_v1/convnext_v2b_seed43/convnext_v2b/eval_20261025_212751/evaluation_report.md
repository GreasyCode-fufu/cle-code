# convnext_v2b 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_v2b_seed43/convnext_v2b/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9558 |
| Accuracy | 0.9566 |
| Macro-AUROC (OvR) | 0.9941 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9564 |
| Balanced Acc | 0.9552 |
| Macro-Sensitivity | 0.9552 |
| Macro-Specificity | 0.9781 |
| Macro-Precision | 0.9569 |
| AUC-PR (Macro) | 0.9875 |
| Cohen's Kappa | 0.9347 |
| MCC | 0.9349 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.93G |
| Params | 28.22M |
| Pure-Inference FPS | 869.0 (1.15 ms/张) |
| End-to-End FPS | 227.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9674 | 0.9437 | 0.9549 | 0.9807 | 0.9918 | 0.9899 | 208 |
| Malignant_lesions | 0.9148 | 0.9800 | 0.9527 | 0.9333 | 0.9631 | 0.9913 | 0.9744 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9870 | 0.9742 | 0.9793 | 0.9921 | 0.9991 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=12 TN=356 FN=7
- **Malignant_lesions**: TP=161 FP=8 TN=392 FN=15
- **body_of_stomach_normal**: TP=189 FP=5 TN=379 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 11 | 161 | 4 |
| body_of_stomach_normal | 1 | 2 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9437    0.9663    0.9549       208
     Malignant_lesions     0.9527    0.9148    0.9333       176
body_of_stomach_normal     0.9742    0.9844    0.9793       192

              accuracy                         0.9566       576
             macro avg     0.9569    0.9552    0.9558       576
          weighted avg     0.9566    0.9566    0.9564       576

```