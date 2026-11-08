# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_fz8m_seed42/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9512 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9958 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9513 |
| Balanced Acc | 0.9512 |
| Macro-Sensitivity | 0.9512 |
| Macro-Specificity | 0.9756 |
| Macro-Precision | 0.9511 |
| AUC-PR (Macro) | 0.9919 |
| Cohen's Kappa | 0.9269 |
| MCC | 0.9269 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 4542.1 (0.22 ms/张) |
| End-to-End FPS | 87.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9674 | 0.9420 | 0.9398 | 0.9648 | 0.9936 | 0.9897 | 208 |
| Malignant_lesions | 0.9318 | 0.9725 | 0.9371 | 0.9345 | 0.9701 | 0.9946 | 0.9878 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9870 | 0.9742 | 0.9793 | 0.9921 | 0.9991 | 0.9983 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=12 TN=356 FN=13
- **Malignant_lesions**: TP=164 FP=11 TN=389 FN=12
- **body_of_stomach_normal**: TP=189 FP=5 TN=379 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 10 | 3 |
| Malignant_lesions | 10 | 164 | 2 |
| body_of_stomach_normal | 2 | 1 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9420    0.9375    0.9398       208
     Malignant_lesions     0.9371    0.9318    0.9345       176
body_of_stomach_normal     0.9742    0.9844    0.9793       192

              accuracy                         0.9514       576
             macro avg     0.9511    0.9512    0.9512       576
          weighted avg     0.9513    0.9514    0.9513       576

```