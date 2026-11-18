# shvit_s1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/shvit_s1_seed43/shvit_s1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9724 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9974 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9765 |
| Balanced Acc | 0.9729 |
| Macro-Sensitivity | 0.9729 |
| Macro-Specificity | 0.9880 |
| Macro-Precision | 0.9719 |
| AUC-PR (Macro) | 0.9941 |
| Cohen's Kappa | 0.9628 |
| MCC | 0.9628 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 245.71M |
| FLOPs | 491.41M |
| Params | 6.01M |
| Pure-Inference FPS | 3681.9 (0.27 ms/张) |
| End-to-End FPS | 219.6 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9828 | 0.9793 | 0.9742 | 0.9744 | 0.9967 | 0.9958 | 195 |
| Malignant_lesions | 0.9494 | 0.9885 | 0.9494 | 0.9494 | 0.9885 | 0.9956 | 0.9867 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 0.9999 | 0.9998 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=4 TN=228 FN=6
- **Malignant_lesions**: TP=75 FP=4 TN=344 FN=4
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 4 | 2 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9793    0.9692    0.9742       195
     Malignant_lesions     0.9494    0.9494    0.9494        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9766       427
             macro avg     0.9719    0.9729    0.9724       427
          weighted avg     0.9765    0.9766    0.9765       427

```