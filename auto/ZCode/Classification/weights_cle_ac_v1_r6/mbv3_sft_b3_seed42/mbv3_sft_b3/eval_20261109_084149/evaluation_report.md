# mbv3_sft_b3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b3_seed42/mbv3_sft_b3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9706 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9976 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9767 |
| Balanced Acc | 0.9729 |
| Macro-Sensitivity | 0.9729 |
| Macro-Specificity | 0.9885 |
| Macro-Precision | 0.9684 |
| AUC-PR (Macro) | 0.9919 |
| Cohen's Kappa | 0.9628 |
| MCC | 0.9629 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1863.9 (0.54 ms/张) |
| End-to-End FPS | 27.2 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9828 | 0.9793 | 0.9742 | 0.9744 | 0.9975 | 0.9972 | 195 |
| Malignant_lesions | 0.9494 | 0.9828 | 0.9259 | 0.9375 | 0.9884 | 0.9953 | 0.9784 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=4 TN=228 FN=6
- **Malignant_lesions**: TP=75 FP=6 TN=342 FN=4
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 6 | 0 |
| Malignant_lesions | 4 | 75 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9793    0.9692    0.9742       195
     Malignant_lesions     0.9259    0.9494    0.9375        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9766       427
             macro avg     0.9684    0.9729    0.9706       427
          weighted avg     0.9768    0.9766    0.9767       427

```