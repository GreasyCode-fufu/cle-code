# mbv3_r6_s2_fz8 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_fz8_seed46/mbv3_r6_s2_fz8/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9675 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9984 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9743 |
| Balanced Acc | 0.9686 |
| Macro-Sensitivity | 0.9686 |
| Macro-Specificity | 0.9871 |
| Macro-Precision | 0.9664 |
| AUC-PR (Macro) | 0.9947 |
| Cohen's Kappa | 0.9591 |
| MCC | 0.9591 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.96M |
| FLOPs | 463.92M |
| Params | 3.25M |
| Pure-Inference FPS | 3535.7 (0.28 ms/张) |
| End-to-End FPS | 245.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9784 | 0.9742 | 0.9717 | 0.9742 | 0.9981 | 0.9979 | 195 |
| Malignant_lesions | 0.9367 | 0.9828 | 0.9250 | 0.9308 | 0.9856 | 0.9969 | 0.9861 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=5 TN=227 FN=6
- **Malignant_lesions**: TP=74 FP=6 TN=342 FN=5
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 6 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9742    0.9692    0.9717       195
     Malignant_lesions     0.9250    0.9367    0.9308        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9742       427
             macro avg     0.9664    0.9686    0.9675       427
          weighted avg     0.9744    0.9742    0.9743       427

```