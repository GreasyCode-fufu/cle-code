# convnext_tiny_fz3_dce_last 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_dce_last_seed45/convnext_tiny_fz3_dce_last/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9679 |
| Accuracy | 0.9742 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9740 |
| Balanced Acc | 0.9636 |
| Macro-Sensitivity | 0.9636 |
| Macro-Specificity | 0.9859 |
| Macro-Precision | 0.9727 |
| AUC-PR (Macro) | 0.9902 |
| Cohen's Kappa | 0.9589 |
| MCC | 0.9590 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1459.5 (0.69 ms/张) |
| End-to-End FPS | 21.1 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9698 | 0.9646 | 0.9720 | 0.9825 | 0.9966 | 0.9961 | 195 |
| Malignant_lesions | 0.9114 | 0.9914 | 0.9600 | 0.9351 | 0.9801 | 0.9940 | 0.9744 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=7 TN=225 FN=4
- **Malignant_lesions**: TP=72 FP=3 TN=345 FN=7
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 3 | 1 |
| Malignant_lesions | 7 | 72 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9646    0.9795    0.9720       195
     Malignant_lesions     0.9600    0.9114    0.9351        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9742       427
             macro avg     0.9727    0.9636    0.9679       427
          weighted avg     0.9741    0.9742    0.9740       427

```