# mbv3_sft_b1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b1_seed45/mbv3_sft_b1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8874 |
| Accuracy | 0.8889 |
| Macro-AUROC (OvR) | 0.9851 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8880 |
| Balanced Acc | 0.8844 |
| Macro-Sensitivity | 0.8844 |
| Macro-Specificity | 0.9431 |
| Macro-Precision | 0.8991 |
| AUC-PR (Macro) | 0.9723 |
| Cohen's Kappa | 0.8322 |
| MCC | 0.8359 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3511.6 (0.28 ms/张) |
| End-to-End FPS | 147.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.8804 | 0.8143 | 0.8674 | 0.9558 | 0.9756 | 0.9604 | 208 |
| Malignant_lesions | 0.7670 | 0.9800 | 0.9441 | 0.8464 | 0.9053 | 0.9839 | 0.9644 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9688 | 0.9388 | 0.9485 | 0.9789 | 0.9959 | 0.9922 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=44 TN=324 FN=15
- **Malignant_lesions**: TP=135 FP=8 TN=392 FN=41
- **body_of_stomach_normal**: TP=184 FP=12 TN=372 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 7 | 8 |
| Malignant_lesions | 37 | 135 | 4 |
| body_of_stomach_normal | 7 | 1 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8143    0.9279    0.8674       208
     Malignant_lesions     0.9441    0.7670    0.8464       176
body_of_stomach_normal     0.9388    0.9583    0.9485       192

              accuracy                         0.8889       576
             macro avg     0.8991    0.8844    0.8874       576
          weighted avg     0.8955    0.8889    0.8880       576

```