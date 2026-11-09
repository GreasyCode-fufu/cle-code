# mbv3_sft_b4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b4_seed43/mbv3_sft_b4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9319 |
| Accuracy | 0.9323 |
| Macro-AUROC (OvR) | 0.9909 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9323 |
| Balanced Acc | 0.9293 |
| Macro-Sensitivity | 0.9293 |
| Macro-Specificity | 0.9653 |
| Macro-Precision | 0.9379 |
| AUC-PR (Macro) | 0.9828 |
| Cohen's Kappa | 0.8979 |
| MCC | 0.8996 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2971.9 (0.34 ms/张) |
| End-to-End FPS | 141.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9212 | 0.8739 | 0.9178 | 0.9798 | 0.9864 | 0.9772 | 208 |
| Malignant_lesions | 0.8580 | 0.9825 | 0.9557 | 0.9042 | 0.9402 | 0.9882 | 0.9743 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9922 | 0.9840 | 0.9737 | 0.9820 | 0.9982 | 0.9969 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=29 TN=339 FN=7
- **Malignant_lesions**: TP=151 FP=7 TN=393 FN=25
- **body_of_stomach_normal**: TP=185 FP=3 TN=381 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 6 | 1 |
| Malignant_lesions | 23 | 151 | 2 |
| body_of_stomach_normal | 6 | 1 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8739    0.9663    0.9178       208
     Malignant_lesions     0.9557    0.8580    0.9042       176
body_of_stomach_normal     0.9840    0.9635    0.9737       192

              accuracy                         0.9323       576
             macro avg     0.9379    0.9293    0.9319       576
          weighted avg     0.9356    0.9323    0.9323       576

```