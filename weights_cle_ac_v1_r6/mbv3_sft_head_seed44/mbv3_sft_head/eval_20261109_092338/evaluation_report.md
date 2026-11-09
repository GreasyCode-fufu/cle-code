# mbv3_sft_head 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_head_seed44/mbv3_sft_head/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8736 |
| Accuracy | 0.8733 |
| Macro-AUROC (OvR) | 0.9764 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8737 |
| Balanced Acc | 0.8690 |
| Macro-Sensitivity | 0.8690 |
| Macro-Specificity | 0.9350 |
| Macro-Precision | 0.8875 |
| AUC-PR (Macro) | 0.9574 |
| Cohen's Kappa | 0.8085 |
| MCC | 0.8131 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3253.1 (0.31 ms/张) |
| End-to-End FPS | 110.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.8560 | 0.7854 | 0.8527 | 0.9574 | 0.9634 | 0.9420 | 208 |
| Malignant_lesions | 0.7784 | 0.9750 | 0.9320 | 0.8483 | 0.9091 | 0.9761 | 0.9493 | 176 |
| body_of_stomach_normal | 0.8958 | 0.9740 | 0.9451 | 0.9198 | 0.9492 | 0.9898 | 0.9808 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=53 TN=315 FN=14
- **Malignant_lesions**: TP=137 FP=10 TN=390 FN=39
- **body_of_stomach_normal**: TP=172 FP=10 TN=374 FN=20

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 7 | 7 |
| Malignant_lesions | 36 | 137 | 3 |
| body_of_stomach_normal | 17 | 3 | 172 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.7854    0.9327    0.8527       208
     Malignant_lesions     0.9320    0.7784    0.8483       176
body_of_stomach_normal     0.9451    0.8958    0.9198       192

              accuracy                         0.8733       576
             macro avg     0.8875    0.8690    0.8736       576
          weighted avg     0.8834    0.8733    0.8737       576

```