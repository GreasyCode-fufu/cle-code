# lsnet_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_t_seed45/lsnet_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9609 |
| Accuracy | 0.9672 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9669 |
| Balanced Acc | 0.9560 |
| Macro-Sensitivity | 0.9560 |
| Macro-Specificity | 0.9820 |
| Macro-Precision | 0.9665 |
| AUC-PR (Macro) | 0.9891 |
| Cohen's Kappa | 0.9477 |
| MCC | 0.9478 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 320.32M |
| FLOPs | 640.63M |
| Params | 11.02M |
| Pure-Inference FPS | 3623.6 (0.28 ms/张) |
| End-to-End FPS | 238.0 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9655 | 0.9594 | 0.9643 | 0.9739 | 0.9958 | 0.9956 | 195 |
| Malignant_lesions | 0.8987 | 0.9914 | 0.9595 | 0.9281 | 0.9773 | 0.9941 | 0.9724 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9891 | 0.9808 | 0.9903 | 1.0000 | 0.9996 | 0.9993 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=8 TN=224 FN=6
- **Malignant_lesions**: TP=71 FP=3 TN=345 FN=8
- **body_of_stomach_normal**: TP=153 FP=3 TN=271 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 3 | 3 |
| Malignant_lesions | 8 | 71 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9594    0.9692    0.9643       195
     Malignant_lesions     0.9595    0.8987    0.9281        79
body_of_stomach_normal     0.9808    1.0000    0.9903       153

              accuracy                         0.9672       427
             macro avg     0.9665    0.9560    0.9609       427
          weighted avg     0.9671    0.9672    0.9669       427

```