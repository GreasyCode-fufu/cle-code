# mbv3_taunorm 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_taunorm_seed45/mbv3_taunorm/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9342 |
| Accuracy | 0.9340 |
| Macro-AUROC (OvR) | 0.9872 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9342 |
| Balanced Acc | 0.9318 |
| Macro-Sensitivity | 0.9318 |
| Macro-Specificity | 0.9662 |
| Macro-Precision | 0.9389 |
| AUC-PR (Macro) | 0.9696 |
| Cohen's Kappa | 0.9006 |
| MCC | 0.9018 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 3161.3 (0.32 ms/张) |
| End-to-End FPS | 7.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9293 | 0.8860 | 0.9266 | 0.9828 | 0.9824 | 0.9555 | 208 |
| Malignant_lesions | 0.8920 | 0.9850 | 0.9632 | 0.9263 | 0.9540 | 0.9866 | 0.9685 | 176 |
| body_of_stomach_normal | 0.9323 | 0.9844 | 0.9676 | 0.9496 | 0.9668 | 0.9926 | 0.9847 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=26 TN=342 FN=6
- **Malignant_lesions**: TP=157 FP=6 TN=394 FN=19
- **body_of_stomach_normal**: TP=179 FP=6 TN=378 FN=13

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 15 | 157 | 4 |
| body_of_stomach_normal | 11 | 2 | 179 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8860    0.9712    0.9266       208
     Malignant_lesions     0.9632    0.8920    0.9263       176
body_of_stomach_normal     0.9676    0.9323    0.9496       192

              accuracy                         0.9340       576
             macro avg     0.9389    0.9318    0.9342       576
          weighted avg     0.9368    0.9340    0.9342       576

```