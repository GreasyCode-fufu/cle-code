# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_dce4m_seed45/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9496 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9911 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9498 |
| Balanced Acc | 0.9494 |
| Macro-Sensitivity | 0.9494 |
| Macro-Specificity | 0.9747 |
| Macro-Precision | 0.9500 |
| AUC-PR (Macro) | 0.9829 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9243 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 5583.9 (0.18 ms/张) |
| End-to-End FPS | 243.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9620 | 0.9336 | 0.9403 | 0.9699 | 0.9871 | 0.9774 | 208 |
| Malignant_lesions | 0.9375 | 0.9700 | 0.9322 | 0.9348 | 0.9724 | 0.9877 | 0.9739 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9922 | 0.9840 | 0.9737 | 0.9820 | 0.9986 | 0.9974 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=14 TN=354 FN=11
- **Malignant_lesions**: TP=165 FP=12 TN=388 FN=11
- **body_of_stomach_normal**: TP=185 FP=3 TN=381 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 9 | 165 | 2 |
| body_of_stomach_normal | 5 | 2 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9336    0.9471    0.9403       208
     Malignant_lesions     0.9322    0.9375    0.9348       176
body_of_stomach_normal     0.9840    0.9635    0.9737       192

              accuracy                         0.9497       576
             macro avg     0.9500    0.9494    0.9496       576
          weighted avg     0.9500    0.9497    0.9498       576

```