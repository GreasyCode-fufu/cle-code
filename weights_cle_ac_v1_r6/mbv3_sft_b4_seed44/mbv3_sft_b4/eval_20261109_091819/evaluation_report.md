# mbv3_sft_b4 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_sft_b4_seed44/mbv3_sft_b4/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9538 |
| Accuracy | 0.9649 |
| Macro-AUROC (OvR) | 0.9966 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9643 |
| Balanced Acc | 0.9467 |
| Macro-Sensitivity | 0.9467 |
| Macro-Specificity | 0.9808 |
| Macro-Precision | 0.9622 |
| AUC-PR (Macro) | 0.9888 |
| Cohen's Kappa | 0.9438 |
| MCC | 0.9441 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 2159.3 (0.46 ms/张) |
| End-to-End FPS | 140.4 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9795 | 0.9612 | 0.9550 | 0.9671 | 0.9824 | 0.9966 | 0.9962 | 195 |
| Malignant_lesions | 0.8608 | 0.9885 | 0.9444 | 0.9007 | 0.9690 | 0.9933 | 0.9701 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9927 | 0.9871 | 0.9935 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=191 FP=9 TN=223 FN=4
- **Malignant_lesions**: TP=68 FP=4 TN=344 FN=11
- **body_of_stomach_normal**: TP=153 FP=2 TN=272 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 191 | 4 | 0 |
| Malignant_lesions | 9 | 68 | 2 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9550    0.9795    0.9671       195
     Malignant_lesions     0.9444    0.8608    0.9007        79
body_of_stomach_normal     0.9871    1.0000    0.9935       153

              accuracy                         0.9649       427
             macro avg     0.9622    0.9467    0.9538       427
          weighted avg     0.9645    0.9649    0.9643       427

```