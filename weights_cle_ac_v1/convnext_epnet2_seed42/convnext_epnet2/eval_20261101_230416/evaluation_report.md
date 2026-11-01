# convnext_epnet2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_epnet2_seed42/convnext_epnet2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9371 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9921 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9373 |
| Balanced Acc | 0.9390 |
| Macro-Sensitivity | 0.9390 |
| Macro-Specificity | 0.9691 |
| Macro-Precision | 0.9367 |
| AUC-PR (Macro) | 0.9846 |
| Cohen's Kappa | 0.9062 |
| MCC | 0.9069 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 28.57M |
| Pure-Inference FPS | 956.3 (1.05 ms/张) |
| End-to-End FPS | 226.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8846 | 0.9755 | 0.9534 | 0.9177 | 0.9373 | 0.9881 | 0.9819 | 208 |
| Malignant_lesions | 0.9375 | 0.9475 | 0.8871 | 0.9116 | 0.9718 | 0.9886 | 0.9728 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9844 | 0.9695 | 0.9820 | 0.9974 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=184 FP=9 TN=359 FN=24
- **Malignant_lesions**: TP=165 FP=21 TN=379 FN=11
- **body_of_stomach_normal**: TP=191 FP=6 TN=378 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 184 | 20 | 4 |
| Malignant_lesions | 9 | 165 | 2 |
| body_of_stomach_normal | 0 | 1 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9534    0.8846    0.9177       208
     Malignant_lesions     0.8871    0.9375    0.9116       176
body_of_stomach_normal     0.9695    0.9948    0.9820       192

              accuracy                         0.9375       576
             macro avg     0.9367    0.9390    0.9371       576
          weighted avg     0.9385    0.9375    0.9373       576

```