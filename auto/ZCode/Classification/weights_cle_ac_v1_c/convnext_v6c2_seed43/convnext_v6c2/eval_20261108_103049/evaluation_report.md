# convnext_v6c2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6c2_seed43/convnext_v6c2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9635 |
| Accuracy | 0.9635 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9635 |
| Balanced Acc | 0.9643 |
| Macro-Sensitivity | 0.9643 |
| Macro-Specificity | 0.9819 |
| Macro-Precision | 0.9629 |
| AUC-PR (Macro) | 0.9922 |
| Cohen's Kappa | 0.9452 |
| MCC | 0.9454 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 5.78G |
| FLOPs | 11.57G |
| Params | 30.61M |
| Pure-Inference FPS | 407.5 (2.45 ms/张) |
| End-to-End FPS | 94.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9837 | 0.9701 | 0.9535 | 0.9653 | 0.9949 | 0.9912 | 208 |
| Malignant_lesions | 0.9659 | 0.9725 | 0.9392 | 0.9524 | 0.9848 | 0.9938 | 0.9862 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9896 | 0.9794 | 0.9845 | 0.9948 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=6 TN=362 FN=13
- **Malignant_lesions**: TP=170 FP=11 TN=389 FN=6
- **body_of_stomach_normal**: TP=190 FP=4 TN=380 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 11 | 2 |
| Malignant_lesions | 4 | 170 | 2 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9701    0.9375    0.9535       208
     Malignant_lesions     0.9392    0.9659    0.9524       176
body_of_stomach_normal     0.9794    0.9896    0.9845       192

              accuracy                         0.9635       576
             macro avg     0.9629    0.9643    0.9635       576
          weighted avg     0.9638    0.9635    0.9635       576

```