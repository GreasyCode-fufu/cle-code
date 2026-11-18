# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_seed46/convnext_dph/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9375 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9938 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9384 |
| Balanced Acc | 0.9366 |
| Macro-Sensitivity | 0.9366 |
| Macro-Specificity | 0.9692 |
| Macro-Precision | 0.9412 |
| AUC-PR (Macro) | 0.9887 |
| Cohen's Kappa | 0.9085 |
| MCC | 0.9096 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 487.3 (2.05 ms/张) |
| End-to-End FPS | 189.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9565 | 0.9252 | 0.9384 | 0.9724 | 0.9915 | 0.9876 | 208 |
| Malignant_lesions | 0.8580 | 0.9850 | 0.9618 | 0.9069 | 0.9403 | 0.9906 | 0.9798 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9661 | 0.9366 | 0.9673 | 1.0000 | 0.9994 | 0.9988 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=16 TN=352 FN=10
- **Malignant_lesions**: TP=151 FP=6 TN=394 FN=25
- **body_of_stomach_normal**: TP=192 FP=13 TN=371 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 6 | 4 |
| Malignant_lesions | 16 | 151 | 9 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9252    0.9519    0.9384       208
     Malignant_lesions     0.9618    0.8580    0.9069       176
body_of_stomach_normal     0.9366    1.0000    0.9673       192

              accuracy                         0.9392       576
             macro avg     0.9412    0.9366    0.9375       576
          weighted avg     0.9402    0.9392    0.9384       576

```