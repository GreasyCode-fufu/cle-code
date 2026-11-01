# convnext_aep 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_aep_seed43/convnext_aep/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9457 |
| Accuracy | 0.9462 |
| Macro-AUROC (OvR) | 0.9954 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9461 |
| Balanced Acc | 0.9476 |
| Macro-Sensitivity | 0.9476 |
| Macro-Specificity | 0.9736 |
| Macro-Precision | 0.9456 |
| AUC-PR (Macro) | 0.9919 |
| Cohen's Kappa | 0.9192 |
| MCC | 0.9202 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 802.4 (1.25 ms/张) |
| End-to-End FPS | 217.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8990 | 0.9891 | 0.9791 | 0.9373 | 0.9455 | 0.9937 | 0.9909 | 208 |
| Malignant_lesions | 0.9489 | 0.9525 | 0.8978 | 0.9227 | 0.9769 | 0.9930 | 0.9856 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9792 | 0.9598 | 0.9770 | 0.9973 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=4 TN=364 FN=21
- **Malignant_lesions**: TP=167 FP=19 TN=381 FN=9
- **body_of_stomach_normal**: TP=191 FP=8 TN=376 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 18 | 3 |
| Malignant_lesions | 4 | 167 | 5 |
| body_of_stomach_normal | 0 | 1 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9791    0.8990    0.9373       208
     Malignant_lesions     0.8978    0.9489    0.9227       176
body_of_stomach_normal     0.9598    0.9948    0.9770       192

              accuracy                         0.9462       576
             macro avg     0.9456    0.9476    0.9457       576
          weighted avg     0.9478    0.9462    0.9461       576

```