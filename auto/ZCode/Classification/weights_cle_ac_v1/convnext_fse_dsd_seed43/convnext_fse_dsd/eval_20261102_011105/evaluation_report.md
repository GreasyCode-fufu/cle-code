# convnext_fse_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fse_dsd_seed43/convnext_fse_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9391 |
| Accuracy | 0.9392 |
| Macro-AUROC (OvR) | 0.9940 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9395 |
| Balanced Acc | 0.9395 |
| Macro-Sensitivity | 0.9395 |
| Macro-Specificity | 0.9697 |
| Macro-Precision | 0.9392 |
| AUC-PR (Macro) | 0.9885 |
| Cohen's Kappa | 0.9087 |
| MCC | 0.9089 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 27.78M |
| Pure-Inference FPS | 348.7 (2.87 ms/张) |
| End-to-End FPS | 164.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9620 | 0.9320 | 0.9275 | 0.9568 | 0.9920 | 0.9868 | 208 |
| Malignant_lesions | 0.9318 | 0.9525 | 0.8962 | 0.9136 | 0.9695 | 0.9903 | 0.9794 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9948 | 0.9893 | 0.9763 | 0.9820 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=14 TN=354 FN=16
- **Malignant_lesions**: TP=164 FP=19 TN=381 FN=12
- **body_of_stomach_normal**: TP=185 FP=2 TN=382 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 15 | 1 |
| Malignant_lesions | 11 | 164 | 1 |
| body_of_stomach_normal | 3 | 4 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9320    0.9231    0.9275       208
     Malignant_lesions     0.8962    0.9318    0.9136       176
body_of_stomach_normal     0.9893    0.9635    0.9763       192

              accuracy                         0.9392       576
             macro avg     0.9392    0.9395    0.9391       576
          weighted avg     0.9402    0.9392    0.9395       576

```