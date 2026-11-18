# convnext_fgc4_dsd1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc4_dsd1_seed43/convnext_fgc4_dsd1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9545 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9936 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9548 |
| Balanced Acc | 0.9549 |
| Macro-Sensitivity | 0.9549 |
| Macro-Specificity | 0.9775 |
| Macro-Precision | 0.9541 |
| AUC-PR (Macro) | 0.9880 |
| Cohen's Kappa | 0.9322 |
| MCC | 0.9322 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.42G |
| FLOPs | 8.84G |
| Params | 27.68M |
| Pure-Inference FPS | 857.3 (1.17 ms/张) |
| End-to-End FPS | 232.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9755 | 0.9559 | 0.9466 | 0.9651 | 0.9911 | 0.9854 | 208 |
| Malignant_lesions | 0.9375 | 0.9700 | 0.9322 | 0.9348 | 0.9724 | 0.9906 | 0.9800 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9993 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=9 TN=359 FN=13
- **Malignant_lesions**: TP=165 FP=12 TN=388 FN=11
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 12 | 1 |
| Malignant_lesions | 7 | 165 | 4 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9559    0.9375    0.9466       208
     Malignant_lesions     0.9322    0.9375    0.9348       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9549       576
             macro avg     0.9541    0.9549    0.9545       576
          weighted avg     0.9548    0.9549    0.9548       576

```