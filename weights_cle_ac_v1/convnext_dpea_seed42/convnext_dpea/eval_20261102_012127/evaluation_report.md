# convnext_dpea 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dpea_seed42/convnext_dpea/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9544 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9952 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9547 |
| Balanced Acc | 0.9537 |
| Macro-Sensitivity | 0.9537 |
| Macro-Specificity | 0.9771 |
| Macro-Precision | 0.9557 |
| AUC-PR (Macro) | 0.9898 |
| Cohen's Kappa | 0.9321 |
| MCC | 0.9323 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.44G |
| FLOPs | 8.88G |
| Params | 28.23M |
| Pure-Inference FPS | 344.5 (2.90 ms/张) |
| End-to-End FPS | 162.1 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9620 | 0.9343 | 0.9454 | 0.9752 | 0.9935 | 0.9900 | 208 |
| Malignant_lesions | 0.9148 | 0.9825 | 0.9583 | 0.9360 | 0.9632 | 0.9930 | 0.9809 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9992 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=14 TN=354 FN=9
- **Malignant_lesions**: TP=161 FP=7 TN=393 FN=15
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 7 | 2 |
| Malignant_lesions | 12 | 161 | 3 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9343    0.9567    0.9454       208
     Malignant_lesions     0.9583    0.9148    0.9360       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9549       576
             macro avg     0.9557    0.9537    0.9544       576
          weighted avg     0.9550    0.9549    0.9547       576

```