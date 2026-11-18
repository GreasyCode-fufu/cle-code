# convnext_v5xrm2 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v5xrm2_seed43/convnext_v5xrm2/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9668 |
| Accuracy | 0.9670 |
| Macro-AUROC (OvR) | 0.9977 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9670 |
| Balanced Acc | 0.9680 |
| Macro-Sensitivity | 0.9680 |
| Macro-Specificity | 0.9838 |
| Macro-Precision | 0.9663 |
| AUC-PR (Macro) | 0.9953 |
| Cohen's Kappa | 0.9505 |
| MCC | 0.9508 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.69M |
| Pure-Inference FPS | 455.2 (2.20 ms/张) |
| End-to-End FPS | 177.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9891 | 0.9799 | 0.9582 | 0.9655 | 0.9967 | 0.9946 | 208 |
| Malignant_lesions | 0.9716 | 0.9700 | 0.9344 | 0.9526 | 0.9873 | 0.9966 | 0.9919 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9922 | 0.9845 | 0.9896 | 0.9974 | 0.9997 | 0.9995 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=4 TN=364 FN=13
- **Malignant_lesions**: TP=171 FP=12 TN=388 FN=5
- **body_of_stomach_normal**: TP=191 FP=3 TN=381 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 12 | 1 |
| Malignant_lesions | 3 | 171 | 2 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9799    0.9375    0.9582       208
     Malignant_lesions     0.9344    0.9716    0.9526       176
body_of_stomach_normal     0.9845    0.9948    0.9896       192

              accuracy                         0.9670       576
             macro avg     0.9663    0.9680    0.9668       576
          weighted avg     0.9676    0.9670    0.9670       576

```