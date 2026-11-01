# convnext_dph_am 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_am_seed43/convnext_dph_am/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9526 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9914 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9530 |
| Balanced Acc | 0.9527 |
| Macro-Sensitivity | 0.9527 |
| Macro-Specificity | 0.9765 |
| Macro-Precision | 0.9527 |
| AUC-PR (Macro) | 0.9856 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9295 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 29.87M |
| Pure-Inference FPS | 1692.9 (0.59 ms/张) |
| End-to-End FPS | 269.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9701 | 0.9469 | 0.9446 | 0.9675 | 0.9882 | 0.9860 | 208 |
| Malignant_lesions | 0.9261 | 0.9725 | 0.9368 | 0.9314 | 0.9677 | 0.9864 | 0.9718 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9995 | 0.9990 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=11 TN=357 FN=12
- **Malignant_lesions**: TP=163 FP=11 TN=389 FN=13
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 10 | 2 |
| Malignant_lesions | 10 | 163 | 3 |
| body_of_stomach_normal | 1 | 1 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9469    0.9423    0.9446       208
     Malignant_lesions     0.9368    0.9261    0.9314       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9531       576
             macro avg     0.9527    0.9527    0.9526       576
          weighted avg     0.9529    0.9531    0.9530       576

```