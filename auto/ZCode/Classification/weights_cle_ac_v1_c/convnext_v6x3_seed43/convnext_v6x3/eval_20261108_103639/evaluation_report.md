# convnext_v6x3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_c/convnext_v6x3_seed43/convnext_v6x3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9508 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9958 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9512 |
| Balanced Acc | 0.9512 |
| Macro-Sensitivity | 0.9512 |
| Macro-Specificity | 0.9758 |
| Macro-Precision | 0.9507 |
| AUC-PR (Macro) | 0.9916 |
| Cohen's Kappa | 0.9269 |
| MCC | 0.9272 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.26G |
| Params | 28.69M |
| Pure-Inference FPS | 1644.4 (0.61 ms/张) |
| End-to-End FPS | 265.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9783 | 0.9604 | 0.9463 | 0.9626 | 0.9945 | 0.9907 | 208 |
| Malignant_lesions | 0.9261 | 0.9725 | 0.9368 | 0.9314 | 0.9677 | 0.9933 | 0.9848 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9766 | 0.9550 | 0.9745 | 0.9973 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=8 TN=360 FN=14
- **Malignant_lesions**: TP=163 FP=11 TN=389 FN=13
- **body_of_stomach_normal**: TP=191 FP=9 TN=375 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 11 | 3 |
| Malignant_lesions | 7 | 163 | 6 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9604    0.9327    0.9463       208
     Malignant_lesions     0.9368    0.9261    0.9314       176
body_of_stomach_normal     0.9550    0.9948    0.9745       192

              accuracy                         0.9514       576
             macro avg     0.9507    0.9512    0.9508       576
          weighted avg     0.9514    0.9514    0.9512       576

```