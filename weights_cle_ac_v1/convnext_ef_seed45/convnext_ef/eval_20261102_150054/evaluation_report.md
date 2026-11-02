# convnext_ef 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_ef_seed45/convnext_ef/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9543 |
| Accuracy | 0.9549 |
| Macro-AUROC (OvR) | 0.9949 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9547 |
| Balanced Acc | 0.9554 |
| Macro-Sensitivity | 0.9554 |
| Macro-Specificity | 0.9777 |
| Macro-Precision | 0.9542 |
| AUC-PR (Macro) | 0.9900 |
| Cohen's Kappa | 0.9322 |
| MCC | 0.9327 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.82G |
| Params | 18.40M |
| Pure-Inference FPS | 415.5 (2.41 ms/张) |
| End-to-End FPS | 176.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9231 | 0.9891 | 0.9796 | 0.9505 | 0.9579 | 0.9925 | 0.9890 | 208 |
| Malignant_lesions | 0.9432 | 0.9700 | 0.9326 | 0.9379 | 0.9749 | 0.9925 | 0.9817 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9740 | 0.9505 | 0.9746 | 1.0000 | 0.9996 | 0.9992 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=192 FP=4 TN=364 FN=16
- **Malignant_lesions**: TP=166 FP=12 TN=388 FN=10
- **body_of_stomach_normal**: TP=192 FP=10 TN=374 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 192 | 12 | 4 |
| Malignant_lesions | 4 | 166 | 6 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9796    0.9231    0.9505       208
     Malignant_lesions     0.9326    0.9432    0.9379       176
body_of_stomach_normal     0.9505    1.0000    0.9746       192

              accuracy                         0.9549       576
             macro avg     0.9542    0.9554    0.9543       576
          weighted avg     0.9555    0.9549    0.9547       576

```