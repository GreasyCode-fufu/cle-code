# convnext_fgc 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_seed44/convnext_fgc/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9647 |
| Accuracy | 0.9653 |
| Macro-AUROC (OvR) | 0.9970 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9651 |
| Balanced Acc | 0.9643 |
| Macro-Sensitivity | 0.9643 |
| Macro-Specificity | 0.9825 |
| Macro-Precision | 0.9655 |
| AUC-PR (Macro) | 0.9940 |
| Cohen's Kappa | 0.9478 |
| MCC | 0.9480 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.90G |
| Params | 27.69M |
| Pure-Inference FPS | 309.0 (3.24 ms/张) |
| End-to-End FPS | 153.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9663 | 0.9810 | 0.9663 | 0.9663 | 0.9810 | 0.9960 | 0.9936 | 208 |
| Malignant_lesions | 0.9318 | 0.9875 | 0.9704 | 0.9507 | 0.9705 | 0.9952 | 0.9891 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9792 | 0.9598 | 0.9770 | 0.9973 | 0.9996 | 0.9993 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=201 FP=7 TN=361 FN=7
- **Malignant_lesions**: TP=164 FP=5 TN=395 FN=12
- **body_of_stomach_normal**: TP=191 FP=8 TN=376 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 201 | 5 | 2 |
| Malignant_lesions | 6 | 164 | 6 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9663    0.9663    0.9663       208
     Malignant_lesions     0.9704    0.9318    0.9507       176
body_of_stomach_normal     0.9598    0.9948    0.9770       192

              accuracy                         0.9653       576
             macro avg     0.9655    0.9643    0.9647       576
          weighted avg     0.9654    0.9653    0.9651       576

```