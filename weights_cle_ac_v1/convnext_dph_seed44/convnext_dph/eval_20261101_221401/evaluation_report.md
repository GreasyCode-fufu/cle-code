# convnext_dph 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_dph_seed44/convnext_dph/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9702 |
| Accuracy | 0.9705 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9704 |
| Balanced Acc | 0.9706 |
| Macro-Sensitivity | 0.9706 |
| Macro-Specificity | 0.9853 |
| Macro-Precision | 0.9699 |
| AUC-PR (Macro) | 0.9941 |
| Cohen's Kappa | 0.9556 |
| MCC | 0.9557 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.70M |
| Pure-Inference FPS | 1701.3 (0.59 ms/张) |
| End-to-End FPS | 271.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9567 | 0.9864 | 0.9755 | 0.9660 | 0.9758 | 0.9953 | 0.9930 | 208 |
| Malignant_lesions | 0.9602 | 0.9800 | 0.9548 | 0.9575 | 0.9825 | 0.9958 | 0.9905 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9896 | 0.9795 | 0.9871 | 0.9974 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=199 FP=5 TN=363 FN=9
- **Malignant_lesions**: TP=169 FP=8 TN=392 FN=7
- **body_of_stomach_normal**: TP=191 FP=4 TN=380 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 199 | 8 | 1 |
| Malignant_lesions | 4 | 169 | 3 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9755    0.9567    0.9660       208
     Malignant_lesions     0.9548    0.9602    0.9575       176
body_of_stomach_normal     0.9795    0.9948    0.9871       192

              accuracy                         0.9705       576
             macro avg     0.9699    0.9706    0.9702       576
          weighted avg     0.9705    0.9705    0.9704       576

```