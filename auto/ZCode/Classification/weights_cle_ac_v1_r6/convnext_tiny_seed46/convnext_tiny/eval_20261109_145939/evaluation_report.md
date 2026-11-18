# convnext_tiny 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_seed46/convnext_tiny/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9506 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9941 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9511 |
| Balanced Acc | 0.9503 |
| Macro-Sensitivity | 0.9503 |
| Macro-Specificity | 0.9756 |
| Macro-Precision | 0.9511 |
| AUC-PR (Macro) | 0.9887 |
| Cohen's Kappa | 0.9269 |
| MCC | 0.9270 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 456.3 (2.19 ms/张) |
| End-to-End FPS | 184.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9674 | 0.9426 | 0.9448 | 0.9700 | 0.9925 | 0.9855 | 208 |
| Malignant_lesions | 0.9091 | 0.9750 | 0.9412 | 0.9249 | 0.9606 | 0.9899 | 0.9808 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9844 | 0.9695 | 0.9820 | 0.9974 | 0.9999 | 0.9997 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=12 TN=356 FN=11
- **Malignant_lesions**: TP=160 FP=10 TN=390 FN=16
- **body_of_stomach_normal**: TP=191 FP=6 TN=378 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 10 | 1 |
| Malignant_lesions | 11 | 160 | 5 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9426    0.9471    0.9448       208
     Malignant_lesions     0.9412    0.9091    0.9249       176
body_of_stomach_normal     0.9695    0.9948    0.9820       192

              accuracy                         0.9514       576
             macro avg     0.9511    0.9503    0.9506       576
          weighted avg     0.9511    0.9514    0.9511       576

```