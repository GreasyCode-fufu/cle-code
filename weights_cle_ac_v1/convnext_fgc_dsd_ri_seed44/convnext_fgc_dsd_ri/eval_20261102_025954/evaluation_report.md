# convnext_fgc_dsd_ri 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_dsd_ri_seed44/convnext_fgc_dsd_ri/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9576 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9961 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9580 |
| Balanced Acc | 0.9565 |
| Macro-Sensitivity | 0.9565 |
| Macro-Specificity | 0.9788 |
| Macro-Precision | 0.9601 |
| AUC-PR (Macro) | 0.9925 |
| Cohen's Kappa | 0.9373 |
| MCC | 0.9378 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 607.1 (1.65 ms/张) |
| End-to-End FPS | 205.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9712 | 0.9620 | 0.9352 | 0.9528 | 0.9833 | 0.9948 | 0.9909 | 208 |
| Malignant_lesions | 0.9034 | 0.9900 | 0.9755 | 0.9381 | 0.9588 | 0.9938 | 0.9871 | 176 |
| body_of_stomach_normal | 0.9948 | 0.9844 | 0.9695 | 0.9820 | 0.9974 | 0.9997 | 0.9994 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=202 FP=14 TN=354 FN=6
- **Malignant_lesions**: TP=159 FP=4 TN=396 FN=17
- **body_of_stomach_normal**: TP=191 FP=6 TN=378 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 202 | 4 | 2 |
| Malignant_lesions | 13 | 159 | 4 |
| body_of_stomach_normal | 1 | 0 | 191 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9352    0.9712    0.9528       208
     Malignant_lesions     0.9755    0.9034    0.9381       176
body_of_stomach_normal     0.9695    0.9948    0.9820       192

              accuracy                         0.9583       576
             macro avg     0.9601    0.9565    0.9576       576
          weighted avg     0.9589    0.9583    0.9580       576

```