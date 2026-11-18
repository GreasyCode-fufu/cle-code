# convnext_ef_r75 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_ef_r75_seed43/convnext_ef_r75/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9594 |
| Accuracy | 0.9601 |
| Macro-AUROC (OvR) | 0.9964 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9599 |
| Balanced Acc | 0.9596 |
| Macro-Sensitivity | 0.9596 |
| Macro-Specificity | 0.9801 |
| Macro-Precision | 0.9592 |
| AUC-PR (Macro) | 0.9927 |
| Cohen's Kappa | 0.9400 |
| MCC | 0.9400 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.55G |
| Params | 26.14M |
| Pure-Inference FPS | 362.0 (2.76 ms/张) |
| End-to-End FPS | 163.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9783 | 0.9610 | 0.9540 | 0.9704 | 0.9957 | 0.9925 | 208 |
| Malignant_lesions | 0.9318 | 0.9725 | 0.9371 | 0.9345 | 0.9701 | 0.9936 | 0.9860 | 176 |
| body_of_stomach_normal | 1.0000 | 0.9896 | 0.9796 | 0.9897 | 1.0000 | 0.9999 | 0.9997 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=8 TN=360 FN=11
- **Malignant_lesions**: TP=164 FP=11 TN=389 FN=12
- **body_of_stomach_normal**: TP=192 FP=4 TN=380 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 11 | 0 |
| Malignant_lesions | 8 | 164 | 4 |
| body_of_stomach_normal | 0 | 0 | 192 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9610    0.9471    0.9540       208
     Malignant_lesions     0.9371    0.9318    0.9345       176
body_of_stomach_normal     0.9796    1.0000    0.9897       192

              accuracy                         0.9601       576
             macro avg     0.9592    0.9596    0.9594       576
          weighted avg     0.9599    0.9601    0.9599       576

```