# convnext_fgc_dsd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc_dsd_seed44/convnext_fgc_dsd/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9520 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9953 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9527 |
| Balanced Acc | 0.9506 |
| Macro-Sensitivity | 0.9506 |
| Macro-Specificity | 0.9761 |
| Macro-Precision | 0.9553 |
| AUC-PR (Macro) | 0.9910 |
| Cohen's Kappa | 0.9294 |
| MCC | 0.9302 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.43G |
| FLOPs | 8.87G |
| Params | 27.67M |
| Pure-Inference FPS | 1705.4 (0.59 ms/张) |
| End-to-End FPS | 274.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9760 | 0.9592 | 0.9312 | 0.9531 | 0.9860 | 0.9943 | 0.9902 | 208 |
| Malignant_lesions | 0.8864 | 0.9900 | 0.9750 | 0.9286 | 0.9519 | 0.9922 | 0.9841 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9792 | 0.9596 | 0.9744 | 0.9947 | 0.9993 | 0.9987 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=203 FP=15 TN=353 FN=5
- **Malignant_lesions**: TP=156 FP=4 TN=396 FN=20
- **body_of_stomach_normal**: TP=190 FP=8 TN=376 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 203 | 4 | 1 |
| Malignant_lesions | 13 | 156 | 7 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9312    0.9760    0.9531       208
     Malignant_lesions     0.9750    0.8864    0.9286       176
body_of_stomach_normal     0.9596    0.9896    0.9744       192

              accuracy                         0.9531       576
             macro avg     0.9553    0.9506    0.9520       576
          weighted avg     0.9540    0.9531    0.9527       576

```