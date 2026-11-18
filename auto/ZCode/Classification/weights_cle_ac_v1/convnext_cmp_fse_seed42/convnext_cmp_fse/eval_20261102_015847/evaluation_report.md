# convnext_cmp_fse 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_cmp_fse_seed42/convnext_cmp_fse/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9757 |
| Accuracy | 0.9813 |
| Macro-AUROC (OvR) | 0.9984 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9810 |
| Balanced Acc | 0.9688 |
| Macro-Sensitivity | 0.9688 |
| Macro-Specificity | 0.9890 |
| Macro-Precision | 0.9838 |
| AUC-PR (Macro) | 0.9951 |
| Cohen's Kappa | 0.9700 |
| MCC | 0.9704 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.46G |
| FLOPs | 8.91G |
| Params | 28.25M |
| Pure-Inference FPS | 465.0 (2.15 ms/张) |
| End-to-End FPS | 183.9 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9949 | 0.9698 | 0.9652 | 0.9798 | 0.9956 | 0.9982 | 0.9979 | 195 |
| Malignant_lesions | 0.9114 | 0.9971 | 0.9863 | 0.9474 | 0.9802 | 0.9969 | 0.9873 | 79 |
| body_of_stomach_normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=7 TN=225 FN=1
- **Malignant_lesions**: TP=72 FP=1 TN=347 FN=7
- **body_of_stomach_normal**: TP=153 FP=0 TN=274 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 1 | 0 |
| Malignant_lesions | 7 | 72 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9652    0.9949    0.9798       195
     Malignant_lesions     0.9863    0.9114    0.9474        79
body_of_stomach_normal     1.0000    1.0000    1.0000       153

              accuracy                         0.9813       427
             macro avg     0.9838    0.9688    0.9757       427
          weighted avg     0.9816    0.9813    0.9810       427

```