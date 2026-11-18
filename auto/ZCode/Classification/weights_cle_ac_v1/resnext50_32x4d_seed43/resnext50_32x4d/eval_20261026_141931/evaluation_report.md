# resnext50_32x4d 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/resnext50_32x4d_seed43/resnext50_32x4d/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9232 |
| Accuracy | 0.9236 |
| Macro-AUROC (OvR) | 0.9866 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9235 |
| Balanced Acc | 0.9213 |
| Macro-Sensitivity | 0.9213 |
| Macro-Specificity | 0.9611 |
| Macro-Precision | 0.9269 |
| AUC-PR (Macro) | 0.9733 |
| Cohen's Kappa | 0.8849 |
| MCC | 0.8858 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.29G |
| FLOPs | 8.57G |
| Params | 22.99M |
| Pure-Inference FPS | 953.0 (1.05 ms/张) |
| End-to-End FPS | 213.2 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9239 | 0.8750 | 0.9074 | 0.9659 | 0.9813 | 0.9714 | 208 |
| Malignant_lesions | 0.8580 | 0.9725 | 0.9321 | 0.8935 | 0.9396 | 0.9805 | 0.9521 | 176 |
| body_of_stomach_normal | 0.9635 | 0.9870 | 0.9737 | 0.9686 | 0.9819 | 0.9980 | 0.9964 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=28 TN=340 FN=12
- **Malignant_lesions**: TP=151 FP=11 TN=389 FN=25
- **body_of_stomach_normal**: TP=185 FP=5 TN=379 FN=7

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 11 | 1 |
| Malignant_lesions | 21 | 151 | 4 |
| body_of_stomach_normal | 7 | 0 | 185 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8750    0.9423    0.9074       208
     Malignant_lesions     0.9321    0.8580    0.8935       176
body_of_stomach_normal     0.9737    0.9635    0.9686       192

              accuracy                         0.9236       576
             macro avg     0.9269    0.9213    0.9232       576
          weighted avg     0.9253    0.9236    0.9235       576

```