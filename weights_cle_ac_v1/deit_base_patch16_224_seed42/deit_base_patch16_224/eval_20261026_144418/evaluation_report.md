# deit_base_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/deit_base_patch16_224_seed42/deit_base_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9478 |
| Accuracy | 0.9479 |
| Macro-AUROC (OvR) | 0.9933 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9480 |
| Balanced Acc | 0.9482 |
| Macro-Sensitivity | 0.9482 |
| Macro-Specificity | 0.9740 |
| Macro-Precision | 0.9475 |
| AUC-PR (Macro) | 0.9883 |
| Cohen's Kappa | 0.9217 |
| MCC | 0.9218 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 334.4 (2.99 ms/张) |
| End-to-End FPS | 158.3 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9279 | 0.9674 | 0.9415 | 0.9346 | 0.9596 | 0.9894 | 0.9859 | 208 |
| Malignant_lesions | 0.9375 | 0.9625 | 0.9167 | 0.9270 | 0.9722 | 0.9910 | 0.9798 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9996 | 0.9991 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=12 TN=356 FN=15
- **Malignant_lesions**: TP=165 FP=15 TN=385 FN=11
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 14 | 1 |
| Malignant_lesions | 9 | 165 | 2 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9415    0.9279    0.9346       208
     Malignant_lesions     0.9167    0.9375    0.9270       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9479       576
             macro avg     0.9475    0.9482    0.9478       576
          weighted avg     0.9482    0.9479    0.9480       576

```