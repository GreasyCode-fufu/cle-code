# deit_small_patch16_224 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/deit_small_patch16_224_seed44/deit_small_patch16_224/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9369 |
| Accuracy | 0.9375 |
| Macro-AUROC (OvR) | 0.9912 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9372 |
| Balanced Acc | 0.9357 |
| Macro-Sensitivity | 0.9357 |
| Macro-Specificity | 0.9683 |
| Macro-Precision | 0.9393 |
| AUC-PR (Macro) | 0.9847 |
| Cohen's Kappa | 0.9059 |
| MCC | 0.9064 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.25G |
| FLOPs | 8.50G |
| Params | 21.59M |
| Pure-Inference FPS | 930.3 (1.07 ms/张) |
| End-to-End FPS | 231.5 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9457 | 0.9078 | 0.9271 | 0.9694 | 0.9864 | 0.9796 | 208 |
| Malignant_lesions | 0.8807 | 0.9800 | 0.9509 | 0.9145 | 0.9492 | 0.9891 | 0.9783 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9792 | 0.9592 | 0.9691 | 0.9895 | 0.9982 | 0.9963 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=20 TN=348 FN=11
- **Malignant_lesions**: TP=155 FP=8 TN=392 FN=21
- **body_of_stomach_normal**: TP=188 FP=8 TN=376 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 7 | 4 |
| Malignant_lesions | 17 | 155 | 4 |
| body_of_stomach_normal | 3 | 1 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9078    0.9471    0.9271       208
     Malignant_lesions     0.9509    0.8807    0.9145       176
body_of_stomach_normal     0.9592    0.9792    0.9691       192

              accuracy                         0.9375       576
             macro avg     0.9393    0.9357    0.9369       576
          weighted avg     0.9381    0.9375    0.9372       576

```