# vit_base_patch16_224 评估报告

- 权重: `/home/login/cle/sunfuxiang/classifyML/weights_3class_hq_seed43/vit_base_patch16_224/vit_base_patch16_224_3class_hq_seed43_best.pth`
- 测试集: `/home/login/cle/sunfuxiang/datasets/3class/test`  (共 671 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8951 |
| Accuracy | 0.8942 |
| Macro-AUROC (OvR) | 0.9759 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8949 |
| Balanced Acc | 0.8942 |
| Macro-Sensitivity | 0.8942 |
| Macro-Specificity | 0.9466 |
| Macro-Precision | 0.8975 |
| AUC-PR (Macro) | 0.9563 |
| Cohen's Kappa | 0.8404 |
| MCC | 0.8410 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 16.86G |
| FLOPs | 33.73G |
| Params | 85.65M |
| Pure-Inference FPS | 870.6 (1.15 ms/张) |
| End-to-End FPS | 196.6 |
| 测试样本数 | 671 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.8945 | 0.9373 | 0.8980 | 0.8963 | 0.9351 | 0.9705 | 0.9603 | 256 |
| Malignant_lesions | 0.8905 | 0.9176 | 0.8311 | 0.8598 | 0.9484 | 0.9647 | 0.9252 | 210 |
| body_of_stomach_normal | 0.8976 | 0.9850 | 0.9634 | 0.9293 | 0.9563 | 0.9924 | 0.9834 | 205 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=229 FP=26 TN=389 FN=27
- **Malignant_lesions**: TP=187 FP=38 TN=423 FN=23
- **body_of_stomach_normal**: TP=184 FP=7 TN=459 FN=21

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 229 | 22 | 5 |
| Malignant_lesions | 21 | 187 | 2 |
| body_of_stomach_normal | 5 | 16 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8980    0.8945    0.8963       256
     Malignant_lesions     0.8311    0.8905    0.8598       210
body_of_stomach_normal     0.9634    0.8976    0.9293       205

              accuracy                         0.8942       671
             macro avg     0.8975    0.8942    0.8951       671
          weighted avg     0.8970    0.8942    0.8949       671

```