# convnext_tiny 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/cle_net_rerun_codex_20261109_v2/results/convnext_tiny_seed42/convnext_tiny/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9530 |
| Accuracy | 0.9531 |
| Macro-AUROC (OvR) | 0.9965 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9530 |
| Balanced Acc | 0.9531 |
| Macro-Sensitivity | 0.9531 |
| Macro-Specificity | 0.9765 |
| Macro-Precision | 0.9530 |
| AUC-PR (Macro) | 0.9933 |
| Cohen's Kappa | 0.9295 |
| MCC | 0.9296 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1720.0 (0.58 ms/张) |
| End-to-End FPS | 223.6 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9728 | 0.9512 | 0.9443 | 0.9650 | 0.9951 | 0.9916 | 208 |
| Malignant_lesions | 0.9375 | 0.9775 | 0.9483 | 0.9429 | 0.9726 | 0.9952 | 0.9897 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9792 | 0.9594 | 0.9717 | 0.9921 | 0.9992 | 0.9985 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=10 TN=358 FN=13
- **Malignant_lesions**: TP=165 FP=9 TN=391 FN=11
- **body_of_stomach_normal**: TP=189 FP=8 TN=376 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 9 | 4 |
| Malignant_lesions | 7 | 165 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9512    0.9375    0.9443       208
     Malignant_lesions     0.9483    0.9375    0.9429       176
body_of_stomach_normal     0.9594    0.9844    0.9717       192

              accuracy                         0.9531       576
             macro avg     0.9530    0.9531    0.9530       576
          weighted avg     0.9530    0.9531    0.9530       576

```