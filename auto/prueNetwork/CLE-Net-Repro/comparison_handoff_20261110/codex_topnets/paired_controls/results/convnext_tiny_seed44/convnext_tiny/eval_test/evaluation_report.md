# convnext_tiny 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/cle_net_rerun_codex_20261109_v2/results/convnext_tiny_seed44/convnext_tiny/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9581 |
| Accuracy | 0.9583 |
| Macro-AUROC (OvR) | 0.9968 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9583 |
| Balanced Acc | 0.9582 |
| Macro-Sensitivity | 0.9582 |
| Macro-Specificity | 0.9791 |
| Macro-Precision | 0.9580 |
| AUC-PR (Macro) | 0.9938 |
| Cohen's Kappa | 0.9374 |
| MCC | 0.9374 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 1718.1 (0.58 ms/张) |
| End-to-End FPS | 224.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9471 | 0.9755 | 0.9563 | 0.9517 | 0.9703 | 0.9957 | 0.9933 | 208 |
| Malignant_lesions | 0.9432 | 0.9775 | 0.9486 | 0.9459 | 0.9751 | 0.9954 | 0.9892 | 176 |
| body_of_stomach_normal | 0.9844 | 0.9844 | 0.9692 | 0.9767 | 0.9921 | 0.9994 | 0.9989 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=197 FP=9 TN=359 FN=11
- **Malignant_lesions**: TP=166 FP=9 TN=391 FN=10
- **body_of_stomach_normal**: TP=189 FP=6 TN=378 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 197 | 9 | 2 |
| Malignant_lesions | 6 | 166 | 4 |
| body_of_stomach_normal | 3 | 0 | 189 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9563    0.9471    0.9517       208
     Malignant_lesions     0.9486    0.9432    0.9459       176
body_of_stomach_normal     0.9692    0.9844    0.9767       192

              accuracy                         0.9583       576
             macro avg     0.9580    0.9582    0.9581       576
          weighted avg     0.9583    0.9583    0.9583       576

```