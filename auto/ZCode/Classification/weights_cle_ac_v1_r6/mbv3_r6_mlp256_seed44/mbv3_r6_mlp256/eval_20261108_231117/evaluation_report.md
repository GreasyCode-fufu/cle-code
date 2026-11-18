# mbv3_r6_mlp256 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_mlp256_seed44/mbv3_r6_mlp256/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9513 |
| Accuracy | 0.9514 |
| Macro-AUROC (OvR) | 0.9910 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9513 |
| Balanced Acc | 0.9502 |
| Macro-Sensitivity | 0.9502 |
| Macro-Specificity | 0.9753 |
| Macro-Precision | 0.9530 |
| AUC-PR (Macro) | 0.9829 |
| Cohen's Kappa | 0.9268 |
| MCC | 0.9272 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.93M |
| FLOPs | 463.85M |
| Params | 3.22M |
| Pure-Inference FPS | 4237.1 (0.24 ms/张) |
| End-to-End FPS | 200.7 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9519 | 0.9538 | 0.9209 | 0.9362 | 0.9723 | 0.9862 | 0.9767 | 208 |
| Malignant_lesions | 0.9091 | 0.9850 | 0.9639 | 0.9357 | 0.9610 | 0.9892 | 0.9753 | 176 |
| body_of_stomach_normal | 0.9896 | 0.9870 | 0.9744 | 0.9819 | 0.9948 | 0.9974 | 0.9967 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=198 FP=17 TN=351 FN=10
- **Malignant_lesions**: TP=160 FP=6 TN=394 FN=16
- **body_of_stomach_normal**: TP=190 FP=5 TN=379 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 198 | 6 | 4 |
| Malignant_lesions | 15 | 160 | 1 |
| body_of_stomach_normal | 2 | 0 | 190 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9209    0.9519    0.9362       208
     Malignant_lesions     0.9639    0.9091    0.9357       176
body_of_stomach_normal     0.9744    0.9896    0.9819       192

              accuracy                         0.9514       576
             macro avg     0.9530    0.9502    0.9513       576
          weighted avg     0.9519    0.9514    0.9513       576

```