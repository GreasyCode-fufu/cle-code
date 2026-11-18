# lsnet_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/lsnet_t_seed42/lsnet_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9253 |
| Accuracy | 0.9253 |
| Macro-AUROC (OvR) | 0.9872 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9256 |
| Balanced Acc | 0.9239 |
| Macro-Sensitivity | 0.9239 |
| Macro-Specificity | 0.9622 |
| Macro-Precision | 0.9276 |
| AUC-PR (Macro) | 0.9759 |
| Cohen's Kappa | 0.8876 |
| MCC | 0.8881 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 320.32M |
| FLOPs | 640.63M |
| Params | 11.02M |
| Pure-Inference FPS | 4356.4 (0.23 ms/张) |
| End-to-End FPS | 226.9 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9327 | 0.9293 | 0.8818 | 0.9065 | 0.9607 | 0.9811 | 0.9701 | 208 |
| Malignant_lesions | 0.8807 | 0.9625 | 0.9118 | 0.8960 | 0.9483 | 0.9821 | 0.9609 | 176 |
| body_of_stomach_normal | 0.9583 | 0.9948 | 0.9892 | 0.9735 | 0.9795 | 0.9983 | 0.9966 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=194 FP=26 TN=342 FN=14
- **Malignant_lesions**: TP=155 FP=15 TN=385 FN=21
- **body_of_stomach_normal**: TP=184 FP=2 TN=382 FN=8

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 194 | 12 | 2 |
| Malignant_lesions | 21 | 155 | 0 |
| body_of_stomach_normal | 5 | 3 | 184 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8818    0.9327    0.9065       208
     Malignant_lesions     0.9118    0.8807    0.8960       176
body_of_stomach_normal     0.9892    0.9583    0.9735       192

              accuracy                         0.9253       576
             macro avg     0.9276    0.9239    0.9253       576
          weighted avg     0.9268    0.9253    0.9256       576

```