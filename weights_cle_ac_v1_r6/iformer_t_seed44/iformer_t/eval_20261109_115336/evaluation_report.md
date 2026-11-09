# iformer_t 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/iformer_t_seed44/iformer_t/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9630 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9926 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9714 |
| Balanced Acc | 0.9544 |
| Macro-Sensitivity | 0.9544 |
| Macro-Specificity | 0.9839 |
| Macro-Precision | 0.9736 |
| AUC-PR (Macro) | 0.9834 |
| Cohen's Kappa | 0.9550 |
| MCC | 0.9555 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 521.61M |
| FLOPs | 1.04G |
| Params | 2.63M |
| Pure-Inference FPS | 3625.9 (0.28 ms/张) |
| End-to-End FPS | 233.1 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9897 | 0.9612 | 0.9554 | 0.9723 | 0.9911 | 0.9938 | 0.9921 | 195 |
| Malignant_lesions | 0.8734 | 0.9943 | 0.9718 | 0.9200 | 0.9719 | 0.9838 | 0.9583 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=193 FP=9 TN=223 FN=2
- **Malignant_lesions**: TP=69 FP=2 TN=346 FN=10
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 193 | 2 | 0 |
| Malignant_lesions | 9 | 69 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9554    0.9897    0.9723       195
     Malignant_lesions     0.9718    0.8734    0.9200        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9719       427
             macro avg     0.9736    0.9544    0.9630       427
          weighted avg     0.9721    0.9719    0.9714       427

```