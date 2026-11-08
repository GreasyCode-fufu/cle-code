# mbv3_r6_s2_lsbb 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_lsbb_seed42/mbv3_r6_s2_lsbb/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9248 |
| Accuracy | 0.9253 |
| Macro-AUROC (OvR) | 0.9902 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9252 |
| Balanced Acc | 0.9227 |
| Macro-Sensitivity | 0.9227 |
| Macro-Specificity | 0.9619 |
| Macro-Precision | 0.9296 |
| AUC-PR (Macro) | 0.9803 |
| Cohen's Kappa | 0.8875 |
| MCC | 0.8887 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 235.94M |
| FLOPs | 471.88M |
| Params | 3.27M |
| Pure-Inference FPS | 3377.9 (0.30 ms/张) |
| End-to-End FPS | 225.0 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9423 | 0.9185 | 0.8673 | 0.9032 | 0.9657 | 0.9859 | 0.9790 | 208 |
| Malignant_lesions | 0.8466 | 0.9750 | 0.9371 | 0.8896 | 0.9353 | 0.9859 | 0.9642 | 176 |
| body_of_stomach_normal | 0.9792 | 0.9922 | 0.9843 | 0.9817 | 0.9896 | 0.9988 | 0.9977 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=196 FP=30 TN=338 FN=12
- **Malignant_lesions**: TP=149 FP=10 TN=390 FN=27
- **body_of_stomach_normal**: TP=188 FP=3 TN=381 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 196 | 10 | 2 |
| Malignant_lesions | 26 | 149 | 1 |
| body_of_stomach_normal | 4 | 0 | 188 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8673    0.9423    0.9032       208
     Malignant_lesions     0.9371    0.8466    0.8896       176
body_of_stomach_normal     0.9843    0.9792    0.9817       192

              accuracy                         0.9253       576
             macro avg     0.9296    0.9227    0.9248       576
          weighted avg     0.9276    0.9253    0.9252       576

```