# mbv3_l2sp_a01 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_l2sp_a01_seed44/mbv3_l2sp_a01/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9273 |
| Accuracy | 0.9271 |
| Macro-AUROC (OvR) | 0.9919 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9273 |
| Balanced Acc | 0.9244 |
| Macro-Sensitivity | 0.9244 |
| Macro-Specificity | 0.9626 |
| Macro-Precision | 0.9337 |
| AUC-PR (Macro) | 0.9851 |
| Cohen's Kappa | 0.8900 |
| MCC | 0.8918 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 3789.9 (0.26 ms/张) |
| End-to-End FPS | 174.8 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9615 | 0.9158 | 0.8658 | 0.9112 | 0.9768 | 0.9882 | 0.9810 | 208 |
| Malignant_lesions | 0.8636 | 0.9850 | 0.9620 | 0.9102 | 0.9426 | 0.9907 | 0.9799 | 176 |
| body_of_stomach_normal | 0.9479 | 0.9870 | 0.9733 | 0.9604 | 0.9743 | 0.9969 | 0.9944 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=200 FP=31 TN=337 FN=8
- **Malignant_lesions**: TP=152 FP=6 TN=394 FN=24
- **body_of_stomach_normal**: TP=182 FP=5 TN=379 FN=10

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 200 | 5 | 3 |
| Malignant_lesions | 22 | 152 | 2 |
| body_of_stomach_normal | 9 | 1 | 182 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.8658    0.9615    0.9112       208
     Malignant_lesions     0.9620    0.8636    0.9102       176
body_of_stomach_normal     0.9733    0.9479    0.9604       192

              accuracy                         0.9271       576
             macro avg     0.9337    0.9244    0.9273       576
          weighted avg     0.9310    0.9271    0.9273       576

```