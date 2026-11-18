# mbv3_r6_s2_lsbb 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/mbv3_r6_s2_lsbb_seed43/mbv3_r6_s2_lsbb/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9708 |
| Accuracy | 0.9766 |
| Macro-AUROC (OvR) | 0.9984 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9768 |
| Balanced Acc | 0.9779 |
| Macro-Sensitivity | 0.9779 |
| Macro-Specificity | 0.9897 |
| Macro-Precision | 0.9647 |
| AUC-PR (Macro) | 0.9946 |
| Cohen's Kappa | 0.9630 |
| MCC | 0.9634 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 235.94M |
| FLOPs | 471.88M |
| Params | 3.27M |
| Pure-Inference FPS | 2525.2 (0.40 ms/张) |
| End-to-End FPS | 152.3 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9590 | 0.9957 | 0.9947 | 0.9765 | 0.9665 | 0.9985 | 0.9984 | 195 |
| Malignant_lesions | 0.9747 | 0.9770 | 0.9059 | 0.9390 | 0.9942 | 0.9967 | 0.9855 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=187 FP=1 TN=231 FN=8
- **Malignant_lesions**: TP=77 FP=8 TN=340 FN=2
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 187 | 8 | 0 |
| Malignant_lesions | 1 | 77 | 1 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9947    0.9590    0.9765       195
     Malignant_lesions     0.9059    0.9747    0.9390        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9766       427
             macro avg     0.9647    0.9779    0.9708       427
          weighted avg     0.9778    0.9766    0.9768       427

```