# convnext_tiny_fz3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1_r6/convnext_tiny_fz3_seed44/convnext_tiny_fz3/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9645 |
| Accuracy | 0.9719 |
| Macro-AUROC (OvR) | 0.9979 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9720 |
| Balanced Acc | 0.9665 |
| Macro-Sensitivity | 0.9665 |
| Macro-Specificity | 0.9861 |
| Macro-Precision | 0.9626 |
| AUC-PR (Macro) | 0.9934 |
| Cohen's Kappa | 0.9554 |
| MCC | 0.9554 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.45G |
| FLOPs | 8.91G |
| Params | 27.80M |
| Pure-Inference FPS | 768.6 (1.30 ms/张) |
| End-to-End FPS | 95.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9692 | 0.9784 | 0.9742 | 0.9717 | 0.9742 | 0.9980 | 0.9977 | 195 |
| Malignant_lesions | 0.9367 | 0.9799 | 0.9136 | 0.9250 | 0.9855 | 0.9958 | 0.9827 | 79 |
| body_of_stomach_normal | 0.9935 | 1.0000 | 1.0000 | 0.9967 | 0.9964 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=189 FP=5 TN=227 FN=6
- **Malignant_lesions**: TP=74 FP=7 TN=341 FN=5
- **body_of_stomach_normal**: TP=152 FP=0 TN=274 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 189 | 6 | 0 |
| Malignant_lesions | 5 | 74 | 0 |
| body_of_stomach_normal | 0 | 1 | 152 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9742    0.9692    0.9717       195
     Malignant_lesions     0.9136    0.9367    0.9250        79
body_of_stomach_normal     1.0000    0.9935    0.9967       153

              accuracy                         0.9719       427
             macro avg     0.9626    0.9665    0.9645       427
          weighted avg     0.9722    0.9719    0.9720       427

```