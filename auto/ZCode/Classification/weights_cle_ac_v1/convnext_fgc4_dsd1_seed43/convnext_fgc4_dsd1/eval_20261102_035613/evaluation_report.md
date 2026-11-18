# convnext_fgc4_dsd1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_cle_ac_v1/convnext_fgc4_dsd1_seed43/convnext_fgc4_dsd1/model_best.pth`
- 测试集: `./datasets/cle_ac_v1/val`  (共 427 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9747 |
| Accuracy | 0.9789 |
| Macro-AUROC (OvR) | 0.9986 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9790 |
| Balanced Acc | 0.9796 |
| Macro-Sensitivity | 0.9796 |
| Macro-Specificity | 0.9902 |
| Macro-Precision | 0.9702 |
| AUC-PR (Macro) | 0.9958 |
| Cohen's Kappa | 0.9666 |
| MCC | 0.9668 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.42G |
| FLOPs | 8.84G |
| Params | 27.68M |
| Pure-Inference FPS | 997.2 (1.00 ms/张) |
| End-to-End FPS | 233.8 |
| 测试样本数 | 427 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9641 | 0.9914 | 0.9895 | 0.9766 | 0.9705 | 0.9982 | 0.9980 | 195 |
| Malignant_lesions | 0.9747 | 0.9828 | 0.9277 | 0.9506 | 0.9942 | 0.9977 | 0.9894 | 79 |
| body_of_stomach_normal | 1.0000 | 0.9964 | 0.9935 | 0.9967 | 1.0000 | 1.0000 | 1.0000 | 153 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=188 FP=2 TN=230 FN=7
- **Malignant_lesions**: TP=77 FP=6 TN=342 FN=2
- **body_of_stomach_normal**: TP=153 FP=1 TN=273 FN=0

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 188 | 6 | 1 |
| Malignant_lesions | 2 | 77 | 0 |
| body_of_stomach_normal | 0 | 0 | 153 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9895    0.9641    0.9766       195
     Malignant_lesions     0.9277    0.9747    0.9506        79
body_of_stomach_normal     0.9935    1.0000    0.9967       153

              accuracy                         0.9789       427
             macro avg     0.9702    0.9796    0.9747       427
          weighted avg     0.9795    0.9789    0.9790       427

```