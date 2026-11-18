# transnext_tiny 评估报告

- 权重: `/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks/results/transnext_tiny/seed42/transnext_tiny/model_best.pth`
- 测试集: `/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test`  (共 576 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.9495 |
| Accuracy | 0.9497 |
| Macro-AUROC (OvR) | 0.9937 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.9497 |
| Balanced Acc | 0.9498 |
| Macro-Sensitivity | 0.9498 |
| Macro-Specificity | 0.9749 |
| Macro-Precision | 0.9493 |
| AUC-PR (Macro) | 0.9878 |
| Cohen's Kappa | 0.9243 |
| MCC | 0.9244 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 5.53G |
| FLOPs | 11.07G |
| Params | 27.63M |
| Pure-Inference FPS | 115.5 (8.66 ms/张) |
| End-to-End FPS | 14.4 |
| 测试样本数 | 576 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| Benign_lesions | 0.9375 | 0.9701 | 0.9466 | 0.9420 | 0.9649 | 0.9906 | 0.9863 | 208 |
| Malignant_lesions | 0.9432 | 0.9650 | 0.9222 | 0.9326 | 0.9747 | 0.9919 | 0.9799 | 176 |
| body_of_stomach_normal | 0.9688 | 0.9896 | 0.9789 | 0.9738 | 0.9845 | 0.9986 | 0.9972 | 192 |

### 每类 TP/FP/TN/FN

- **Benign_lesions**: TP=195 FP=11 TN=357 FN=13
- **Malignant_lesions**: TP=166 FP=14 TN=386 FN=10
- **body_of_stomach_normal**: TP=186 FP=4 TN=380 FN=6

## 混淆矩阵 (行=真实, 列=预测)

| | Benign_lesions | Malignant_lesions | body_of_stomach_normal |
|---|---|---|---|
| Benign_lesions | 195 | 11 | 2 |
| Malignant_lesions | 8 | 166 | 2 |
| body_of_stomach_normal | 3 | 3 | 186 |

## Classification Report (sklearn)

```
                        precision    recall  f1-score   support

        Benign_lesions     0.9466    0.9375    0.9420       208
     Malignant_lesions     0.9222    0.9432    0.9326       176
body_of_stomach_normal     0.9789    0.9688    0.9738       192

              accuracy                         0.9497       576
             macro avg     0.9493    0.9498    0.9495       576
          weighted avg     0.9499    0.9497    0.9497       576

```