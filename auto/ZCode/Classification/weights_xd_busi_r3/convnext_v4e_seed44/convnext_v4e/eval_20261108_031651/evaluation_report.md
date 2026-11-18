# convnext_v4e 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r3/convnext_v4e_seed44/convnext_v4e/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8300 |
| Accuracy | 0.8390 |
| Macro-AUROC (OvR) | 0.9461 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8402 |
| Balanced Acc | 0.8602 |
| Macro-Sensitivity | 0.8602 |
| Macro-Specificity | 0.9170 |
| Macro-Precision | 0.8113 |
| AUC-PR (Macro) | 0.9169 |
| Cohen's Kappa | 0.7348 |
| MCC | 0.7389 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.28G |
| FLOPs | 8.56G |
| Params | 21.17M |
| Pure-Inference FPS | 454.8 (2.20 ms/张) |
| End-to-End FPS | 136.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8182 | 0.9038 | 0.9153 | 0.8640 | 0.7966 | 0.9162 | 0.9470 | 66 |
| malignant | 0.8125 | 0.9186 | 0.7879 | 0.8000 | 0.9294 | 0.9404 | 0.8771 | 32 |
| normal | 0.9500 | 0.9286 | 0.7308 | 0.8261 | 0.9891 | 0.9816 | 0.9267 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=54 FP=5 TN=47 FN=12
- **malignant**: TP=26 FP=7 TN=79 FN=6
- **normal**: TP=19 FP=7 TN=91 FN=1

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 54 | 7 | 5 |
| malignant | 4 | 26 | 2 |
| normal | 1 | 0 | 19 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9153    0.8182    0.8640        66
   malignant     0.7879    0.8125    0.8000        32
      normal     0.7308    0.9500    0.8261        20

    accuracy                         0.8390       118
   macro avg     0.8113    0.8602    0.8300       118
weighted avg     0.8494    0.8390    0.8402       118

```