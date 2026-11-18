# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_svd_seed44/convnext_svd/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8290 |
| Accuracy | 0.8390 |
| Macro-AUROC (OvR) | 0.9372 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8398 |
| Balanced Acc | 0.8424 |
| Macro-Sensitivity | 0.8424 |
| Macro-Specificity | 0.9110 |
| Macro-Precision | 0.8180 |
| AUC-PR (Macro) | 0.8801 |
| Cohen's Kappa | 0.7300 |
| MCC | 0.7312 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 413.2 (2.42 ms/张) |
| End-to-End FPS | 134.4 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8333 | 0.8654 | 0.8871 | 0.8594 | 0.8036 | 0.9146 | 0.9419 | 66 |
| malignant | 0.8438 | 0.9186 | 0.7941 | 0.8182 | 0.9405 | 0.9393 | 0.8155 | 32 |
| normal | 0.8500 | 0.9490 | 0.7727 | 0.8095 | 0.9688 | 0.9577 | 0.8828 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=55 FP=7 TN=45 FN=11
- **malignant**: TP=27 FP=7 TN=79 FN=5
- **normal**: TP=17 FP=5 TN=93 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 55 | 7 | 4 |
| malignant | 4 | 27 | 1 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8871    0.8333    0.8594        66
   malignant     0.7941    0.8438    0.8182        32
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8390       118
   macro avg     0.8180    0.8424    0.8290       118
weighted avg     0.8425    0.8390    0.8398       118

```