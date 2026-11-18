# convnext_svd 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_ep/convnext_svd_seed44/convnext_svd/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8628 |
| Accuracy | 0.8879 |
| Macro-AUROC (OvR) | 0.9490 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8879 |
| Balanced Acc | 0.8650 |
| Macro-Sensitivity | 0.8650 |
| Macro-Specificity | 0.9382 |
| Macro-Precision | 0.8633 |
| AUC-PR (Macro) | 0.9052 |
| Cohen's Kappa | 0.8081 |
| MCC | 0.8088 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.41G |
| FLOPs | 6.83G |
| Params | 18.51M |
| Pure-Inference FPS | 428.2 (2.34 ms/张) |
| End-to-End FPS | 130.7 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9385 | 0.9020 | 0.9242 | 0.9313 | 0.9200 | 0.9623 | 0.9564 | 65 |
| malignant | 0.8065 | 0.9647 | 0.8929 | 0.8475 | 0.9318 | 0.9488 | 0.9257 | 31 |
| normal | 0.8500 | 0.9479 | 0.7727 | 0.8095 | 0.9681 | 0.9359 | 0.8336 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=61 FP=5 TN=46 FN=4
- **malignant**: TP=25 FP=3 TN=82 FN=6
- **normal**: TP=17 FP=5 TN=91 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 61 | 3 | 1 |
| malignant | 2 | 25 | 4 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9242    0.9385    0.9313        65
   malignant     0.8929    0.8065    0.8475        31
      normal     0.7727    0.8500    0.8095        20

    accuracy                         0.8879       116
   macro avg     0.8633    0.8650    0.8628       116
weighted avg     0.8897    0.8879    0.8879       116

```