# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/gmnet_s3_seed44/gmnet_s3/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.7967 |
| Accuracy | 0.8276 |
| Macro-AUROC (OvR) | 0.8979 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8269 |
| Balanced Acc | 0.7947 |
| Macro-Sensitivity | 0.7947 |
| Macro-Specificity | 0.9007 |
| Macro-Precision | 0.8005 |
| AUC-PR (Macro) | 0.8533 |
| Cohen's Kappa | 0.7031 |
| MCC | 0.7037 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1441.0 (0.69 ms/张) |
| End-to-End FPS | 165.3 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8923 | 0.8235 | 0.8657 | 0.8788 | 0.8571 | 0.9095 | 0.8915 | 65 |
| malignant | 0.7419 | 0.9412 | 0.8214 | 0.7797 | 0.9091 | 0.9192 | 0.8867 | 31 |
| normal | 0.7500 | 0.9375 | 0.7143 | 0.7317 | 0.9474 | 0.8651 | 0.7817 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=9 TN=42 FN=7
- **malignant**: TP=23 FP=5 TN=80 FN=8
- **normal**: TP=15 FP=6 TN=90 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 4 | 3 |
| malignant | 5 | 23 | 3 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8657    0.8923    0.8788        65
   malignant     0.8214    0.7419    0.7797        31
      normal     0.7143    0.7500    0.7317        20

    accuracy                         0.8276       116
   macro avg     0.8005    0.7947    0.7967       116
weighted avg     0.8277    0.8276    0.8269       116

```