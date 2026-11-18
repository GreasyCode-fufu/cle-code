# gmnet_s3 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/gmnet_s3_seed44/gmnet_s3/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8235 |
| Accuracy | 0.8390 |
| Macro-AUROC (OvR) | 0.9041 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8405 |
| Balanced Acc | 0.8432 |
| Macro-Sensitivity | 0.8432 |
| Macro-Specificity | 0.9124 |
| Macro-Precision | 0.8144 |
| AUC-PR (Macro) | 0.8435 |
| Cohen's Kappa | 0.7305 |
| MCC | 0.7330 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.27G |
| FLOPs | 2.54G |
| Params | 7.44M |
| Pure-Inference FPS | 1397.3 (0.72 ms/张) |
| End-to-End FPS | 172.0 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.8654 | 0.8889 | 0.8682 | 0.8182 | 0.9034 | 0.8921 | 66 |
| malignant | 0.7812 | 0.9535 | 0.8621 | 0.8197 | 0.9213 | 0.9159 | 0.8857 | 32 |
| normal | 0.9000 | 0.9184 | 0.6923 | 0.7826 | 0.9783 | 0.8931 | 0.7527 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=7 TN=45 FN=10
- **malignant**: TP=25 FP=4 TN=82 FN=7
- **normal**: TP=18 FP=8 TN=90 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 4 | 6 |
| malignant | 5 | 25 | 2 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8889    0.8485    0.8682        66
   malignant     0.8621    0.7812    0.8197        32
      normal     0.6923    0.9000    0.7826        20

    accuracy                         0.8390       118
   macro avg     0.8144    0.8432    0.8235       118
weighted avg     0.8483    0.8390    0.8405       118

```