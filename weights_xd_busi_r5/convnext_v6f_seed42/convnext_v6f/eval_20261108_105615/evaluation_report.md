# convnext_v6f 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r5/convnext_v6f_seed42/convnext_v6f/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8629 |
| Accuracy | 0.8966 |
| Macro-AUROC (OvR) | 0.9435 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8930 |
| Balanced Acc | 0.8414 |
| Macro-Sensitivity | 0.8414 |
| Macro-Specificity | 0.9334 |
| Macro-Precision | 0.8946 |
| AUC-PR (Macro) | 0.8963 |
| Cohen's Kappa | 0.8163 |
| MCC | 0.8228 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.63G |
| FLOPs | 9.27G |
| Params | 28.25M |
| Pure-Inference FPS | 601.6 (1.66 ms/张) |
| End-to-End FPS | 142.8 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 1.0000 | 0.8431 | 0.8904 | 0.9420 | 1.0000 | 0.9538 | 0.9300 | 65 |
| malignant | 0.7742 | 0.9882 | 0.9600 | 0.8571 | 0.9231 | 0.9658 | 0.9320 | 31 |
| normal | 0.7500 | 0.9688 | 0.8333 | 0.7895 | 0.9490 | 0.9109 | 0.8268 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=65 FP=8 TN=43 FN=0
- **malignant**: TP=24 FP=1 TN=84 FN=7
- **normal**: TP=15 FP=3 TN=93 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 65 | 0 | 0 |
| malignant | 4 | 24 | 3 |
| normal | 4 | 1 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8904    1.0000    0.9420        65
   malignant     0.9600    0.7742    0.8571        31
      normal     0.8333    0.7500    0.7895        20

    accuracy                         0.8966       116
   macro avg     0.8946    0.8414    0.8629       116
weighted avg     0.8992    0.8966    0.8930       116

```