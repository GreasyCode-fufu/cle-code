# resnet34 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/resnet34_seed43/resnet34/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8247 |
| Accuracy | 0.8305 |
| Macro-AUROC (OvR) | 0.9429 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8309 |
| Balanced Acc | 0.8266 |
| Macro-Sensitivity | 0.8266 |
| Macro-Specificity | 0.9011 |
| Macro-Precision | 0.8230 |
| AUC-PR (Macro) | 0.9073 |
| Cohen's Kappa | 0.7114 |
| MCC | 0.7115 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 3.68G |
| FLOPs | 7.36G |
| Params | 21.29M |
| Pure-Inference FPS | 1975.6 (0.51 ms/张) |
| End-to-End FPS | 185.6 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.8269 | 0.8615 | 0.8550 | 0.8113 | 0.9228 | 0.9390 | 66 |
| malignant | 0.7812 | 0.9070 | 0.7576 | 0.7692 | 0.9176 | 0.9437 | 0.8989 | 32 |
| normal | 0.8500 | 0.9694 | 0.8500 | 0.8500 | 0.9694 | 0.9622 | 0.8841 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=9 TN=43 FN=10
- **malignant**: TP=25 FP=8 TN=78 FN=7
- **normal**: TP=17 FP=3 TN=95 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 7 | 3 |
| malignant | 7 | 25 | 0 |
| normal | 2 | 1 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8615    0.8485    0.8550        66
   malignant     0.7576    0.7812    0.7692        32
      normal     0.8500    0.8500    0.8500        20

    accuracy                         0.8305       118
   macro avg     0.8230    0.8266    0.8247       118
weighted avg     0.8314    0.8305    0.8309       118

```