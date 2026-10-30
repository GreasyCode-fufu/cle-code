# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/repvit_m1_seed43/repvit_m1/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.3898 |
| Accuracy | 0.5085 |
| Macro-AUROC (OvR) | 0.5629 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.4876 |
| Balanced Acc | 0.3907 |
| Macro-Sensitivity | 0.3907 |
| Macro-Specificity | 0.6968 |
| Macro-Precision | 0.3992 |
| AUC-PR (Macro) | 0.4275 |
| Cohen's Kappa | 0.1016 |
| MCC | 0.1037 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 2032.6 (0.49 ms/张) |
| End-to-End FPS | 184.9 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.6970 | 0.3654 | 0.5823 | 0.6345 | 0.4872 | 0.5323 | 0.5702 | 66 |
| malignant | 0.3750 | 0.8372 | 0.4615 | 0.4138 | 0.7826 | 0.6257 | 0.4810 | 32 |
| normal | 0.1000 | 0.8878 | 0.1538 | 0.1212 | 0.8286 | 0.5306 | 0.2314 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=46 FP=33 TN=19 FN=20
- **malignant**: TP=12 FP=14 TN=72 FN=20
- **normal**: TP=2 FP=11 TN=87 FN=18

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 46 | 12 | 8 |
| malignant | 17 | 12 | 3 |
| normal | 16 | 2 | 2 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.5823    0.6970    0.6345        66
   malignant     0.4615    0.3750    0.4138        32
      normal     0.1538    0.1000    0.1212        20

    accuracy                         0.5085       118
   macro avg     0.3992    0.3907    0.3898       118
weighted avg     0.4769    0.5085    0.4876       118

```