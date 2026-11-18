# repvit_m1 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_busi_bk/repvit_m1_seed44/repvit_m1/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.3909 |
| Accuracy | 0.5000 |
| Macro-AUROC (OvR) | 0.5531 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.4780 |
| Balanced Acc | 0.3963 |
| Macro-Sensitivity | 0.3963 |
| Macro-Specificity | 0.6905 |
| Macro-Precision | 0.4092 |
| AUC-PR (Macro) | 0.4102 |
| Cohen's Kappa | 0.0911 |
| MCC | 0.0929 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 1.14G |
| FLOPs | 2.28G |
| Params | 6.41M |
| Pure-Inference FPS | 2040.7 (0.49 ms/张) |
| End-to-End FPS | 185.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.6515 | 0.3654 | 0.5658 | 0.6056 | 0.4524 | 0.5006 | 0.5583 | 66 |
| malignant | 0.4375 | 0.7674 | 0.4118 | 0.4242 | 0.7857 | 0.6061 | 0.4600 | 32 |
| normal | 0.1000 | 0.9388 | 0.2500 | 0.1429 | 0.8364 | 0.5526 | 0.2123 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=43 FP=33 TN=19 FN=23
- **malignant**: TP=14 FP=20 TN=66 FN=18
- **normal**: TP=2 FP=6 TN=92 FN=18

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 43 | 18 | 5 |
| malignant | 17 | 14 | 1 |
| normal | 16 | 2 | 2 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.5658    0.6515    0.6056        66
   malignant     0.4118    0.4375    0.4242        32
      normal     0.2500    0.1000    0.1429        20

    accuracy                         0.5000       118
   macro avg     0.4092    0.3963    0.3909       118
weighted avg     0.4705    0.5000    0.4780       118

```