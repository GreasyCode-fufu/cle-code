# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz8m_seed44/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/busi_v1/test`  (共 116 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8264 |
| Accuracy | 0.8534 |
| Macro-AUROC (OvR) | 0.9298 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8555 |
| Balanced Acc | 0.8447 |
| Macro-Sensitivity | 0.8447 |
| Macro-Specificity | 0.9243 |
| Macro-Precision | 0.8240 |
| AUC-PR (Macro) | 0.8955 |
| Cohen's Kappa | 0.7540 |
| MCC | 0.7577 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1154.4 (0.87 ms/张) |
| End-to-End FPS | 20.9 |
| 测试样本数 | 116 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8923 | 0.9020 | 0.9206 | 0.9062 | 0.8679 | 0.9409 | 0.9417 | 65 |
| malignant | 0.7419 | 0.9647 | 0.8846 | 0.8070 | 0.9111 | 0.9104 | 0.8714 | 31 |
| normal | 0.9000 | 0.9062 | 0.6667 | 0.7660 | 0.9775 | 0.9380 | 0.8736 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=5 TN=46 FN=7
- **malignant**: TP=23 FP=3 TN=82 FN=8
- **normal**: TP=18 FP=9 TN=87 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 3 | 4 |
| malignant | 3 | 23 | 5 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9206    0.8923    0.9062        65
   malignant     0.8846    0.7419    0.8070        31
      normal     0.6667    0.9000    0.7660        20

    accuracy                         0.8534       116
   macro avg     0.8240    0.8447    0.8264       116
weighted avg     0.8672    0.8534    0.8555       116

```