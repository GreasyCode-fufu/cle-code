# mbv3_fz16bn_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16bn_dce4m_seed42/mbv3_fz16bn_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8418 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9379 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8556 |
| Balanced Acc | 0.8346 |
| Macro-Sensitivity | 0.8346 |
| Macro-Specificity | 0.9148 |
| Macro-Precision | 0.8537 |
| AUC-PR (Macro) | 0.8761 |
| Cohen's Kappa | 0.7526 |
| MCC | 0.7534 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 830.4 (1.20 ms/张) |
| End-to-End FPS | 7.4 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8788 | 0.8462 | 0.8788 | 0.8788 | 0.8462 | 0.9272 | 0.9518 | 66 |
| malignant | 0.8750 | 0.9186 | 0.8000 | 0.8358 | 0.9518 | 0.9540 | 0.8312 | 32 |
| normal | 0.7500 | 0.9796 | 0.8824 | 0.8108 | 0.9505 | 0.9327 | 0.8453 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=58 FP=8 TN=44 FN=8
- **malignant**: TP=28 FP=7 TN=79 FN=4
- **normal**: TP=15 FP=2 TN=96 FN=5

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 58 | 7 | 1 |
| malignant | 3 | 28 | 1 |
| normal | 5 | 0 | 15 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8788    0.8788    0.8788        66
   malignant     0.8000    0.8750    0.8358        32
      normal     0.8824    0.7500    0.8108        20

    accuracy                         0.8559       118
   macro avg     0.8537    0.8346    0.8418       118
weighted avg     0.8580    0.8559    0.8556       118

```