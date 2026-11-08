# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_dce4m_seed44/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8726 |
| Accuracy | 0.8898 |
| Macro-AUROC (OvR) | 0.9506 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8889 |
| Balanced Acc | 0.8610 |
| Macro-Sensitivity | 0.8610 |
| Macro-Specificity | 0.9328 |
| Macro-Precision | 0.8861 |
| AUC-PR (Macro) | 0.9102 |
| Cohen's Kappa | 0.8087 |
| MCC | 0.8095 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1679.9 (0.60 ms/张) |
| End-to-End FPS | 156.3 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.9394 | 0.8654 | 0.8986 | 0.9185 | 0.9184 | 0.9397 | 0.9596 | 66 |
| malignant | 0.8438 | 0.9535 | 0.8710 | 0.8571 | 0.9425 | 0.9453 | 0.8610 | 32 |
| normal | 0.8000 | 0.9796 | 0.8889 | 0.8421 | 0.9600 | 0.9668 | 0.9100 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=62 FP=7 TN=45 FN=4
- **malignant**: TP=27 FP=4 TN=82 FN=5
- **normal**: TP=16 FP=2 TN=96 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 62 | 4 | 0 |
| malignant | 3 | 27 | 2 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8986    0.9394    0.9185        66
   malignant     0.8710    0.8438    0.8571        32
      normal     0.8889    0.8000    0.8421        20

    accuracy                         0.8898       118
   macro avg     0.8861    0.8610    0.8726       118
weighted avg     0.8894    0.8898    0.8889       118

```