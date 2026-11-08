# mbv3_fz16_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_fz16_dce4m_seed42/mbv3_fz16_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8521 |
| Accuracy | 0.8644 |
| Macro-AUROC (OvR) | 0.9497 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8644 |
| Balanced Acc | 0.8521 |
| Macro-Sensitivity | 0.8521 |
| Macro-Specificity | 0.9217 |
| Macro-Precision | 0.8521 |
| AUC-PR (Macro) | 0.9090 |
| Cohen's Kappa | 0.7682 |
| MCC | 0.7682 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1699.2 (0.59 ms/张) |
| End-to-End FPS | 162.9 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.8654 | 0.8939 | 0.8939 | 0.8654 | 0.9376 | 0.9563 | 66 |
| malignant | 0.8125 | 0.9302 | 0.8125 | 0.8125 | 0.9302 | 0.9430 | 0.8485 | 32 |
| normal | 0.8500 | 0.9694 | 0.8500 | 0.8500 | 0.9694 | 0.9684 | 0.9221 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=7 TN=45 FN=7
- **malignant**: TP=26 FP=6 TN=80 FN=6
- **normal**: TP=17 FP=3 TN=95 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 6 | 1 |
| malignant | 4 | 26 | 2 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8939    0.8939    0.8939        66
   malignant     0.8125    0.8125    0.8125        32
      normal     0.8500    0.8500    0.8500        20

    accuracy                         0.8644       118
   macro avg     0.8521    0.8521    0.8521       118
weighted avg     0.8644    0.8644    0.8644       118

```