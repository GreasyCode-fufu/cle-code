# mbv3_dce4m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_dce4m_seed42/mbv3_dce4m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8605 |
| Accuracy | 0.8729 |
| Macro-AUROC (OvR) | 0.9478 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8732 |
| Balanced Acc | 0.8626 |
| Macro-Sensitivity | 0.8626 |
| Macro-Specificity | 0.9281 |
| Macro-Precision | 0.8586 |
| AUC-PR (Macro) | 0.9100 |
| Cohen's Kappa | 0.7836 |
| MCC | 0.7837 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 231.68M |
| FLOPs | 463.37M |
| Params | 2.98M |
| Pure-Inference FPS | 1698.5 (0.59 ms/张) |
| End-to-End FPS | 159.7 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.8846 | 0.9077 | 0.9008 | 0.8679 | 0.9372 | 0.9537 | 66 |
| malignant | 0.8438 | 0.9302 | 0.8182 | 0.8308 | 0.9412 | 0.9528 | 0.8760 | 32 |
| normal | 0.8500 | 0.9694 | 0.8500 | 0.8500 | 0.9694 | 0.9536 | 0.9001 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=6 TN=46 FN=7
- **malignant**: TP=27 FP=6 TN=80 FN=5
- **normal**: TP=17 FP=3 TN=95 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 6 | 1 |
| malignant | 3 | 27 | 2 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9077    0.8939    0.9008        66
   malignant     0.8182    0.8438    0.8308        32
      normal     0.8500    0.8500    0.8500        20

    accuracy                         0.8729       118
   macro avg     0.8586    0.8626    0.8605       118
weighted avg     0.8736    0.8729    0.8732       118

```