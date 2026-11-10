# lsnet_s 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/lsnet_s_seed42/lsnet_s/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.6686 |
| Accuracy | 0.6949 |
| Macro-AUROC (OvR) | 0.8441 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.7021 |
| Balanced Acc | 0.7020 |
| Macro-Sensitivity | 0.7020 |
| Macro-Specificity | 0.8411 |
| Macro-Precision | 0.6706 |
| AUC-PR (Macro) | 0.6977 |
| Cohen's Kappa | 0.5051 |
| MCC | 0.5143 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 545.67M |
| FLOPs | 1.09G |
| Params | 15.70M |
| Pure-Inference FPS | 1451.0 (0.69 ms/张) |
| End-to-End FPS | 172.4 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.7121 | 0.7885 | 0.8103 | 0.7581 | 0.6833 | 0.8000 | 0.8604 | 66 |
| malignant | 0.5938 | 0.9186 | 0.7308 | 0.6552 | 0.8587 | 0.8614 | 0.7372 | 32 |
| normal | 0.8000 | 0.8163 | 0.4706 | 0.5926 | 0.9524 | 0.8709 | 0.4956 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=47 FP=11 TN=41 FN=19
- **malignant**: TP=19 FP=7 TN=79 FN=13
- **normal**: TP=16 FP=18 TN=80 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 47 | 6 | 13 |
| malignant | 8 | 19 | 5 |
| normal | 3 | 1 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8103    0.7121    0.7581        66
   malignant     0.7308    0.5938    0.6552        32
      normal     0.4706    0.8000    0.5926        20

    accuracy                         0.6949       118
   macro avg     0.6706    0.7020    0.6686       118
weighted avg     0.7312    0.6949    0.7021       118

```