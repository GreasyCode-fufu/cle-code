# convnext_egb_scg 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6b/convnext_egb_scg_seed42/convnext_egb_scg/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8170 |
| Accuracy | 0.8390 |
| Macro-AUROC (OvR) | 0.9443 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8380 |
| Balanced Acc | 0.8209 |
| Macro-Sensitivity | 0.8209 |
| Macro-Specificity | 0.9064 |
| Macro-Precision | 0.8195 |
| AUC-PR (Macro) | 0.8856 |
| Cohen's Kappa | 0.7236 |
| MCC | 0.7253 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 4.85G |
| FLOPs | 9.70G |
| Params | 31.28M |
| Pure-Inference FPS | 468.5 (2.13 ms/张) |
| End-to-End FPS | 138.5 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.8269 | 0.8676 | 0.8806 | 0.8600 | 0.9315 | 0.9553 | 66 |
| malignant | 0.7188 | 0.9535 | 0.8519 | 0.7797 | 0.9011 | 0.9295 | 0.8235 | 32 |
| normal | 0.8500 | 0.9388 | 0.7391 | 0.7907 | 0.9684 | 0.9719 | 0.8779 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=9 TN=43 FN=7
- **malignant**: TP=23 FP=4 TN=82 FN=9
- **normal**: TP=17 FP=6 TN=92 FN=3

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 4 | 3 |
| malignant | 6 | 23 | 3 |
| normal | 3 | 0 | 17 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8676    0.8939    0.8806        66
   malignant     0.8519    0.7188    0.7797        32
      normal     0.7391    0.8500    0.7907        20

    accuracy                         0.8390       118
   macro avg     0.8195    0.8209    0.8170       118
weighted avg     0.8416    0.8390    0.8380       118

```