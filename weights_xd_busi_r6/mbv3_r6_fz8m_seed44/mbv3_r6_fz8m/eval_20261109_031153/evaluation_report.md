# mbv3_r6_fz8m 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mbv3_r6_fz8m_seed44/mbv3_r6_fz8m/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8301 |
| Accuracy | 0.8559 |
| Macro-AUROC (OvR) | 0.9650 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8573 |
| Balanced Acc | 0.8480 |
| Macro-Sensitivity | 0.8480 |
| Macro-Specificity | 0.9252 |
| Macro-Precision | 0.8238 |
| AUC-PR (Macro) | 0.9356 |
| Cohen's Kappa | 0.7578 |
| MCC | 0.7605 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.91M |
| FLOPs | 465.83M |
| Params | 4.21M |
| Pure-Inference FPS | 1246.8 (0.80 ms/张) |
| End-to-End FPS | 23.3 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8939 | 0.9038 | 0.9219 | 0.9077 | 0.8704 | 0.9572 | 0.9709 | 66 |
| malignant | 0.7500 | 0.9535 | 0.8571 | 0.8000 | 0.9111 | 0.9531 | 0.8826 | 32 |
| normal | 0.9000 | 0.9184 | 0.6923 | 0.7826 | 0.9783 | 0.9847 | 0.9534 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=59 FP=5 TN=47 FN=7
- **malignant**: TP=24 FP=4 TN=82 FN=8
- **normal**: TP=18 FP=8 TN=90 FN=2

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 59 | 4 | 3 |
| malignant | 3 | 24 | 5 |
| normal | 2 | 0 | 18 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.9219    0.8939    0.9077        66
   malignant     0.8571    0.7500    0.8000        32
      normal     0.6923    0.9000    0.7826        20

    accuracy                         0.8559       118
   macro avg     0.8238    0.8480    0.8301       118
weighted avg     0.8654    0.8559    0.8573       118

```