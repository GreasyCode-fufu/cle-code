# mobilenet_v3_large 评估报告

- 权重: `/data/cle/auto/ZCode/Classification/weights_xd_busi_r6/mobilenet_v3_large_seed42/mobilenet_v3_large/model_best.pth`
- 测试集: `./datasets/busi_v1/val`  (共 118 张)
- 归一化: run_meta.json(imagenet)  mean=[0.485, 0.456, 0.406]
- img_size: 224

## 主指标

| 指标 | 值 |
|---|---|
| **Macro-F1** | 0.8165 |
| Accuracy | 0.8305 |
| Macro-AUROC (OvR) | 0.9334 |

## 补充总体指标

| 指标 | 值 |
|---|---|
| Weighted-F1 | 0.8311 |
| Balanced Acc | 0.8203 |
| Macro-Sensitivity | 0.8203 |
| Macro-Specificity | 0.9041 |
| Macro-Precision | 0.8132 |
| AUC-PR (Macro) | 0.8897 |
| Cohen's Kappa | 0.7126 |
| MCC | 0.7130 |

## 模型复杂度与推理效率

| 指标 | 值 |
|---|---|
| MACs | 232.96M |
| FLOPs | 465.92M |
| Params | 4.21M |
| Pure-Inference FPS | 1571.0 (0.64 ms/张) |
| End-to-End FPS | 176.1 |
| 测试样本数 | 118 |

## 每类指标 (One-vs-Rest)

| 类 | Sensitivity | Specificity | Precision | F1 | NPV | AUROC | AUC-PR | support |
|---|---|---|---|---|---|---|---|---|
| benign | 0.8485 | 0.8462 | 0.8750 | 0.8615 | 0.8148 | 0.9207 | 0.9262 | 66 |
| malignant | 0.8125 | 0.9070 | 0.7647 | 0.7879 | 0.9286 | 0.9426 | 0.8697 | 32 |
| normal | 0.8000 | 0.9592 | 0.8000 | 0.8000 | 0.9592 | 0.9367 | 0.8734 | 20 |

### 每类 TP/FP/TN/FN

- **benign**: TP=56 FP=8 TN=44 FN=10
- **malignant**: TP=26 FP=8 TN=78 FN=6
- **normal**: TP=16 FP=4 TN=94 FN=4

## 混淆矩阵 (行=真实, 列=预测)

| | benign | malignant | normal |
|---|---|---|---|
| benign | 56 | 8 | 2 |
| malignant | 4 | 26 | 2 |
| normal | 4 | 0 | 16 |

## Classification Report (sklearn)

```
              precision    recall  f1-score   support

      benign     0.8750    0.8485    0.8615        66
   malignant     0.7647    0.8125    0.7879        32
      normal     0.8000    0.8000    0.8000        20

    accuracy                         0.8305       118
   macro avg     0.8132    0.8203    0.8165       118
weighted avg     0.8324    0.8305    0.8311       118

```